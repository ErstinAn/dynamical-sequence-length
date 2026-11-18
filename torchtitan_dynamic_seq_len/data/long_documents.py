"""Read token windows within single documents from a verified local artifact."""

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, default_collate
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

from torchtitan_dynamic_seq_len.data.dynamic_seq_bs_loader import DynamicSeqBSDataLoader


class LongDocumentDataset(IterableDataset, Stateful):
    """Drop short tails rather than joining unrelated documents in a window.

    Full documents remain in storage. The training token budget must count emitted
    label tokens, because the discarded tail depends on the current window length.
    """

    def __init__(self, root, *, split, seq_len, dp_rank=0, dp_world_size=1, infinite=False):
        self.root = Path(root)
        encoded = (self.root / "manifest.json").read_bytes()
        self.manifest_hash = hashlib.sha256(encoded).hexdigest()
        manifest = json.loads(encoded)
        verified = json.loads((self.root / "verification.json").read_text())
        if not manifest["complete"] or not verified["passed"] or verified["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Long-document artifact has not passed verification")
        if split not in {"train", "validation"} or not 0 <= dp_rank < dp_world_size:
            raise ValueError("Invalid split or data-parallel rank")
        self.files = [f for f in manifest["files"] if f["split"] == split]
        self.indexes = [np.load(self.root / f["index"]["path"], mmap_mode="r") for f in self.files]
        self.starts = np.cumsum([0] + [len(index) for index in self.indexes])
        self.document_order = None
        self.document_order_index = 0
        order_entry = manifest.get("document_orders", {}).get(split)
        if order_entry is not None:
            order_path = self.root / order_entry["path"]
            if hashlib.sha256(order_path.read_bytes()).hexdigest() != order_entry["sha256"]:
                raise ValueError("Document order checksum mismatch")
            order = np.load(order_path, mmap_mode="r", allow_pickle=False)
            count = int(self.starts[-1])
            if (order.ndim != 1 or order.dtype.kind not in "ui" or len(order) != count
                    or not np.array_equal(np.sort(order), np.arange(count))):
                raise ValueError("Document order must contain every split document exactly once")
            self.document_order = order
        self.minimum_length = manifest["selection"]["minimum_llama3_tokens"]
        self.tokenizer_sha256 = manifest["tokenizer_sha256"]
        self.vocab_size = manifest["tokenizer"]["vocab_size"]
        self.bos = manifest["tokenizer"]["bos_id"]
        self.eos = manifest["tokenizer"]["eos_id"]
        if not self.files or seq_len <= 0 or seq_len % 8:
            raise ValueError("Empty split or invalid sequence length")
        if int(self.starts[-1]) <= dp_rank:
            raise ValueError("No documents are assigned to this data-parallel rank")
        self.seq_len = seq_len
        self.dp_rank, self.dp_world_size = dp_rank, dp_world_size
        self.infinite = infinite
        self.file_idx = self.row_idx = self.offset = self.epoch = 0
        self.emitted_label_tokens = self.dropped_tail_tokens = 0
        self._mmap = None
        self._mmap_file = None
        self._token_maps = OrderedDict()

    def _advance_document(self):
        if self.document_order is not None:
            self.document_order_index += 1
        else:
            self.row_idx += 1
        self.offset = 0

    def _locate_document(self):
        """Resolve the next owned document without changing its token offset.

        An explicit order is shared by all ranks. Ownership uses the position
        in that order, so shuffling occurs before the data-parallel split.
        """
        while True:
            if self.document_order is not None:
                if self.document_order_index >= len(self.document_order):
                    if not self.infinite:
                        return None
                    self.document_order_index = self.offset = 0
                    self.epoch += 1
                slot = self.document_order_index
                if slot % self.dp_world_size != self.dp_rank:
                    self._advance_document()
                    continue
                row = int(self.document_order[slot])
                self.file_idx = int(np.searchsorted(self.starts, row, side="right") - 1)
                self.row_idx = row - int(self.starts[self.file_idx])
            else:
                if self.file_idx >= len(self.files):
                    if not self.infinite:
                        return None
                    self.file_idx = self.row_idx = self.offset = 0
                    self.epoch += 1
                if self.row_idx >= len(self.indexes[self.file_idx]):
                    self.file_idx += 1
                    self.row_idx = self.offset = 0
                    continue
                row = int(self.starts[self.file_idx]) + self.row_idx
                if row % self.dp_world_size != self.dp_rank:
                    self._advance_document()
                    continue
            return tuple(map(int, self.indexes[self.file_idx][self.row_idx]))

    def _token_map(self):
        # Global document shuffling revisits storage shards frequently.
        # Retain a bounded number of mappings; no token arrays are copied here.
        mapped = self._token_maps.pop(self.file_idx, None)
        if mapped is None:
            mapped = np.memmap(
                self.root / self.files[self.file_idx]["token_ids"]["path"],
                mode="r", dtype="<u4",
            )
        self._token_maps[self.file_idx] = mapped
        if len(self._token_maps) > 256:
            self._token_maps.popitem(last=False)
        self._mmap, self._mmap_file = mapped, self.file_idx
        return mapped

    def __iter__(self):
        produced = bool(self.file_idx or self.row_idx or self.offset or self.document_order_index)
        while True:
            epoch = self.epoch
            location = self._locate_document()
            if location is None:
                return
            if self.epoch != epoch:
                if not produced:
                    raise RuntimeError("No single-document windows available for this rank and sequence length")
                produced = False
            start, size = location
            # BOS + N raw tokens + EOS produce N+1 document-local labels.
            remaining = size + 1 - self.offset
            length = self.seq_len
            if remaining < length:
                self.dropped_tail_tokens += max(0, remaining)
                self._advance_document()
                continue
            self._token_map()
            # Read only this window. Positions 0 and size+1 are virtual BOS/EOS.
            positions = np.arange(self.offset, self.offset + length + 1, dtype=np.int64)
            values = np.empty(length + 1, dtype=np.int64)
            raw_mask = (positions > 0) & (positions <= size)
            values[raw_mask] = self._mmap[start + positions[raw_mask] - 1]
            values[positions == 0] = self.bos
            values[positions == size + 1] = self.eos
            self.offset += length
            self.emitted_label_tokens += length
            produced = True
            yield {"input": torch.from_numpy(values[:-1].copy()),
                   "positions": torch.arange(length, dtype=torch.long)}, torch.from_numpy(values[1:].copy())

    def state_dict(self):
        state = {"manifest_sha256": self.manifest_hash, "dp_rank": self.dp_rank,
                "dp_world_size": self.dp_world_size,
                **{k: getattr(self, k) for k in ["file_idx", "row_idx", "offset", "epoch",
                                                 "emitted_label_tokens", "dropped_tail_tokens"]}}
        if self.document_order is not None:
            state["document_order_index"] = self.document_order_index
        return state

    def load_state_dict(self, state):
        if (state["manifest_sha256"], state["dp_rank"], state["dp_world_size"]) != (
            self.manifest_hash, self.dp_rank, self.dp_world_size):
            raise ValueError("Checkpoint belongs to another dataset or data-parallel layout")
        for key in ["file_idx", "row_idx", "offset", "epoch", "emitted_label_tokens", "dropped_tail_tokens"]:
            setattr(self, key, state[key])
        if self.document_order is not None:
            self.document_order_index = state["document_order_index"]
            if not 0 <= self.document_order_index <= len(self.document_order):
                raise ValueError("Checkpoint document order position is invalid")
        self._mmap = self._mmap_file = None
        self._token_maps.clear()


class LongDocumentBatchDataset(LongDocumentDataset):
    """Read the current microbatch size before consuming document windows."""

    def __iter__(self):
        windows = super().__iter__()
        while True:
            size = self.microbatch_size
            samples = list(islice(windows, size))
            if not samples:
                return
            yield default_collate(samples)
            if len(samples) < size:
                return


class LongDocumentDataLoader(ParallelAwareDataloader):
    dataset_type = LongDocumentDataset
    dynamic_batches = False

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTextDataLoader.Config):
        dataset: str = "fineweb_edu_long"
        split: str = "train"
        expected_manifest_sha256: str = ""
        packing: str = "none"
        num_workers: int = 0
        prefetch_factor: int | None = None
        persistent_workers: bool = False

    def __init__(self, config, *, dp_world_size, dp_rank, tokenizer, seq_len,
                 local_batch_size, snapshot_every_n_steps=1, **kwargs):
        if config.num_workers or config.prefetch_factor is not None or config.persistent_workers:
            raise ValueError("Long-document loader requires synchronous reads for scheduler changes")
        if config.packing not in {"none", "stream"}:
            raise ValueError("packing must be none or stream")
        dataset_type = self.dataset_type
        if config.packing == "stream":
            from torchtitan_dynamic_seq_len.data.packed_documents import (
                PackedDocumentDataset, PackedDocumentBatchDataset,
            )
            dataset_type = PackedDocumentBatchDataset if self.dynamic_batches else PackedDocumentDataset
        dataset = dataset_type(config.dataset_path, split=config.split, seq_len=seq_len,
                                      dp_rank=dp_rank, dp_world_size=dp_world_size,
                                      infinite=config.infinite)
        if config.expected_manifest_sha256 and dataset.manifest_hash != config.expected_manifest_sha256:
            raise ValueError("Long-document artifact differs from the experiment manifest")
        path = Path(tokenizer.tokenizer_path) / "tokenizer.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != dataset.tokenizer_sha256:
            raise ValueError("Training tokenizer differs from cached token IDs")
        if tokenizer.get_vocab_size() != dataset.vocab_size:
            raise ValueError("Training vocabulary differs from dataset vocabulary")
        self.max_seq_len = seq_len
        if self.dynamic_batches:
            dataset.microbatch_size = local_batch_size
        super().__init__(dataset, dp_rank=dp_rank, dp_world_size=dp_world_size,
                         batch_size=None if self.dynamic_batches else local_batch_size,
                         num_workers=0, pin_memory=config.pin_memory,
                         snapshot_every_n_steps=1 if self.dynamic_batches else snapshot_every_n_steps)

    def set_seq_len(self, length):
        if not 0 < length <= self.max_seq_len or length % 8:
            raise ValueError("Sequence length must be a positive multiple of 8 within model capacity")
        self.dataset.seq_len = length


class LongDocumentBatchDataLoader(LongDocumentDataLoader):
    """Use document-local tokens with the joint and loss/gradient schedulers."""

    dataset_type = LongDocumentBatchDataset
    dynamic_batches = True

    @dataclass(kw_only=True, slots=True)
    class Config(DynamicSeqBSDataLoader.Config):
        dataset: str = "fineweb_edu_long"
        split: str = "train"
        expected_manifest_sha256: str = ""
        packing: str = "none"

    def set_batch_shape(self, *, seq_len, microbatch_size):
        if type(microbatch_size) is not int or microbatch_size < 1:
            raise ValueError("microbatch_size must be a positive integer")
        self.set_seq_len(seq_len)
        self.dataset.microbatch_size = microbatch_size
