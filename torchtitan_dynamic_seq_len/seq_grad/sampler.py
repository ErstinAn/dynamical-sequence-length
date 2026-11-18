"""Independent random packed probes sharing only read-only source storage."""

import copy
import hashlib
from itertools import islice

import numpy as np
import torch
from torch.utils.data import default_collate

from torchtitan_dynamic_seq_len.data.dynamic_seq_loader import DynamicSeqDataset
from torchtitan_dynamic_seq_len.data.packed_documents import PACKING_PROTOCOL, PackedDocumentDataset
from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentDataset
from torchtitan_dynamic_seq_len.data.seq_grad_windows import crop_batch_at_offsets


def stream_seed(seed, step, length, sample=0, purpose="data"):
    value = f"seq-grad-v2:{seed}:{step}:{length}:{sample}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


class PairedWindowProbeLoader:
    """One independently sampled H-parent batch per event, shared by every s.

    H uses all its valid targets. Each candidate uses one uniform contiguous
    window per parent, including the parent's aligned next-token labels.
    The CPU parent cache is released at the event boundary, also on failure.
    """

    def __init__(self, base, reference_length):
        self.base, self.reference_length = base, reference_length
        self.identity = {**base.identity, "protocol": "paired-random-window-v1",
                         "parent_protocol": base.identity["protocol"],
                         "reference_length": reference_length,
                         "reference_targets": "all_valid_targets_in_full_parent",
                         "window_start": "uniform_integer_0_through_H_minus_s_per_parent",
                         "window_seed": "seed,step,length,global_sample_index,paired-window-v1",
                         "positions": "rebase_first_fragment_preserve_document_resets"}
        self.clear_event()

    def clear_event(self):
        self._key, self._parents = None, None

    def sample_batch(self, *, length, global_sequences, local_batch_size, rank, degree, seed, step):
        if not 0 < length <= self.reference_length:
            raise ValueError("Paired window length must lie within H")
        key = (global_sequences, local_batch_size, rank, degree, seed, step)
        if self._key != key:
            self.clear_event()
            self._parents = self.base.sample_batch(
                length=self.reference_length, global_sequences=global_sequences,
                local_batch_size=local_batch_size, rank=rank, degree=degree, seed=seed, step=step)
            self._key = key
        if length == self.reference_length:
            return self._parents
        result, local_index = [], 0
        for inputs, labels in self._parents:
            offsets = [int(np.random.default_rng(stream_seed(
                seed, step, length, rank + (local_index + j) * degree,
                "paired-window-v1")).integers(self.reference_length - length + 1))
                for j in range(labels.shape[0])]
            result.append(crop_batch_at_offsets(inputs, labels, length, offsets))
            local_index += labels.shape[0]
        return result


class SingleDocumentProbeLoader:
    """Uniform valid H+1-token spans, wholly inside raw documents longer than H.

    Documents are weighted by their number of valid parent starts (raw length-H).
    These monitor-only reads do not add BOS/EOS, pack documents, or mask labels.
    """

    def __init__(self, source, reference_length):
        if not isinstance(source, LongDocumentDataset) or any(f["split"] != "train" for f in source.files):
            raise ValueError("Single-document probes require indexed training documents")
        self.source, self.reference_length = source, reference_length
        self.ends = [np.cumsum(np.maximum(np.asarray(index[:, 1], dtype=np.int64) - reference_length, 0), dtype=np.int64)
                     for index in source.indexes]
        self.file_ends = np.cumsum([int(ends[-1]) for ends in self.ends])
        if not len(self.file_ends) or self.file_ends[-1] <= 0:
            raise ValueError("No documents have more than H raw tokens")
        self.identity = dict(protocol="uniform-single-document-H-plus-one-window-v1",
                             manifest_sha256=source.manifest_hash, tokenizer_sha256=source.tokenizer_sha256,
                             minimum_raw_document_tokens=reference_length + 1,
                             eligible_documents=sum(int(np.count_nonzero(index[:, 1] > reference_length)) for index in source.indexes),
                             population_parent_starts=int(self.file_ends[-1]),
                             document_weight="number_of_valid_H_plus_one_token_windows",
                             special_tokens="raw_tokens_only_no_inserted_bos_eos",
                             attention="causal_only", packing="none")

    def locate(self, seed):
        position = int(np.random.default_rng(seed).integers(int(self.file_ends[-1])))
        file = int(np.searchsorted(self.file_ends, position, side="right"))
        position -= int(self.file_ends[file - 1]) if file else 0
        row = int(np.searchsorted(self.ends[file], position, side="right"))
        position -= int(self.ends[file][row - 1]) if row else 0
        return file, row, position

    def _sample(self, seed):
        file, row, offset = self.locate(seed)
        start, size = map(int, self.source.indexes[file][row])
        h = self.reference_length
        if not size > h or not 0 <= offset <= size - h - 1:
            raise ValueError("Parent and its next-token target must remain inside one document")
        tokens = np.memmap(self.source.root / self.source.files[file]["token_ids"]["path"], mode="r", dtype="<u4")
        span = torch.from_numpy(np.array(tokens[start+offset:start+offset+h+1], dtype=np.int64))
        return {"input": span[:-1].clone(), "positions": torch.arange(h)}, span[1:].clone()

    def sample_batch(self, *, length, global_sequences, local_batch_size, rank, degree, seed, step):
        if length != self.reference_length:
            raise ValueError("Sample H parents, then use paired windows for candidate lengths")
        samples = (self._sample(stream_seed(seed, step, length, i, "single-document-parent-v1"))
                   for i in range(rank, global_sequences, degree))
        return [default_collate(list(islice(samples, local_batch_size)))
                for _ in range(global_sequences // (degree * local_batch_size))]


class PackedProbeLoader:
    """Uniform random token starts in the production packed training population.

    Each global sample index selects a new start, with replacement. The normal
    packer reads forward, preserving BOS/EOS, ignored boundary labels and rebased
    fragment positions. No document length filter or parent-prefix crop is used.
    """

    def __init__(self, source):
        if not isinstance(source, PackedDocumentDataset):
            raise ValueError("Packed probes require the production stream packer")
        if any(f["split"] != "train" for f in source.files):
            raise ValueError("Probe population must use the training split")
        self.source = source
        self.ends = [np.cumsum(index[:, 1] + 2, dtype=np.int64) for index in source.indexes]
        self.file_ends = np.cumsum([int(ends[-1]) for ends in self.ends])
        self.identity = dict(protocol="uniform-packed-token-start-v1", packing=PACKING_PROTOCOL,
                             manifest_sha256=source.manifest_hash,
                             tokenizer_sha256=source.tokenizer_sha256,
                             population_tokens=int(self.file_ends[-1]))

    def _sample(self, length, seed):
        offset = int(np.random.default_rng(seed).integers(int(self.file_ends[-1])))
        file_index = int(np.searchsorted(self.file_ends, offset, side="right"))
        offset -= int(self.file_ends[file_index - 1]) if file_index else 0
        row = int(np.searchsorted(self.ends[file_index], offset, side="right"))
        offset -= int(self.ends[file_index][row - 1]) if row else 0
        # Share immutable index/token storage, never the training cursor or buffers.
        stream = copy.copy(self.source)
        stream.dp_rank, stream.dp_world_size = 0, 1
        stream.file_idx, stream.row_idx, stream.offset = file_index, row, offset
        stream.epoch = 0
        stream.seq_len, stream.infinite = length, True
        stream._mmap = stream._mmap_file = None
        stream.last_fragments = []
        stream.valid_labels = stream.boundary_labels = stream.padding_positions = 0
        stream.emitted_label_tokens = stream.dropped_tail_tokens = 0
        return next(PackedDocumentDataset.__iter__(stream))

    def sample_batch(self, *, length, global_sequences, local_batch_size, rank, degree, seed, step):
        indices = range(rank, global_sequences, degree)
        samples = (self._sample(length, stream_seed(seed, step, length, i)) for i in indices)
        return [default_collate(list(islice(samples, local_batch_size)))
                for _ in range(global_sequences // (degree * local_batch_size))]


class TextProbeLoader:
    """Keep the existing text packer available for small fixtures and old datasets."""

    def __init__(self, source):
        if not isinstance(source, DynamicSeqDataset):
            raise ValueError("Unsupported independent probe source")
        self.source = source
        self.identity = dict(protocol="independent-shuffled-text-stream-v1",
                             dataset=source.dataset_name,
                             packing="production-document-local-labels")

    def sample_batch(self, *, length, global_sequences, local_batch_size, rank, degree, seed, step):
        stream = copy.copy(self.source)
        # The data object can own iterator state even though its storage is read-only.
        original = copy.deepcopy(self.source._original_data)
        stream._original_data = original.shuffle(seed=stream_seed(seed, step, length, rank))
        stream._data = stream._original_data
        stream._sample_idx = stream._epoch = 0
        stream._resumed = False
        stream._inputs_buffer, stream._labels_buffer, stream._positions_buffer = [], [], []
        stream.seq_len, stream.infinite = length, True
        samples = iter(stream)
        return [default_collate(list(islice(samples, local_batch_size)))
                for _ in range(global_sequences // (degree * local_batch_size))]
