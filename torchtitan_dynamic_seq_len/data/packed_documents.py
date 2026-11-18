"""Pack document streams without discarding short documents or their tails."""

import numpy as np
import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentDataset

PACKING_PROTOCOL = "stream-v1-bos-eos-ignore-boundary-rebase-fragment"


class PackedDocumentDataset(LongDocumentDataset):
    """Keep the cursor at the next input token, including the label lookahead."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not any(((int(self.starts[i]) + np.arange(len(index))) % self.dp_world_size
                    == self.dp_rank).any() for i, index in enumerate(self.indexes)):
            raise ValueError("Packed stream has no documents for this rank")
        self.last_fragments = []
        self.valid_labels = self.boundary_labels = self.padding_positions = 0

    def _cursor(self):
        return self.file_idx, self.row_idx, self.offset, self.epoch, self.document_order_index

    def _restore_cursor(self, cursor):
        self.file_idx, self.row_idx, self.offset, self.epoch, self.document_order_index = cursor

    def _read(self, count):
        values, positions, fragments = [], [], []
        remaining = count
        while remaining:
            location = self._locate_document()
            if location is None:
                break
            start, size = location
            available = size + 2 - self.offset
            if available == 0:
                self._advance_document()
                continue
            take = min(available, remaining)
            self._token_map()
            pos = np.arange(self.offset, self.offset + take, dtype=np.int64)
            ids = np.full(take, self.eos, dtype=np.int64)
            raw = (pos > 0) & (pos <= size)
            ids[raw] = self._mmap[start + pos[raw] - 1]
            ids[pos == 0] = self.bos
            fragments.append(dict(file=self.files[self.file_idx]["text"]["path"],
                                  row=self.row_idx, epoch=self.epoch,
                                  document_token_offset=self.offset,
                                  window_offset=count - remaining, length=take))
            values.append(ids)
            positions.append(pos)
            self.offset += take
            remaining -= take
        if not values:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), fragments
        return np.concatenate(values), np.concatenate(positions), fragments

    def __iter__(self):
        while True:
            length = self.seq_len
            ids, positions, fragments = self._read(length)
            if len(ids) == 0:
                return
            cursor = self._cursor()
            successor, successor_pos, lookahead = self._read(1)
            self._restore_cursor(cursor)
            targets = np.concatenate((ids[1:], successor))
            target_pos = np.concatenate((positions[1:], successor_pos))
            labels = np.full(length, IGNORE_INDEX, dtype=np.int64)
            labels[:len(targets)] = np.where(target_pos == 0, IGNORE_INDEX, targets)
            # EOS-only finite tails have no prediction targets.
            if not np.any(labels != IGNORE_INDEX):
                if self.infinite:
                    raise ValueError("Packed window has no valid labels")
                return
            real = len(ids)
            inputs = np.full(length, self.eos, dtype=np.int64)
            inputs[:real] = ids
            normalized = np.zeros(length, dtype=np.int64)
            normalized[:real] = positions
            first_fragment = np.cumsum(positions == 0) == 0
            normalized[:real] -= first_fragment * positions[0]
            self.last_fragments = fragments
            self.last_lookahead = lookahead
            self.valid_labels += int(np.count_nonzero(labels != IGNORE_INDEX))
            self.boundary_labels += int(np.count_nonzero(target_pos == 0))
            self.padding_positions += length - real
            self.emitted_label_tokens += length
            yield {"input": torch.from_numpy(inputs),
                   "positions": torch.from_numpy(normalized)}, torch.from_numpy(labels)

    def state_dict(self):
        return {**super().state_dict(), "packing_protocol": PACKING_PROTOCOL,
                "infinite": self.infinite,
                "split": self.files[0]["split"],
                **{k: getattr(self, k) for k in
                   ("valid_labels", "boundary_labels", "padding_positions")}}

    def load_state_dict(self, state):
        if (state.get("packing_protocol") != PACKING_PROTOCOL
                or state.get("infinite") != self.infinite
                or state.get("split") != self.files[0]["split"]):
            raise ValueError("Checkpoint packing protocol or split differs")
        super().load_state_dict(state)
        for key in ("valid_labels", "boundary_labels", "padding_positions"):
            setattr(self, key, state[key])


class PackedDocumentBatchDataset(PackedDocumentDataset):
    def __iter__(self):
        from itertools import islice
        from torch.utils.data import default_collate

        windows = super().__iter__()
        while True:
            samples = list(islice(windows, self.microbatch_size))
            if not samples:
                return
            yield default_collate(samples)
