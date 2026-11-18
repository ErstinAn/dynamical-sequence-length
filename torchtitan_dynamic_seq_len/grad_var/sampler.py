"""Indexed sampling of continuous sequences inside individual documents."""

import numpy as np
import torch
from torch.utils.data import default_collate

from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentDataset


class ParentWindowSampler:
    identity = "uniform_parent_prefix_v1"

    def __init__(self, dataset: LongDocumentDataset, parent_length: int, seed: int):
        if not isinstance(dataset, LongDocumentDataset):
            raise ValueError("X probes require the verified long-document token cache")
        self.dataset = dataset  # Read-only indexes/storage; never read its iterator.
        self.parent_length, self.seed = parent_length, seed
        self.cumulative = [
            np.cumsum((index[:, 1].astype(np.int64) + 1) // parent_length)
            for index in dataset.indexes
        ]
        self.file_ends = np.cumsum(
            [int(counts[-1]) if len(counts) else 0 for counts in self.cumulative]
        )
        self.pool_size = int(self.file_ends[-1])
        if not self.pool_size:
            raise ValueError("No complete parent windows after label construction")
        self._maps = {}

    def sample_groups(self, completed_updates, num_groups, group_sequences):
        # Stateless event replay; neither NumPy's nor Torch's global RNG advances.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, completed_updates]))
        return rng.integers(self.pool_size, size=(num_groups, group_sequences))

    def read(self, anchor, length):
        if not 0 < length <= self.parent_length or not 0 <= anchor < self.pool_size:
            raise ValueError("Invalid parent index or prefix length")
        file = int(np.searchsorted(self.file_ends, anchor, side="right"))
        local = int(anchor) - (int(self.file_ends[file - 1]) if file else 0)
        counts = self.cumulative[file]
        row = int(np.searchsorted(counts, local, side="right"))
        window = local - (int(counts[row - 1]) if row else 0)
        start, size = map(int, self.dataset.indexes[file][row])
        if file not in self._maps:
            self._maps[file] = np.memmap(
                self.dataset.root / self.dataset.files[file]["token_ids"]["path"],
                mode="r",
                dtype="<u4",
            )
        positions = np.arange(window * self.parent_length, window * self.parent_length + length + 1)
        values = np.empty(length + 1, dtype=np.int64)
        raw = (positions > 0) & (positions <= size)
        values[raw] = self._maps[file][start + positions[raw] - 1]
        values[positions == 0] = self.dataset.bos
        values[positions == size + 1] = self.dataset.eos
        return {
            "input": torch.from_numpy(values[:-1].copy()),
            "positions": torch.arange(length),
        }, torch.from_numpy(values[1:].copy())

    def microbatches(self, anchors, length, dp_rank, dp_degree, local_batch_size):
        # Every global draw has exactly one owner, even when window IDs repeat.
        local = anchors[dp_rank::dp_degree]
        for start in range(0, len(local), local_batch_size):
            yield default_collate(
                [self.read(int(a), length) for a in local[start : start + local_batch_size]]
            )


class RandomWindowSampler:
    """Draw with replacement from all valid document/start pairs at each length.

    Each draw contains s inputs and their s next-token labels. Documents have
    virtual BOS and EOS tokens, as in the training reader. Different lengths
    and events use separate reproducible RNG streams. Training state is read-only.
    """

    identity = "uniform_valid_document_start_v1"

    def __init__(self, dataset: LongDocumentDataset, lengths: list[int], seed: int):
        if not isinstance(dataset, LongDocumentDataset):
            raise ValueError("X probes require the verified long-document token cache")
        if not lengths or any(length <= 0 for length in lengths) or seed < 0:
            raise ValueError("Random windows require positive lengths and a nonnegative seed")
        self.dataset, self.seed = dataset, seed
        self.cumulative, self.file_ends, self.pool_sizes = {}, {}, {}
        for length in lengths:
            # N raw tokens plus BOS/EOS give N + 2 - s valid starts for s labels.
            cumulative = [
                np.cumsum(np.maximum(index[:, 1].astype(np.int64) + 2 - length, 0))
                for index in dataset.indexes
            ]
            ends = np.cumsum([int(c[-1]) if len(c) else 0 for c in cumulative])
            pool_size = int(ends[-1]) if len(ends) else 0
            if not pool_size:
                raise ValueError(f"No complete document-local sequences of length {length}")
            self.cumulative[length], self.file_ends[length] = cumulative, ends
            self.pool_sizes[length] = pool_size
        self._maps = {}

    def sample_groups(self, completed_updates, num_groups, group_sequences, *, length):
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, completed_updates, length, 0x58535631])
        )
        return rng.integers(self.pool_sizes[length], size=(num_groups, group_sequences))

    def read(self, anchor, length):
        if length not in self.pool_sizes or not 0 <= anchor < self.pool_sizes[length]:
            raise ValueError("Invalid random window index or sequence length")
        ends = self.file_ends[length]
        file = int(np.searchsorted(ends, anchor, side="right"))
        local = int(anchor) - (int(ends[file - 1]) if file else 0)
        counts = self.cumulative[length][file]
        row = int(np.searchsorted(counts, local, side="right"))
        offset = local - (int(counts[row - 1]) if row else 0)
        start, size = map(int, self.dataset.indexes[file][row])
        if file not in self._maps:
            self._maps[file] = np.memmap(
                self.dataset.root / self.dataset.files[file]["token_ids"]["path"],
                mode="r",
                dtype="<u4",
            )
        positions = np.arange(offset, offset + length + 1)
        values = np.empty(length + 1, dtype=np.int64)
        raw = (positions > 0) & (positions <= size)
        values[raw] = self._maps[file][start + positions[raw] - 1]
        values[positions == 0] = self.dataset.bos
        values[positions == size + 1] = self.dataset.eos
        return {
            "input": torch.from_numpy(values[:-1].copy()),
            "positions": torch.arange(length),
        }, torch.from_numpy(values[1:].copy())

    def microbatches(self, anchors, length, dp_rank, dp_degree, local_batch_size):
        # Partition draw positions, so duplicate samples retain their multiplicity.
        local = anchors[dp_rank::dp_degree]
        for start in range(0, len(local), local_batch_size):
            yield default_collate(
                [self.read(int(a), length) for a in local[start : start + local_batch_size]]
            )
