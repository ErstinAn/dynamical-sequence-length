"""Pack tokens in one process. Use the current sequence length for each batch."""

from dataclasses import dataclass

import torch
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader, HuggingFaceTextDataset


class DynamicSeqDataset(HuggingFaceTextDataset):
    def __iter__(self):
        documents = self._get_data_iter()
        empty_epoch = self._sample_idx == 0 and not getattr(self, "_resumed", False)
        while True:
            # Use the restored buffers before the next document.
            # Read the current length before each sequence.
            # Remove its tokens from the buffers before yield.
            while len(self._inputs_buffer) >= self.seq_len:
                length = self.seq_len
                inputs = torch.tensor(self._inputs_buffer[:length], dtype=torch.long)
                labels = torch.tensor(self._labels_buffer[:length], dtype=torch.long)
                positions = torch.tensor(
                    self._normalize_positions(self._positions_buffer[:length]), dtype=torch.long
                )
                self._inputs_buffer = self._inputs_buffer[length:]
                self._labels_buffer = self._labels_buffer[length:]
                self._positions_buffer = self._positions_buffer[length:]
                yield {"input": inputs, "positions": positions}, labels
            try:
                sample = next(documents)
            except StopIteration:
                if not self.infinite:
                    return  # Keep unused tokens in the buffers. Do not add padding.
                if empty_epoch:
                    raise RuntimeError("Cannot loop an empty token dataset")
                self.reloop()
                documents = self._get_data_iter()
                empty_epoch = True
                continue
            tokens = self._tokenizer.encode(
                self._text_processor(sample), add_bos=True, add_eos=True
            )
            self._sample_idx += 1
            if len(tokens) < 2:
                continue
            empty_epoch = False
            self._inputs_buffer.extend(tokens[:-1])
            self._labels_buffer.extend(tokens[1:])
            self._positions_buffer.extend(range(len(tokens) - 1))

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._resumed = True


class DynamicSeqDataLoader(HuggingFaceTextDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTextDataLoader.Config):
        num_workers: int = 0
        prefetch_factor: int | None = None
        persistent_workers: bool = False

    def __init__(
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        tokenizer,
        seq_len,
        local_batch_size,
        snapshot_every_n_steps=1,
        **kwargs,
    ):
        if (
            config.num_workers != 0
            or config.persistent_workers
            or config.prefetch_factor is not None
        ):
            raise ValueError(
                "Dynamic seq_len requires num_workers=0, no prefetch or persistent workers"
            )
        self.max_seq_len = seq_len
        dataset = DynamicSeqDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        ParallelAwareDataloader.__init__(
            self,
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            num_workers=0,
            pin_memory=config.pin_memory,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )

    def set_seq_len(self, length: int):
        if not 0 < length <= self.max_seq_len:
            raise ValueError(f"seq_len {length} outside model capacity {self.max_seq_len}")
        self.dataset.seq_len = length
