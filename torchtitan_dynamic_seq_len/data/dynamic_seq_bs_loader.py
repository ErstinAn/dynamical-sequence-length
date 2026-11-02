"""Read variable batches from the existing sequence dataset in one process."""

from dataclasses import dataclass
from itertools import islice

from torch.utils.data import default_collate
from torchtitan.components.dataloader import ParallelAwareDataloader

from torchtitan_dynamic_seq_len.data.dynamic_seq_loader import (
    DynamicSeqDataLoader,
    DynamicSeqDataset,
)


class DynamicSeqBatchDataset(DynamicSeqDataset):
    microbatch_size: int

    def __iter__(self):
        sequences = super().__iter__()
        while True:
            size = self.microbatch_size
            samples = list(islice(sequences, size))
            if not samples:
                return
            yield default_collate(samples)
            if len(samples) < size:
                return


class DynamicSeqBSDataLoader(DynamicSeqDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(DynamicSeqDataLoader.Config):
        pass

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
                "Dynamic batches require num_workers=0 without prefetch or persistent workers"
            )
        self.max_seq_len = seq_len
        dataset = DynamicSeqBatchDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        dataset.microbatch_size = local_batch_size
        ParallelAwareDataloader.__init__(
            self,
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=None,
            num_workers=0,
            pin_memory=config.pin_memory,
            # An optimizer step can read a different number of batches each time.
            snapshot_every_n_steps=1,
        )

    def set_batch_shape(self, *, seq_len: int, microbatch_size: int):
        if not isinstance(microbatch_size, int) or microbatch_size < 1:
            raise ValueError("microbatch_size must be a positive integer")
        self.set_seq_len(seq_len)
        self.dataset.microbatch_size = microbatch_size
