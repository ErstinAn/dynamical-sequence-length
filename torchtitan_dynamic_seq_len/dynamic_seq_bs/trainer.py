"""Apply independent schedules for length, global batch, and microbatch."""

import hashlib
import json
from dataclasses import asdict, dataclass, field

from torchtitan_dynamic_seq_len.data.dynamic_seq_bs_loader import DynamicSeqBSDataLoader
from torchtitan_dynamic_seq_len.dynamic_seq.trainer import DynamicSeqLenTrainer
from torchtitan_dynamic_seq_len.dynamic_seq_bs.compile import compile_dynamic_blocks
from torchtitan_dynamic_seq_len.dynamic_seq_bs.metrics import DynamicSeqBSMetricsProcessor
from torchtitan_dynamic_seq_len.schedulers.batch_size import BatchSizeSchedulerConfig


def accumulation_for_step(step, length, global_batch, microbatch, dp):
    if any(
        type(value) is not int or value < 1 for value in (length, global_batch, microbatch, dp)
    ) or global_batch % (dp * microbatch):
        raise ValueError(
            f"Invalid schedules at step {step}: seq_len={length}, global_batch={global_batch}, "
            f"microbatch={microbatch}, dp={dp}. Use positive integers. "
            "global_batch must be divisible by dp * microbatch."
        )
    return global_batch // (dp * microbatch)


class DynamicSeqBSTrainer(DynamicSeqLenTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(DynamicSeqLenTrainer.Config):
        batch_size_scheduler: BatchSizeSchedulerConfig = field(
            default_factory=lambda: BatchSizeSchedulerConfig(batch_size=512)
        )
        microbatch_scheduler: BatchSizeSchedulerConfig = field(
            default_factory=BatchSizeSchedulerConfig
        )

        def validate_dynamic(self):
            lengths = DynamicSeqLenTrainer.Config.validate_dynamic(self)
            if not isinstance(self.dataloader, DynamicSeqBSDataLoader.Config):
                raise ValueError("Dynamic batch training requires DynamicSeqBSDataLoader")
            if not isinstance(self.metrics, DynamicSeqBSMetricsProcessor.Config):
                raise ValueError("Dynamic batch training requires DynamicSeqBSMetricsProcessor")
            batches = self.batch_size_scheduler.build(self.training.steps)
            microbatches = self.microbatch_scheduler.build(self.training.steps)
            for step in range(1, self.training.steps + 1):
                accumulation_for_step(
                    step,
                    lengths.length_for_step(step),
                    batches.value_for_step(step),
                    microbatches.value_for_step(step),
                    self.parallelism.data_parallel_replicate_degree,
                )
            if self.training.global_batch_size != batches.value_for_step(1):
                raise ValueError("training.global_batch_size must match the schedule at step 1")
            if self.training.local_batch_size != microbatches.value_for_step(1):
                raise ValueError(
                    "training.local_batch_size must match the microbatch schedule at step 1"
                )
            return lengths

    def __init__(self, config):
        # Checkpoint loading occurs inside the parent constructor.
        self.batch_size_scheduler = config.batch_size_scheduler.build(config.training.steps)
        self.microbatch_scheduler = config.microbatch_scheduler.build(config.training.steps)
        super().__init__(config)
        compile_dynamic_blocks(self.model_parts, config.compile)

    def _set_batch_step(self, step):
        length = self.seq_len_scheduler.length_for_step(step)
        batch = self.batch_size_scheduler.value_for_step(step)
        microbatch = self.microbatch_scheduler.value_for_step(step)
        self.gradient_accumulation_steps = accumulation_for_step(
            step, length, batch, microbatch, self.parallel_dims.dp_replicate
        )
        self.current_seq_len = length
        self.current_global_batch_size = batch
        self.current_microbatch_size = microbatch
        self.dataloader.set_batch_shape(seq_len=length, microbatch_size=microbatch)
        metrics = self.metrics_processor
        metrics.current_global_batch_size = batch
        metrics.current_microbatch_size = microbatch
        metrics.current_accumulation_steps = self.gradient_accumulation_steps
        return length

    def _prepare_step(self):
        return self._set_batch_step(self.step)

    def _stage_batches(self, data_iterator, length):
        return super()._stage_batches(
            data_iterator, length, microbatch_size=self.current_microbatch_size
        )

    def _schedule_signature(self):
        protocol = dict(
            mode="dynamic_seq_bs",
            seq_len_scheduler=asdict(self.config.seq_len_scheduler),
            batch_size_scheduler=asdict(self.config.batch_size_scheduler),
            microbatch_scheduler=asdict(self.config.microbatch_scheduler),
            steps=self.config.training.steps,
            capacity=self.config.training.seq_len,
            dp=self.parallel_dims.dp_replicate,
        )
        for name in ("seq_len_scheduler", "batch_size_scheduler", "microbatch_scheduler"):
            if protocol[name].get("phase_offset_steps") == 0:
                protocol[name].pop("phase_offset_steps", None)
            if protocol[name].get("anchor_steps") == []:
                protocol[name].pop("anchor_steps", None)
            if protocol[name].get("repeat") is False:
                protocol[name].pop("repeat", None)
        return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._set_batch_step(max(1, self.step))
