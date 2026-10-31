"""This Trainer uses one sequence length for each optimizer step."""

import hashlib
import json
from dataclasses import asdict, dataclass, field

import torch
import torch.distributed as dist
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.loss import IGNORE_INDEX, ChunkedLossWrapper
from torchtitan.trainer import Trainer

from torchtitan_dynamic_seq_len.data.dynamic_seq_loader import DynamicSeqDataLoader
from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator
from torchtitan_dynamic_seq_len.dynamic_seq_metrics import DynamicSeqMetricsProcessor
from torchtitan_dynamic_seq_len.seq_len_scheduler import SeqLenSchedulerConfig
from torchtitan_dynamic_seq_len.training_config import ExperimentTrainerConfig


class DynamicSeqLenTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(ExperimentTrainerConfig):
        seq_len_scheduler: SeqLenSchedulerConfig = field(default_factory=SeqLenSchedulerConfig)

        def validate_dynamic(self):
            scheduler = self.seq_len_scheduler.build(self.training.steps)
            if not isinstance(self.dataloader, DynamicSeqDataLoader.Config):
                raise ValueError("Dynamic trainer requires DynamicSeqDataLoader")
            if (
                self.dataloader.num_workers != 0
                or self.dataloader.persistent_workers
                or self.dataloader.prefetch_factor is not None
            ):
                raise ValueError("Dynamic seq_len requires num_workers=0 without prefetch")
            p = self.parallelism
            if (
                any(
                    getattr(p, name) != 1
                    for name in (
                        "data_parallel_shard_degree",
                        "tensor_parallel_degree",
                        "pipeline_parallel_degree",
                        "context_parallel_degree",
                        "expert_parallel_degree",
                    )
                )
                or p.spmd_backend != "default"
            ):
                raise ValueError("Dynamic seq_len supports single GPU or replicated DP only")
            if not isinstance(self.metrics, DynamicSeqMetricsProcessor.Config):
                raise ValueError("Dynamic trainer requires DynamicSeqMetricsProcessor")
            if max(scheduler.lengths) > self.training.seq_len:
                raise ValueError("Scheduled length exceeds training.seq_len (model/RoPE capacity)")
            if self.validator.enable:
                if not isinstance(self.validator, FixedBudgetValidator.Config):
                    raise ValueError("Dynamic trainer requires FixedBudgetValidator")
                if not 0 < self.validator.eval_seq_len <= self.training.seq_len:
                    raise ValueError("Set a fixed validator.eval_seq_len within model capacity")
            if isinstance(self.loss, ChunkedLossWrapper.Config):
                lengths = list(scheduler.lengths)
                if self.validator.enable:
                    lengths.append(self.validator.eval_seq_len)
                if self.loss.num_chunks <= 0 or any(x % self.loss.num_chunks for x in lengths):
                    raise ValueError("Every train/eval length must be divisible by loss.num_chunks")
            return scheduler

    def __init__(self, config):
        self.seq_len_scheduler = config.validate_dynamic()
        self.n_valid_tokens_seen = 0  # global count, identical on every checkpoint rank
        self.current_seq_len = self.seq_len_scheduler.length_for_step(1)
        super().__init__(config)
        if self.num_pipeline_parallel_microbatches != 1:
            raise ValueError("Dynamic seq_len requires one microbatch per accumulation group")
        with torch.device("meta"):
            estimate_model = self.model_config.build()
        self.flops_by_length = {
            length: self.model_config.get_nparams_and_flops(estimate_model, length)[1]
            for length in self.seq_len_scheduler.lengths
        }

    def train_step(self, data_iterator):
        length = self.seq_len_scheduler.length_for_step(self.step)
        self.current_seq_len = length
        self.dataloader.set_seq_len(length)
        metrics = self.metrics_processor
        metrics.begin_step()
        previous_tokens, previous_valid = self.ntokens_seen, self.n_valid_tokens_seen
        try:
            # Collect all accumulation groups on the CPU before the parameter update.
            # This lets all ranks compare batch sizes before gradient communication.
            batches = self._stage_batches(data_iterator, length)
            tokens = sum(labels.numel() for _, labels in batches)
            self._count_valid_tokens = True
            metrics.record_step(
                length=length,
                tokens=tokens,
                flops_per_token=self.flops_by_length[length],
            )
            result = super().train_step(iter(batches))
            metrics.finish_step()
            return result
        except Exception:
            self.ntokens_seen, self.n_valid_tokens_seen = previous_tokens, previous_valid
            metrics.reset_window()
            raise

    def _stage_batches(self, data_iterator, length):
        batches = []
        status = 0
        try:
            batches = [next(data_iterator) for _ in range(self.gradient_accumulation_steps)]
        except (StopIteration, DataloaderExhaustedError):
            status = 1
        if not status:
            expected = (self.config.training.local_batch_size, length)
            if any(
                labels.ndim == 2 and 0 < labels.shape[0] < expected[0] and labels.shape[1] == length
                for _, labels in batches
            ):
                status = 1  # A finite dataset can give a partial final batch.
            elif any(tuple(labels.shape) != expected for _, labels in batches):
                status = 2
            elif sum(int((labels != IGNORE_INDEX).sum()) for _, labels in batches) == 0:
                status = 3
        if self.parallel_dims.dp_enabled:
            # All ranks must agree on cancellation before loss or gradient communication.
            # An exhausted dataset or invalid batch must stop all ranks.
            flag = torch.tensor(status, device=self.device, dtype=torch.int64)
            dist.all_reduce(
                flag, op=dist.ReduceOp.MAX, group=self.parallel_dims.get_mesh("loss").get_group()
            )
            status = int(flag)
        if status == 1:
            raise DataloaderExhaustedError()
        if status:
            raise ValueError("Every DP rank needs full microbatches with valid labels")
        return batches

    def forward_backward_step(self, *, input_dict, labels, global_valid_tokens):
        if self._count_valid_tokens:
            # Use the global token count from the TorchTitan Trainer.
            # DCP saves only one copy of the Trainer state that all ranks share.
            count = int(global_valid_tokens)
            self.n_valid_tokens_seen += count
            self.metrics_processor.global_valid_tokens = count
            self.metrics_processor.global_valid_tokens_seen = self.n_valid_tokens_seen
            self._count_valid_tokens = False
        return super().forward_backward_step(
            input_dict=input_dict, labels=labels, global_valid_tokens=global_valid_tokens
        )

    def _schedule_signature(self):
        protocol = dict(
            schedule=asdict(self.config.seq_len_scheduler),
            steps=self.config.training.steps,
            capacity=self.config.training.seq_len,
            local_batch=self.config.training.local_batch_size,
            accumulation=self.gradient_accumulation_steps,
            dp=self.parallel_dims.dp_replicate,
        )
        return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()

    def state_dict(self):
        return {
            **super().state_dict(),
            "n_valid_tokens_seen": self.n_valid_tokens_seen,
            "seq_len_signature": self._schedule_signature(),
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("seq_len_signature") != self._schedule_signature():
            raise ValueError("Checkpoint sequence schedule/batch/topology differs from this run")
        super().load_state_dict(state_dict)
        self.n_valid_tokens_seen = state_dict["n_valid_tokens_seen"]
        self.current_seq_len = self.seq_len_scheduler.length_for_step(max(1, self.step))
        self.metrics_processor.reset_window()
