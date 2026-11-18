"""Measure long loss, then update on the same samples' independently batched crops."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.distributed as dist
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.loss import IGNORE_INDEX, ChunkedLossWrapper
from torchtitan.trainer import Trainer

from torchtitan_dynamic_seq_len.data.dynamic_seq_bs_loader import DynamicSeqBSDataLoader
from torchtitan_dynamic_seq_len.data.seq_grad_windows import crop_batches
from torchtitan_dynamic_seq_len.dynamic_seq_bs.compile import compile_dynamic_blocks
from torchtitan_dynamic_seq_len.loss_grad_seq.batching import plan_batches, regroup_batches
from torchtitan_dynamic_seq_len.loss_grad_seq.metrics import LossGradMetricsProcessor
from torchtitan_dynamic_seq_len.loss_grad_seq.validation import LossLengthValidator
from torchtitan_dynamic_seq_len.schedulers.batch_size import BatchSizeSchedulerConfig
from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig
from torchtitan_dynamic_seq_len.training_config import ExperimentTrainerConfig


class LossGradSeqTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(ExperimentTrainerConfig):
        loss_seq_len_scheduler: SeqLenSchedulerConfig = field(
            default_factory=lambda: SeqLenSchedulerConfig(length=12288)
        )
        grad_seq_len_scheduler: SeqLenSchedulerConfig = field(default_factory=SeqLenSchedulerConfig)
        batch_size_scheduler: BatchSizeSchedulerConfig = field(
            default_factory=lambda: BatchSizeSchedulerConfig(batch_size=512)
        )
        loss_microbatch_scheduler: BatchSizeSchedulerConfig = field(
            default_factory=lambda: BatchSizeSchedulerConfig(batch_size=8)
        )
        grad_microbatch_scheduler: BatchSizeSchedulerConfig = field(
            default_factory=lambda: BatchSizeSchedulerConfig(batch_size=64)
        )
        loss_tokens_per_microbatch: int = 0
        grad_tokens_per_microbatch: int = 0
        crop_seed: int = 42
        document_masking: bool = True

        def schedules(self):
            return {
                name: getattr(self, name).build(self.training.steps)
                for name in (
                    "loss_seq_len_scheduler",
                    "grad_seq_len_scheduler",
                    "batch_size_scheduler",
                    "loss_microbatch_scheduler",
                    "grad_microbatch_scheduler",
                )
            }

        def plan(self, schedules, step, dp):
            return plan_batches(
                loss_length=schedules["loss_seq_len_scheduler"].length_for_step(step),
                grad_length=schedules["grad_seq_len_scheduler"].length_for_step(step),
                global_batch=schedules["batch_size_scheduler"].value_for_step(step),
                dp=dp,
                loss_limit=schedules["loss_microbatch_scheduler"].value_for_step(step),
                grad_limit=schedules["grad_microbatch_scheduler"].value_for_step(step),
                loss_token_budget=self.loss_tokens_per_microbatch,
                grad_token_budget=self.grad_tokens_per_microbatch,
            )

        def validate_loss_grad(self):
            if not isinstance(self.dataloader, DynamicSeqBSDataLoader.Config) or (
                self.dataloader.num_workers != 0
                or self.dataloader.persistent_workers
                or self.dataloader.prefetch_factor is not None
            ):
                raise ValueError(
                    "Loss/grad training requires DynamicSeqBSDataLoader without prefetch"
                )
            if not isinstance(self.metrics, LossGradMetricsProcessor.Config):
                raise ValueError("Loss/grad training requires LossGradMetricsProcessor")
            p = self.parallelism
            if p.spmd_backend != "default" or any(
                getattr(p, name) != 1
                for name in (
                    "data_parallel_shard_degree",
                    "tensor_parallel_degree",
                    "pipeline_parallel_degree",
                    "context_parallel_degree",
                    "expert_parallel_degree",
                )
            ):
                raise ValueError("Loss/grad training supports single GPU or replicated DP only")
            if isinstance(self.validator, LossLengthValidator.Config):
                self.validator.document_masking = self.document_masking
            packed = getattr(self.dataloader, "packing", "none") == "stream"
            if packed or not self.document_masking:
                from torchtitan.models.common.attention import FlexAttention
                if not isinstance(self.model_spec.model.first_attention.inner_attention, FlexAttention.Config):
                    raise ValueError("Packed loss/gradient training requires FlexAttention")
            if self.validator.enable:
                validation_packed = getattr(self.validator.dataloader, "packing", "none") == "stream"
                if packed != validation_packed and getattr(self.validator, "follow_loss_length", True):
                    raise ValueError("Train and validation must use the same packing mode")
            schedules = self.schedules()
            lengths = set()
            for step in range(1, self.training.steps + 1):
                plan = self.plan(schedules, step, p.data_parallel_replicate_degree)
                if plan.loss_length > self.training.seq_len:
                    raise ValueError(f"Loss sequence exceeds model capacity at step {step}")
                lengths.update((plan.loss_length, plan.grad_length))
            initial = self.plan(schedules, 1, p.data_parallel_replicate_degree)
            if (
                self.training.global_batch_size != initial.global_batch
                or self.training.local_batch_size < 1
                or initial.global_batch
                % (p.data_parallel_replicate_degree * self.training.local_batch_size)
            ):
                raise ValueError(
                    "Initial global batch must match; bootstrap local batch must divide global/DP"
                )
            if self.validator.enable:
                maximum_loss_length = max(schedules["loss_seq_len_scheduler"].lengths)
                if not isinstance(self.validator, LossLengthValidator.Config):
                    raise ValueError("Use LossLengthValidator for loss/gradient training")
                if self.validator.follow_loss_length and self.validator.eval_seq_len != maximum_loss_length:
                    raise ValueError(
                        "Use LossLengthValidator with eval_seq_len equal to maximum loss length"
                    )
                if not 0 < self.validator.eval_seq_len <= self.training.seq_len:
                    raise ValueError("Validation length must be positive and fit model capacity")
                lengths.add(self.validator.eval_seq_len)
            if isinstance(self.loss, ChunkedLossWrapper.Config) and (
                self.loss.num_chunks <= 0 or any(s % self.loss.num_chunks for s in lengths)
            ):
                raise ValueError(
                    "All loss/gradient/validation lengths must divide into loss chunks"
                )
            return schedules

    def __init__(self, config):
        self.schedules = config.validate_loss_grad()
        self.n_valid_tokens_seen = 0
        self.parent_tokens_seen = 0
        self.parent_valid_tokens_seen = 0
        super().__init__(config)
        if self.num_pipeline_parallel_microbatches != 1 or len(self.model_parts) != 1:
            raise ValueError("Loss/grad training requires one model and one batch per backward")
        compile_dynamic_blocks(self.model_parts, config.compile)
        with torch.device("meta"):
            estimate_model = self.model_config.build()
        lengths = set(self.schedules["loss_seq_len_scheduler"].lengths)
        lengths.update(self.schedules["grad_seq_len_scheduler"].lengths)
        self.flops_by_length = {
            s: self.model_config.get_nparams_and_flops(estimate_model, s)[1] for s in lengths
        }

    def post_dataloading_process(self, input_dict, labels):
        if self.config.document_masking:
            return super().post_dataloading_process(input_dict, labels)
        from torchtitan_dynamic_seq_len.loss_grad_seq.attention import select_attention
        # This trainer supports replicated DP only. Keep RoPE positions intact.
        positions = input_dict["positions"]
        return input_dict["input"], labels, {
            "positions": positions,
            "attention_masks": select_attention(self.model_parts[0], positions, False),
        }

    def _group(self):
        return (
            self.parallel_dims.get_mesh("loss").get_group()
            if self.parallel_dims.dp_enabled
            else None
        )

    def _prepare_step(self):
        plan = self.config.plan(self.schedules, self.step, self.parallel_dims.dp_replicate)
        self.current_plan = plan
        self.dataloader.set_batch_shape(
            seq_len=plan.loss_length, microbatch_size=plan.loss_microbatch
        )
        return plan

    def _crop(self, parents):
        return crop_batches(
            parents,
            self.current_plan.grad_length,
            seed=self.config.crop_seed,
            step=self.step,
            rank=dist.get_rank(self._group()) if self.parallel_dims.dp_enabled else 0,
            stream="loss-grad-train-v1",
        )

    def _stage_parents(self, data_iterator):
        """Read exactly the local sample budget, including an intentional smaller tail."""
        plan = self.current_plan
        remaining = plan.global_batch // self.parallel_dims.dp_replicate
        batches, status = [], 0
        try:
            while remaining:
                size = min(remaining, plan.loss_microbatch)
                self.dataloader.set_batch_shape(seq_len=plan.loss_length, microbatch_size=size)
                inputs, labels = next(data_iterator)
                if not isinstance(labels, torch.Tensor) or labels.device.type != "cpu":
                    status = 2
                    break
                if tuple(labels.shape) != (size, plan.loss_length):
                    partial = (
                        labels.ndim == 2
                        and 0 < labels.shape[0] < size
                        and labels.shape[1] == plan.loss_length
                    )
                    status = 1 if partial else 2
                    break
                batches.append((inputs, labels))
                remaining -= size
        except (StopIteration, DataloaderExhaustedError):
            status = 1
        flag = torch.tensor(status, device=self.device, dtype=torch.int64)
        if self.parallel_dims.dp_enabled:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self._group())
        status = int(flag)
        if status == 1:
            raise DataloaderExhaustedError()
        if status:
            raise ValueError("Parent labels must be CPU tensors matching the scheduled shape")
        return batches

    def _check_data(self, batches):
        bad = any(
            "input" not in inputs
            or any(
                key not in ("input", "positions")
                or not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.shape != labels.shape
                for key, value in inputs.items()
            )
            for inputs, labels in batches
        )
        count = sum(int((labels != IGNORE_INDEX).sum()) for _, labels in batches)
        totals = torch.tensor(
            [count, int(bad or count == 0)], device=self.device, dtype=torch.int64
        )
        if self.parallel_dims.dp_enabled:
            dist.all_reduce(totals, group=self._group())
        if totals[1].item():
            raise ValueError("Every rank needs aligned CPU inputs/positions and valid labels")
        return totals[0]

    @torch.no_grad()
    def _long_forward(self, parents, valid_tokens):
        loss = torch.zeros((), device=self.device, dtype=torch.float64)
        local_count = sum(int((labels != IGNORE_INDEX).sum()) for _, labels in parents)
        model = self.model_parts[0]
        buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
        previous_tokens = self.ntokens_seen
        devices = [self.device.index] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                for inputs, labels in parents:
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    labels = labels.to(self.device)
                    x, labels, kwargs = self.post_dataloading_process(inputs, labels)
                    with self.train_context():
                        pred = model(x, **kwargs)
                        loss_kwargs = (
                            {"positions": kwargs["positions"]} if "positions" in kwargs else {}
                        )
                        value, _ = self.loss_fn(pred, labels, valid_tokens, **loss_kwargs)
                    loss += value.detach().double()
                    del pred, value, x, labels, inputs, kwargs
        finally:
            self.ntokens_seen = previous_tokens
            for buffer, saved in buffers:
                buffer.copy_(saved)
        max_loss = loss * valid_tokens / local_count
        if self.parallel_dims.dp_enabled:
            dist.all_reduce(loss, group=self._group())
            dist.all_reduce(max_loss, op=dist.ReduceOp.MAX, group=self._group())
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite long-sequence loss")
        return loss.item(), max_loss.item()

    def train_step(self, data_iterator):
        plan = self._prepare_step()
        metrics = self.metrics_processor
        metrics.begin_step()
        saved = (
            self.ntokens_seen,
            self.n_valid_tokens_seen,
            self.parent_tokens_seen,
            self.parent_valid_tokens_seen,
        )
        try:
            parents = self._stage_parents(data_iterator)
            parent_valid = self._check_data(parents)
            # Crop on CPU before the first forward to reject invalid crops collectively.
            crops = regroup_batches(self._crop(parents), plan.grad_microbatch)
            self._check_data(crops)
            self.optimizers.zero_grad()
            started = time.perf_counter()
            metrics.long_loss, metrics.long_loss_max = self._long_forward(parents, parent_valid)
            long_seconds = time.perf_counter() - started
            fragment_count = sum(int((inputs.get("positions", torch.ones_like(labels)) == 0).sum())
                                 for inputs, labels in parents)
            fragments = torch.tensor(fragment_count, device=self.device, dtype=torch.int64)
            if self.parallel_dims.dp_enabled:
                dist.all_reduce(fragments, group=self._group())
            del parents
            self.parent_tokens_seen += plan.global_batch * plan.loss_length
            self.parent_valid_tokens_seen += int(parent_valid)
            self.gradient_accumulation_steps = plan.grad_batches
            self._count_valid_tokens = True
            metrics.ntokens_since_last_log = 0
            # Upstream model FLOPs describe F+B. Forward-only FLOPs use the conventional
            # one-third estimate, NOT a memory-capacity ratio (nor measured work under AC).
            effective_flops = self.flops_by_length[plan.grad_length] + (
                self.flops_by_length[plan.loss_length] * plan.loss_length / (3 * plan.grad_length)
            )
            metrics.record_step(
                length=plan.grad_length,
                tokens=sum(y.numel() for _, y in crops),
                flops_per_token=effective_flops,
            )
            metrics.phase_metrics = {
                "packing/document_masking": int(self.config.document_masking),
                "packing/parent_fragments": int(fragments),
                "packing/mean_fragment_positions": plan.global_batch * plan.loss_length / max(1, int(fragments)),
                "packing/ignored_parent_labels": plan.global_batch * plan.loss_length - int(parent_valid),
                "loss_grad/loss_seq_length": plan.loss_length,
                "loss_grad/grad_seq_length": plan.grad_length,
                "loss_grad/global_batch_size": plan.global_batch,
                "loss_grad/loss_microbatch_size": plan.loss_microbatch,
                "loss_grad/grad_microbatch_size": plan.grad_microbatch,
                "loss_grad/loss_forward_batches": plan.loss_batches,
                "loss_grad/gradient_accumulation_steps": plan.grad_batches,
                "loss_grad/parent_tokens": plan.global_batch * plan.loss_length,
                "loss_grad/grad_tokens": plan.global_batch * plan.grad_length,
                "loss_grad/parent_valid_tokens": int(parent_valid),
                "loss_grad/parent_tokens_seen": self.parent_tokens_seen,
                "loss_grad/parent_valid_tokens_seen": self.parent_valid_tokens_seen,
                "loss_grad/long_forward_seconds": long_seconds,
            }
            metrics.grad_phase_started = time.perf_counter()
            result = super().train_step(iter(crops))
            metrics.finish_step()
            return result
        except Exception:
            (
                self.ntokens_seen,
                self.n_valid_tokens_seen,
                self.parent_tokens_seen,
                self.parent_valid_tokens_seen,
            ) = saved
            metrics.reset_window()
            raise

    def forward_backward_step(self, *, input_dict, labels, global_valid_tokens):
        if self._count_valid_tokens:
            self.n_valid_tokens_seen += int(global_valid_tokens)
            self.metrics_processor.global_valid_tokens = int(global_valid_tokens)
            self.metrics_processor.global_valid_tokens_seen = self.n_valid_tokens_seen
            self._count_valid_tokens = False
        try:
            return super().forward_backward_step(
                input_dict=input_dict, labels=labels, global_valid_tokens=global_valid_tokens
            )
        finally:
            # Native train_step moves input dicts in-place and retains all groups.
            # Drop processed GPU input references instead of retaining the whole update.
            input_dict.clear()

    def _signature(self):
        protocol = {
            "version": 2,
            "document_masking": self.config.document_masking,
            "packing": getattr(self.config.dataloader, "packing", "none"),
            "packing_protocol": "stream-v1-bos-eos-ignore-boundary-rebase-fragment",
            "dataset_manifest": getattr(self.config.dataloader, "expected_manifest_sha256", ""),
            "validation_manifest": getattr(getattr(getattr(self, "validator", None), "parent_artifact", None), "manifest_sha256", None),
            **{name: asdict(getattr(self.config, name)) for name in self.schedules},
            "steps": self.config.training.steps,
            "capacity": self.config.training.seq_len,
            "dp": self.parallel_dims.dp_replicate,
            "crop_seed": self.config.crop_seed,
            "loss_tokens_per_microbatch": self.config.loss_tokens_per_microbatch,
            "grad_tokens_per_microbatch": self.config.grad_tokens_per_microbatch,
        }
        return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()

    def state_dict(self):
        return {
            **super().state_dict(),
            "loss_grad_signature": self._signature(),
            "n_valid_tokens_seen": self.n_valid_tokens_seen,
            "parent_tokens_seen": self.parent_tokens_seen,
            "parent_valid_tokens_seen": self.parent_valid_tokens_seen,
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("loss_grad_signature") != self._signature():
            raise ValueError("Checkpoint loss/gradient schedules, batches or topology differ")
        super().load_state_dict(state_dict)
        for name in ("n_valid_tokens_seen", "parent_tokens_seen", "parent_valid_tokens_seen"):
            setattr(self, name, state_dict[name])
        self.metrics_processor.reset_window()
