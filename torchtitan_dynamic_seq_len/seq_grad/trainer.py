"""Probe sequence lengths at fixed parameters, then update with one training crop."""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.distributed as dist
from torchtitan.components.loss import IGNORE_INDEX, ChunkedLossWrapper
from torchtitan.trainer import Trainer

from torchtitan_dynamic_seq_len.data.dynamic_seq_loader import DynamicSeqDataLoader
from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator
from torchtitan_dynamic_seq_len.data.seq_grad_windows import crop_batches
from torchtitan_dynamic_seq_len.dynamic_seq.trainer import DynamicSeqLenTrainer
from torchtitan_dynamic_seq_len.grad_var.trainer import GradVarTrainer
from torchtitan_dynamic_seq_len.seq_grad.metrics import SeqGradMetricsProcessor
from torchtitan_dynamic_seq_len.seq_grad.monitor import SeqGradConfig, SeqGradMonitor
from torchtitan_dynamic_seq_len.training_config import ExperimentTrainerConfig


class SeqGradTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(ExperimentTrainerConfig):
        seq_grad: SeqGradConfig = field(default_factory=SeqGradConfig)

        def validate_seq_grad(self):
            cfg, capacity = self.seq_grad, self.training.seq_len
            if cfg.interval < 1:
                raise ValueError("Sequence monitoring interval must be positive")
            if self.training.global_batch_size <= 0 or self.training.local_batch_size <= 0:
                raise ValueError(
                    "Sequence probes require explicit positive global and local batches"
                )
            if not cfg.lengths or len(set(cfg.lengths)) != len(cfg.lengths):
                raise ValueError("Provide unique candidate sequence lengths")
            if cfg.train_length not in cfg.lengths:
                raise ValueError("Candidate lengths must include train_length")
            if any(length <= 0 or length > capacity for length in cfg.lengths):
                raise ValueError(
                    "Candidate lengths must be within training.seq_len (parent length)"
                )
            if not math.isfinite(cfg.relative_epsilon) or cfg.relative_epsilon <= 0:
                raise ValueError("relative_epsilon must be finite and positive")
            if not isinstance(self.dataloader, DynamicSeqDataLoader.Config) or (
                self.dataloader.num_workers != 0
                or self.dataloader.persistent_workers
                or self.dataloader.prefetch_factor is not None
            ):
                raise ValueError("Sequence probes require DynamicSeqDataLoader without prefetch")
            if not isinstance(self.metrics, SeqGradMetricsProcessor.Config):
                raise ValueError("Sequence probes require SeqGradMetricsProcessor")
            if self.metrics.log_freq != 1 or self.metrics.profile_freq < 0:
                raise ValueError("Sequence probes require log_freq=1 and profile_freq >= 0")
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
                raise ValueError("Sequence probes support single GPU or replicated DP only")
            lengths = [capacity, *cfg.lengths]
            if self.validator.enable:
                if not isinstance(self.validator, FixedBudgetValidator.Config):
                    raise ValueError("Sequence probes require FixedBudgetValidator")
                if not 0 < self.validator.eval_seq_len <= capacity:
                    raise ValueError("Set a fixed validation length within model capacity")
                lengths.append(self.validator.eval_seq_len)
            if isinstance(self.loss, ChunkedLossWrapper.Config) and (
                self.loss.num_chunks <= 0 or any(s % self.loss.num_chunks for s in lengths)
            ):
                raise ValueError(
                    "All parent, candidate and validation lengths must divide into loss chunks"
                )

    def __init__(self, config):
        config.validate_seq_grad()
        self.n_valid_tokens_seen = 0
        self._probe_active = False
        self._training_calls = 0
        super().__init__(config)
        if self.num_pipeline_parallel_microbatches != 1 or len(self.model_parts) != 1:
            raise ValueError("Sequence probes require one local batch per backward and one model")
        self.seq_grad_monitor = SeqGradMonitor(
            self.model_parts[0].named_parameters(), config.seq_grad.relative_epsilon
        )
        with torch.device("meta"):
            estimate_model = self.model_config.build()
        self.flops_by_length = {
            s: self.model_config.get_nparams_and_flops(estimate_model, s)[1]
            for s in {config.training.seq_len, *config.seq_grad.lengths}
        }

    def _group(self):
        return (
            self.parallel_dims.get_mesh("batch").get_group()
            if self.parallel_dims.dp_enabled
            else None
        )

    def _crop(self, batches, length, stream):
        rank = dist.get_rank(self._group()) if self.parallel_dims.dp_enabled else 0
        return crop_batches(
            batches,
            length,
            seed=self.config.seq_grad.seed,
            step=self.step,
            rank=rank,
            stream=stream,
        )

    def _valid_tokens(self, batches, *, require_local=False):
        count = sum(int((labels != IGNORE_INDEX).sum()) for _, labels in batches)
        counts = torch.tensor([count, int(count == 0)], dtype=torch.int64, device=self.device)
        if self.parallel_dims.dp_enabled:
            dist.all_reduce(counts, group=self._group())
        if require_local and counts[1].item():
            raise ValueError("Every DP rank needs valid labels in its training crops")
        if counts[0].item() <= 0:
            raise ValueError("A sequence gradient needs positive global valid-token count")
        return counts[0]

    def _statistics(self, capture=False):
        error, result = None, None
        try:
            GradVarTrainer._check_gradient_sync(self)
            result = self.seq_grad_monitor.capture() if capture else self.seq_grad_monitor.compare()
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        failed = torch.tensor(int(error is not None), device=self.device)
        if self.parallel_dims.dp_enabled:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self._group())
        if failed.item():
            raise RuntimeError(error or "A DP rank rejected sequence gradient statistics")
        return result

    def _probe(self, batches):
        self.optimizers.zero_grad()
        valid_tokens = self._valid_tokens(batches)
        loss = torch.zeros((), device=self.device)
        for inputs, labels in batches:
            loss = (
                loss
                + self.forward_backward_step(
                    input_dict={key: value.to(self.device) for key, value in inputs.items()},
                    labels=labels.to(self.device),
                    global_valid_tokens=valid_tokens,
                ).detach()
            )
        if self.parallel_dims.dp_enabled:
            dist.all_reduce(loss, group=self._group())
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite sequence probe loss")
        return {"loss": loss.item(), "valid_tokens": int(valid_tokens)}

    def _record(self, length, values):
        self._step_metrics.update(
            {f"seq_grad_{name}/s{length:05d}": value for name, value in values.items()}
        )

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def train_step(self, data_iterator):
        cfg, metrics = self.config.seq_grad, self.metrics_processor
        metrics.begin_step()
        previous_tokens, previous_valid = self.ntokens_seen, self.n_valid_tokens_seen
        self._training_calls = 0
        self._monitor_this_step = cfg.should_monitor(self.step, self.config.training.steps)
        self._step_metrics = {"seq_grad/model_step": self.step - 1}
        capacity = self.config.training.seq_len
        try:
            parents = DynamicSeqLenTrainer._stage_batches(self, data_iterator, capacity)
            # Reject unsupported tensors collectively before any forward/backward.
            malformed = any(
                "input" not in inputs
                or any(
                    key not in ("input", "positions")
                    or not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.shape != labels.shape
                    for key, value in inputs.items()
                )
                for inputs, labels in parents
            )
            flag = torch.tensor(int(malformed), device=self.device)
            if self.parallel_dims.dp_enabled:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self._group())
            if flag.item():
                raise ValueError("Parent inputs and positions must be CPU tensors matching labels")
            training_batches = self._crop(parents, cfg.train_length, "train")
            # Fail before update, even when monitoring is disabled.
            self._valid_tokens(training_batches, require_local=True)
            metrics.ntokens_since_last_log = 0  # Parent reads are not training tokens.
            tokens = sum(labels.numel() for _, labels in training_batches)
            evaluated = [cfg.train_length]
            probe_seconds = 0.0
            if self._monitor_this_step:
                self._sync_device()
                started = time.perf_counter()
                self._probe_active = True
                devices = [self.device.index] if self.device.type == "cuda" else []
                buffers = [
                    (buffer, buffer.detach().clone()) for buffer in self.model_parts[0].buffers()
                ]
                try:
                    # Probes cannot consume the RNG used for the training forward.
                    with torch.random.fork_rng(devices=devices):
                        reference = self._probe(parents)
                        reference["grad_norm"] = self._statistics(capture=True)
                        self._step_metrics.update(
                            {
                                f"seq_grad_reference/{name}": value
                                for name, value in reference.items()
                            }
                        )
                        for length in cfg.lengths:
                            if length == cfg.train_length:
                                continue
                            # Restore mutable model buffers before each independent probe.
                            with torch.no_grad():
                                for buffer, saved in buffers:
                                    buffer.copy_(saved)
                            values = self._probe(self._crop(parents, length, f"probe:{length}"))
                            values.update(self._statistics())
                            self._record(length, values)
                finally:
                    with torch.no_grad():
                        for buffer, saved in buffers:
                            buffer.copy_(saved)
                    self._probe_active = False
                    self.optimizers.zero_grad()
                evaluated = [capacity, *cfg.lengths]
                self._sync_device()
                probe_seconds = time.perf_counter() - started
            del parents
            global_batch = self.config.training.global_batch_size
            self._step_metrics.update(
                {
                    "seq_grad/reference_length": capacity,
                    "seq_grad/train_length": cfg.train_length,
                    "seq_grad/global_batch_size": global_batch,
                    "seq_grad/samples_per_length": int(self._monitor_this_step),
                    "seq_grad_cost/probe_seconds": probe_seconds,
                    "seq_grad_cost/probe_tokens": global_batch
                    * (sum(evaluated) - cfg.train_length),
                    "seq_grad_cost/forward_backward_tokens": global_batch * sum(evaluated),
                    "seq_grad_cost/parent_tokens": global_batch * capacity,
                }
            )
            # Native TPS counts training tokens. FLOPs include every evaluated length.
            flops_per_training_token = (
                sum(s * self.flops_by_length[s] for s in evaluated) / cfg.train_length
            )
            metrics.record_step(
                length=cfg.train_length, tokens=tokens, flops_per_token=flops_per_training_token
            )
            result = super().train_step(iter(training_batches))
            metrics.finish_step()
            return result
        except Exception:
            self.ntokens_seen, self.n_valid_tokens_seen = previous_tokens, previous_valid
            metrics.reset_window()
            raise
        finally:
            self._probe_active = False
            self.seq_grad_monitor.reset()

    def forward_backward_step(self, *, input_dict, labels, global_valid_tokens):
        previous_tokens = self.ntokens_seen
        try:
            loss = super().forward_backward_step(
                input_dict=input_dict, labels=labels, global_valid_tokens=global_valid_tokens
            )
        finally:
            if self._probe_active:
                self.ntokens_seen = previous_tokens
        if not self._probe_active:
            self._training_calls += 1
            if self._training_calls == 1:
                self.n_valid_tokens_seen += int(global_valid_tokens)
                self.metrics_processor.global_valid_tokens = int(global_valid_tokens)
                self.metrics_processor.global_valid_tokens_seen = self.n_valid_tokens_seen
            if self._training_calls == self.gradient_accumulation_steps:
                if self._monitor_this_step:
                    values = self._statistics()
                    values["valid_tokens"] = int(global_valid_tokens)
                    self._record(self.config.seq_grad.train_length, values)
                self.metrics_processor.queue(self.step, self._step_metrics)
        return loss

    def _signature(self):
        protocol = dict(
            version=1,
            sampling=asdict(self.config.seq_grad),
            capacity=self.config.training.seq_len,
            local_batch=self.config.training.local_batch_size,
            global_batch=self.config.training.global_batch_size,
            accumulation=self.gradient_accumulation_steps,
            dp=self.parallel_dims.dp_replicate,
        )
        return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()

    def state_dict(self):
        return {
            **super().state_dict(),
            "n_valid_tokens_seen": self.n_valid_tokens_seen,
            "seq_grad_signature": self._signature(),
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("seq_grad_signature") != self._signature():
            raise ValueError("Checkpoint sequence probe protocol or batch topology differs")
        super().load_state_dict(state_dict)
        self.n_valid_tokens_seen = state_dict["n_valid_tokens_seen"]
        self.metrics_processor.reset_window()
