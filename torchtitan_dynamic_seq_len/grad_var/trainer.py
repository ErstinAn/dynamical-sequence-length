"""TorchTitan Trainer with pre-clipping, fixed-batch gradient variance monitoring."""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.trainer import Trainer

from torchtitan_dynamic_seq_len.grad_var.metrics import GradVarMetricsProcessor
from torchtitan_dynamic_seq_len.grad_var.monitor import GradVarMonitor, GradVarMonitorConfig
from torchtitan_dynamic_seq_len.training_config import ExperimentTrainerConfig


class GradVarTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(ExperimentTrainerConfig):
        grad_var_monitor: GradVarMonitorConfig = field(default_factory=GradVarMonitorConfig)

    def __init__(self, config: Config):
        self._x_last_event = -1
        if config.grad_var_monitor.enable and config.grad_var_monitor.mode == "x":
            from torchtitan_dynamic_seq_len.grad_var.probes import validate_config

            validate_config(config)
        super().__init__(config)
        self._init_grad_var_monitor()

    def _init_grad_var_monitor(self):
        self.grad_var_monitor = None
        self._grad_var_active = False
        self._grad_var_group = None
        if not self.config.grad_var_monitor.enable:
            return
        dims = self.parallel_dims
        if (
            any(getattr(dims, axis) != 1 for axis in ("dp_shard", "tp", "pp", "cp", "ep"))
            or dims.spmd_backend != "default"
            or len(self.model_parts) != 1
        ):
            raise ValueError("gradient variance supports a single model with replicated DP only")
        if not isinstance(self.metrics_processor, GradVarMetricsProcessor):
            raise ValueError("gradient variance requires GradVarMetricsProcessor")
        if self.num_pipeline_parallel_microbatches != 1:
            raise ValueError("gradient variance requires one local batch per backward")
        if self.config.grad_var_monitor.mode == "x":
            from torchtitan_dynamic_seq_len.grad_var.probes import initialize

            initialize(self)
            return
        self.grad_var_monitor = GradVarMonitor(
            self.model_parts[0].named_parameters(), self.gradient_accumulation_steps
        )
        if dims.dp_enabled:
            self._grad_var_group = dims.get_mesh("batch").get_group()
        training = self.config.training
        self._grad_var_shape = (training.local_batch_size, training.seq_len)
        self._grad_var_fixed_batch = training.global_batch_size

    def _check_gradient_sync(self):
        # Pinned TorchTitan uses FSDP2, with SUM reduction and synchronous backward
        # on every accumulation group. Reject no_sync if introduced by a caller.
        for module in self.model_parts[0].modules():
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if not state._state_ctx.is_last_backward:
                    raise ValueError("gradient variance requires backward communication completion")
                for group in state._fsdp_param_groups:
                    if not group.reduce_grads or not group.all_reduce_grads:
                        raise ValueError(
                            "gradient variance does not support unsynchronized accumulation"
                        )
                    if group.gradient_divide_factor != 1.0:
                        raise ValueError(
                            "gradient variance requires TorchTitan's SUM gradient reduction"
                        )

    def train_step(self, data_iterator):
        if self.config.grad_var_monitor.enable and self.config.grad_var_monitor.mode == "x":
            from torchtitan_dynamic_seq_len.grad_var.probes import maybe_monitor

            maybe_monitor(self, self.step - 1)
            result = super().train_step(data_iterator)
            maybe_monitor(self, self.step)
            return result
        monitor = self.grad_var_monitor
        self._grad_var_active = monitor is not None and self.metrics_processor.should_log(self.step)
        if self._grad_var_active:
            self.metrics_processor.clear_grad_var()
            monitor.begin()
            self._grad_var_counts = []
            self._grad_var_error = None
            self._grad_var_calls = 0
        try:
            return super().train_step(data_iterator)
        finally:
            if self._grad_var_active:
                monitor.reset()
                self.metrics_processor.clear_grad_var()
                self._grad_var_counts = []
                self._grad_var_error = None
            self._grad_var_active = False

    def forward_backward_step(self, *, input_dict, labels, global_valid_tokens):
        loss = super().forward_backward_step(
            input_dict=input_dict, labels=labels, global_valid_tokens=global_valid_tokens
        )
        if self._grad_var_active:
            self._grad_var_calls += 1
            self._grad_var_counts.append((labels != IGNORE_INDEX).sum())
            try:
                training = self.config.training
                if (
                    tuple(labels.shape) != self._grad_var_shape
                    or (training.local_batch_size, training.seq_len) != self._grad_var_shape
                    or training.global_batch_size != self._grad_var_fixed_batch
                    or self.gradient_accumulation_steps != self.grad_var_monitor.k
                ):
                    raise ValueError(
                        "gradient variance requires fixed batch size and sequence length"
                    )
                self._check_gradient_sync()
                if self._grad_var_error is None:
                    self.grad_var_monitor.observe()
            except (ValueError, RuntimeError) as error:
                # Continue existing backwards on every rank before collectively failing.
                self._grad_var_error = str(error)
            if self._grad_var_calls == self.gradient_accumulation_steps:
                self._finish_grad_var(global_valid_tokens)
        return loss

    def _finish_grad_var(self, global_valid_tokens):
        metrics = None
        if self._grad_var_error is None:
            try:
                metrics = self.grad_var_monitor.finish()
            except (ValueError, RuntimeError) as error:
                self._grad_var_error = str(error)
        # Only counts/error flags are reduced. Gradients are already synchronized;
        # reducing variance scalars again would incorrectly multiply them by DP size.
        status = torch.stack(
            self._grad_var_counts
            + [
                torch.tensor(
                    int(self._grad_var_error is not None), dtype=torch.int64, device=self.device
                )
            ]
        )
        if self._grad_var_group is not None:
            dist.all_reduce(status, op=dist.ReduceOp.SUM, group=self._grad_var_group)
        counts = status[:-1]
        if status[-1].item():
            detail = self._grad_var_error or "a different DP rank rejected gradient statistics"
            raise RuntimeError(f"gradient variance failed at step {self.step}: {detail}")
        if (
            not bool((counts > 0).all())
            or not bool((counts == counts[0]).all())
            or not bool(counts.sum() == global_valid_tokens)
        ):
            raise RuntimeError(
                f"gradient variance requires equal positive global valid-token counts per group "
                f"at step {self.step}; got {counts.tolist()}"
            )
        self.metrics_processor.queue_grad_var(self.step, metrics)

    def post_dataloading_process(self, input_dict, labels):
        before = self.ntokens_seen
        try:
            return super().post_dataloading_process(input_dict, labels)
        finally:
            if getattr(self, "_x_probe_active", False):
                self.ntokens_seen = before

    def state_dict(self):
        state = super().state_dict()
        if self.config.grad_var_monitor.enable and self.config.grad_var_monitor.mode == "x":
            state["x_last_event"] = self._x_last_event
            state["x_signature"] = self._x_signature
        return state

    def load_state_dict(self, state_dict):
        if self.config.grad_var_monitor.enable and self.config.grad_var_monitor.mode == "x":
            if state_dict.get("x_signature") != self._x_signature:
                raise ValueError("Checkpoint X protocol or dataset differs")
            self._x_last_event = state_dict["x_last_event"]
        super().load_state_dict(state_dict)
