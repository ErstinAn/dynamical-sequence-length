"""Merge variance scalars into TorchTitan's existing training log payload."""

from dataclasses import dataclass

from torchtitan.components.metrics import BaseLogger, LoggerContainer, MetricsProcessor, WandBLogger


class XWandBLogger(BaseLogger):
    """Leave a model step open for training, post-update probes and validation."""

    def __init__(self, wrapped, metadata):
        self.wrapped = wrapped
        self.last_step = -1
        run = wrapped.wandb.run
        run.define_metric("grad_var/step")
        run.define_metric("grad_var/x_s*", step_metric="grad_var/step")
        run.config.update({"x_probe": metadata})

    def log(self, metrics, step):
        if step < self.last_step:
            raise RuntimeError("X logger cannot write an already committed step")
        self.last_step = step
        payload = {
            (k if self.wrapped.tag is None else f"{self.wrapped.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wrapped.wandb.log(payload, step=step, commit=False)

    def close(self):
        if self.last_step >= 0:
            self.wrapped.wandb.log({}, commit=True)
        self.wrapped.close()


class GradVarMetricsProcessor(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self._grad_var_pending = None

    def clear_grad_var(self):
        self._grad_var_pending = None

    def configure_x(self, metadata):
        import torch.distributed as dist

        if (
            self.config.enable_wandb
            and dist.get_rank() == 0
            and (
                not isinstance(self.logger, LoggerContainer)
                or not any(isinstance(item, WandBLogger) for item in self.logger._loggers)
            )
        ):
            raise RuntimeError("X monitoring requires a successfully initialized W&B run")
        if not isinstance(self.logger, LoggerContainer):
            return
        self.logger._loggers = [
            XWandBLogger(item, metadata) if isinstance(item, WandBLogger) else item
            for item in self.logger._loggers
        ]

    def emit_probe_metrics(self, payload):
        self.logger.log(payload, payload["grad_var/step"])

    def queue_grad_var(self, step: int, metrics: dict[str, float]):
        if self._grad_var_pending is not None:
            raise RuntimeError("unconsumed gradient variance metrics")
        self._grad_var_pending = (step, metrics)

    def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        pending = self._grad_var_pending
        self.clear_grad_var()
        merged = dict(extra_metrics or {})
        if pending is not None:
            pending_step, metrics = pending
            if pending_step != step:
                raise RuntimeError("gradient variance metrics belong to a different step")
            if merged.keys() & metrics.keys():
                raise RuntimeError("gradient variance metric name collision")
            merged.update(metrics)
        return super().log(step, global_avg_loss, global_max_loss, grad_norm, merged)
