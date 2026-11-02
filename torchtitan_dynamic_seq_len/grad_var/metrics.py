"""Merge variance scalars into TorchTitan's existing training log payload."""

from dataclasses import dataclass

from torchtitan.components.metrics import MetricsProcessor


class GradVarMetricsProcessor(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self._grad_var_pending = None

    def clear_grad_var(self):
        self._grad_var_pending = None

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
