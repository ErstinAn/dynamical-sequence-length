"""Keep long loss in native metrics and add the short-sequence training loss."""

import time
from dataclasses import dataclass

from torchtitan_dynamic_seq_len.dynamic_seq.metrics import DynamicSeqMetricsProcessor


class LongLossLogger:
    """Rename native loss keys only for this processor's log call."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def log(self, values, step):
        values = dict(values)
        for old, new in (("global_avg_loss", "long_avg_loss"), ("global_max_loss", "long_max_loss")):
            key = "loss_metrics/" + old
            if key in values:
                values["loss_metrics/" + new] = values.pop(key)
        return self.wrapped.log(values, step)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class LossGradMetricsProcessor(DynamicSeqMetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(DynamicSeqMetricsProcessor.Config):
        pass

    def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        merged = dict(extra_metrics or {})
        merged.update(self.phase_metrics)
        merged.update(
            {
                "loss_metrics/short_seq_loss": global_avg_loss,
                "loss_grad/grad_phase_seconds": time.perf_counter() - self.grad_phase_started,
            }
        )
        original = self.logger
        self.logger = LongLossLogger(original)
        try:
            return super().log(step, self.long_loss, self.long_loss_max, grad_norm, merged)
        finally:
            self.logger = original

    def log_validation(self, loss, step, extra_metrics=None):
        merged = dict(extra_metrics or {})
        merged["validation_metrics/seq_len"] = self.validation_seq_length
        return super().log_validation(loss, step, merged)
