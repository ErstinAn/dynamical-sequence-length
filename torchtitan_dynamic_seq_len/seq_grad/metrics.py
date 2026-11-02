"""Per-step scalar series and occasional W&B sequence-length profiles."""

from dataclasses import dataclass

from torchtitan.components.metrics import BaseLogger, LoggerContainer, WandBLogger

from torchtitan_dynamic_seq_len.dynamic_seq.metrics import DynamicSeqMetricsProcessor


class SeqGradWandBLogger(BaseLogger):
    def __init__(self, wrapped, profile_freq, minimal=False):
        self.wrapped = wrapped
        self.profile_freq = profile_freq
        self.minimal = minimal

    def log(self, metrics, step):
        prefix = "seq_grad_distance/s"
        if self.minimal:
            bounds = {
                "seq_grad_distance/s": "context-truncation bound",
                "seq_grad_distance_sq/s": "context-truncation bound sq",
            }
            payload = {
                f"{name}/seq_{int(key.removeprefix(source))}": value
                for key, value in metrics.items()
                for source, name in bounds.items()
                if key.startswith(source)
            }
            aliases = {
                "loss_metrics/global_avg_loss": "train/loss",
                "validation_metrics/loss": "val/loss",
                "time_metrics/end_to_end(s)": "time/step_seconds",
            }
            payload.update({new: metrics[old] for old, new in aliases.items() if old in metrics})
            if payload:
                self.wrapped.log(payload, step)
            return
        payload = dict(metrics)
        lengths = sorted(int(key.removeprefix(prefix)) for key in metrics if key.startswith(prefix))
        if lengths and self.profile_freq > 0 and (step == 1 or step % self.profile_freq == 0):
            for measure in ("distance", "relative_distance", "cosine"):
                payload[f"seq_grad_profile/{measure}"] = self.wrapped.wandb.plot.line_series(
                    xs=lengths,
                    ys=[[metrics[f"seq_grad_{measure}/s{length:05d}"] for length in lengths]],
                    keys=[f"step {step} (before update)"],
                    title=f"Gradient {measure}, step {step}",
                    xname="Sequence length",
                )
        # One W&B write includes scalars and profiles. TensorBoard receives only scalars.
        self.wrapped.log(payload, step)

    def close(self):
        self.wrapped.close()


class SeqGradMetricsProcessor(DynamicSeqMetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(DynamicSeqMetricsProcessor.Config):
        profile_freq: int = 100
        wandb_minimal: bool = False

    def _build_metric_logger(self, **kwargs):
        logger = super()._build_metric_logger(**kwargs)
        if isinstance(logger, LoggerContainer):
            logger._loggers = [
                SeqGradWandBLogger(
                    item, kwargs["config"].profile_freq, kwargs["config"].wandb_minimal
                )
                if isinstance(item, WandBLogger)
                else item
                for item in logger._loggers
            ]
        return logger

    def reset_window(self):
        super().reset_window()
        self.pending = None

    def queue(self, step, values):
        if self.pending is not None:
            raise RuntimeError("Unconsumed sequence gradient metrics")
        self.pending = (step, values)

    def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        merged = dict(extra_metrics or {})
        if self.pending is not None:
            pending_step, values = self.pending
            if pending_step != step or merged.keys() & values.keys():
                raise RuntimeError("Sequence gradient metric step or key collision")
            merged.update(values)
            length = values["seq_grad/train_length"]
            if f"seq_grad_distance/s{length:05d}" in values:
                merged[f"seq_grad_loss/s{length:05d}"] = global_avg_loss
        return super().log(step, global_avg_loss, global_max_loss, grad_norm, merged)
