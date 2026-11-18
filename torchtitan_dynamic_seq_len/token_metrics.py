"""Keep existing losses and plot them against cumulative nominal training tokens."""

from dataclasses import dataclass

from torchtitan.components.metrics import WandBLogger

from torchtitan_dynamic_seq_len.dynamic_seq_bs.metrics import DynamicSeqBSMetricsProcessor

LOSS_KEYS = ("loss_metrics/global_avg_loss", "loss_metrics/global_max_loss",
             "validation_metrics/loss")


def define_token_axes(container):
    for logger in getattr(container, "_loggers", ()):
        if isinstance(logger, WandBLogger):
            prefix = "" if logger.tag is None else f"{logger.tag}/"
            run = logger.wandb.run
            run.define_metric(prefix + "n_tokens_seen")
            for key in LOSS_KEYS:
                run.define_metric(prefix + key, step_metric=prefix + "n_tokens_seen",
                                  step_sync=False)


class TokenMetricsProcessor(DynamicSeqBSMetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(DynamicSeqBSMetricsProcessor.Config):
        enable_wandb: bool = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.training_tokens = 0
        define_token_axes(self.logger)

    def set_training_tokens(self, tokens):
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Training token count must be a nonnegative integer.")
        self.training_tokens = tokens

    def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        if extra_metrics is None or "n_tokens_seen" not in extra_metrics:
            raise ValueError("Training loss requires the cumulative nominal token count.")
        self.set_training_tokens(int(extra_metrics["n_tokens_seen"]))
        return super().log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics)

    def log_validation(self, loss, step, extra_metrics=None):
        merged = dict(extra_metrics or {})
        merged["n_tokens_seen"] = self.training_tokens
        return super().log_validation(loss, step, merged)
