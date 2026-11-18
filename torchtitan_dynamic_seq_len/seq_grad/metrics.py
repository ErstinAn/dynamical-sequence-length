"""Log one raw distance per length at an explicit completed-update coordinate."""

import sys
from dataclasses import dataclass

import torch.distributed as dist
from torchtitan.components.metrics import BaseLogger, LoggerContainer, MetricsProcessor, WandBLogger


class SeqGradWandBLogger(BaseLogger):
    def __init__(self, wrapped, metadata):
        self.wrapped = wrapped
        self.last_step = -1
        run = wrapped.wandb.run
        run.define_metric("context_bias/step")
        run.define_metric("context_bias/distance_s*", step_metric="context_bias/step")
        run.config.update({"seq_grad_probe": metadata})

    def log(self, metrics, step):
        if step < self.last_step:
            raise RuntimeError("Sequence logger cannot write an already committed step")
        self.last_step = step
        # Keep this step open for post-update probes and validation.
        payload = {(k if self.wrapped.tag is None else f"{self.wrapped.tag}/{k}"): v
                   for k, v in metrics.items()}
        self.wrapped.wandb.log(payload, step=step, commit=False)

    def close(self):
        if sys.exception() is not None:
            self.wrapped.wandb.finish(exit_code=1)
            return
        if self.last_step >= 0:
            self.wrapped.wandb.log({}, commit=True)
        self.wrapped.close()


class SeqGradMetricsProcessor(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def configure_probe(self, metadata):
        loggers = self.logger._loggers if isinstance(self.logger, LoggerContainer) else []
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.config.enable_wandb and rank == 0 and not any(
            isinstance(item, WandBLogger) for item in loggers
        ):
            raise RuntimeError("Sequence monitoring requires a successfully initialized W&B run")
        if isinstance(self.logger, LoggerContainer):
            self.logger._loggers = [
                SeqGradWandBLogger(item, metadata) if isinstance(item, WandBLogger) else item
                for item in loggers
            ]

    def emit_probe_metrics(self, payload):
        self.logger.log(payload, payload["context_bias/step"])
