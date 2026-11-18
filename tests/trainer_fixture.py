"""Build a tiny TorchTitan trainer for CPU gradient and recovery checks."""

import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.trainer import Trainer


def cpu_metrics(dims, log_freq=1):
    metrics = object.__new__(MetricsProcessor)
    metrics.config = MetricsProcessor.Config(log_freq=log_freq)
    metrics.parallel_dims = dims
    metrics.logger = Mock()
    metrics.num_flops_per_token = 1
    metrics.gpu_peak_flops = 1e12
    metrics.has_quantization = False
    metrics.time_last_log = time.perf_counter()
    metrics.ntokens_since_last_log = 1
    metrics.data_loading_times = [0.001]
    metrics.device_memory_monitor = Mock()
    metrics.device_memory_monitor.get_peak_stats.return_value = SimpleNamespace(
        max_active_gib=0,
        max_active_pct=0,
        max_reserved_gib=0,
        max_reserved_pct=0,
        num_alloc_retries=0,
        num_ooms=0,
    )
    metrics.color = SimpleNamespace(
        **dict.fromkeys(
            ("red", "green", "orange", "turquoise", "blue", "cyan", "magenta", "reset", "yellow"),
            "",
        )
    )
    return metrics


class OptimizerFixture:
    def __init__(self, model):
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        self.model = model
        self.gradients = []

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.gradients.append([p.grad.detach().clone() for p in self.model.parameters()])
        self.optimizer.step()


def tiny_trainer(*, k=4, b=2, seq=3, log_freq=1, model=None):
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            local_batch_size=b, global_batch_size=b * k, seq_len=seq, max_norm=0.01
        ),
        parallelism=SimpleNamespace(spmd_backend="default"),
    )
    trainer.parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, tp=1, cp=1, pp=1, ep=1, world_size=1
    )
    trainer.parallel_dims.get_optional_mesh = lambda name: None
    trainer.model_parts = [model if model is not None else torch.nn.Linear(seq, seq).double()]
    trainer.model_config = None
    trainer.device = torch.device("cpu")
    trainer.train_context = nullcontext
    trainer.gradient_accumulation_steps = k
    trainer.num_pipeline_parallel_microbatches = 1
    trainer.step = 1
    trainer.ntokens_seen = 0
    trainer.metrics_processor = cpu_metrics(trainer.parallel_dims, log_freq)
    trainer.optimizers = OptimizerFixture(trainer.model_parts[0])
    scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizers.optimizer, lambda n: 0.95**n)
    trainer.lr_schedulers = SimpleNamespace(
        step=scheduler.step,
        get_metrics=lambda: {"lr": scheduler.get_last_lr()[0]},
        scheduler=scheduler,
    )
    trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)

    def loss_fn(pred, labels, global_valid_tokens):
        mask = labels != IGNORE_INDEX
        return ((pred - labels)[mask].square().sum() / global_valid_tokens, None)

    trainer.loss_fn = loss_fn
    return trainer
