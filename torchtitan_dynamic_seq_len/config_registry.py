# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Define the models and the default training configuration."""

import os
from pathlib import Path

from datasets import load_dataset
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataLoader
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm, compute_ffn_hidden_dim
from torchtitan.models.llama3 import (
    _EMBEDDING_SKIP_INIT,
    _NORM_INIT,
    _build_llama3_layers,
    _output_linear_init,
    llama3_configs,
    model_registry,
)
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.trainer import Trainer

from torchtitan_dynamic_seq_len.data.dynamic_seq_bs_loader import DynamicSeqBSDataLoader
from torchtitan_dynamic_seq_len.data.dynamic_seq_loader import DynamicSeqDataLoader
from torchtitan_dynamic_seq_len.data.fineweb100 import register_fineweb100
from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator
from torchtitan_dynamic_seq_len.dynamic_seq.metrics import DynamicSeqMetricsProcessor
from torchtitan_dynamic_seq_len.dynamic_seq.trainer import DynamicSeqLenTrainer
from torchtitan_dynamic_seq_len.dynamic_seq_bs.metrics import DynamicSeqBSMetricsProcessor
from torchtitan_dynamic_seq_len.dynamic_seq_bs.trainer import DynamicSeqBSTrainer
from torchtitan_dynamic_seq_len.grad_var.metrics import GradVarMetricsProcessor
from torchtitan_dynamic_seq_len.grad_var.trainer import GradVarTrainer
from torchtitan_dynamic_seq_len.optimizer import ExperimentOptimizers
from torchtitan_dynamic_seq_len.seq_grad.metrics import SeqGradMetricsProcessor
from torchtitan_dynamic_seq_len.seq_grad.trainer import SeqGradTrainer
from torchtitan_dynamic_seq_len.training_config import ExperimentTrainerConfig

ASSETS = Path(
    os.environ.get("DYNAMIC_SEQ_LEN_ASSETS", Path(__file__).resolve().parents[1] / "assets")
)
TOKENIZER = str(ASSETS / "hf/llama-tokenizer")
register_fineweb100()
DATASETS.setdefault(
    "sequence_length_smoke",
    DatasetConfig(
        path=str(Path(__file__).resolve().parents[1] / "tests/fixtures/text.jsonl"),
        loader=lambda path: load_dataset("json", data_files=path, split="train"),
        sample_processor=lambda sample: sample["text"],
    ),
)


def _model(attn_backend: str, *, dim=768, num_layers=12, num_heads=12, num_kv_heads=4):
    return Llama3Model.Config(
        vocab_size=32000,
        dim=dim,
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=32000,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=32000,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_llama3_layers(
            fuse_qkv=False,
            n_layers=num_layers,
            dim=dim,
            n_heads=num_heads,
            n_kv_heads=num_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            attn_backend=attn_backend,
            rope=ComplexRoPE.Config(dim=dim // num_heads, max_seq_len=40960, theta=500000),
        ),
    )


def _model_1b_llama2(attn_backend: str) -> Llama3Model.Config:
    """Reuse TorchTitan's 1B backbone with the experiment's 32k vocabulary."""
    config = llama3_configs["1B"](attn_backend=attn_backend)
    config.vocab_size = 32000
    config.tok_embeddings.num_embeddings = 32000
    config.lm_head.out_features = 32000
    return config


llama3_configs.setdefault("llama3_1b_llama2", _model_1b_llama2)
llama3_configs.setdefault("llama3_100m_llama2_split", _model)
llama3_configs.setdefault(
    "sequence_length_debug",
    lambda attn_backend: _model(attn_backend, dim=64, num_layers=2, num_heads=4, num_kv_heads=2),
)


def _trainer(
    *,
    flavor="llama3_100m_llama2_split",
    method="adamw",
    config_type=GradVarTrainer.Config,
    metrics_type=GradVarMetricsProcessor.Config,
):
    config = config_type(
        model_spec=model_registry(flavor),
        hf_assets_path=TOKENIZER,
        loss=ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config(global_vocab_size=32000)),
        optimizer=ExperimentOptimizers.Config(method=method),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        ),
        training=TrainingConfig(
            steps=10,
            global_batch_size=-1,
            local_batch_size=16,
            seq_len=1024,
            max_norm=1.0,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=None,
            decay_type="cosine",
            min_lr_factor=0.02,
        ),
        metrics=metrics_type(log_freq=1),
        validator=FixedBudgetValidator.Config(enable=False, freq=50, steps=20),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=500,
            folder="checkpoint",
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        debug=DebugConfig(seed=42, deterministic=False),
    )
    config.parallelism.data_parallel_replicate_degree = int(os.environ.get("WORLD_SIZE", "1"))
    config.parallelism.data_parallel_shard_degree = 1
    return config


def llama3_100m_llama2_adamw() -> Trainer.Config:
    return _trainer()


def llama3_100m_llama2_adamw_constant_lr() -> Trainer.Config:
    """Use the TorchTitan Trainer with constant learning rates from the first update."""
    config = _trainer(config_type=ExperimentTrainerConfig, metrics_type=MetricsProcessor.Config)
    # The TorchTitan factor is min_lr_factor + (1 - min_lr_factor) * decay.
    # No warmup and min_lr_factor=1 keep each learning rate constant.
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0, decay_type="linear", min_lr_factor=1.0
    )
    config.dataloader.num_workers = 0
    config.dataloader.prefetch_factor = None
    config.dataloader.persistent_workers = False
    return config


def llama3_100m_llama2_muon() -> Trainer.Config:
    return _trainer(method="muon")


def llama3_debug() -> Trainer.Config:
    config = _trainer(flavor="sequence_length_debug")
    config.dataloader.prefetch_factor = None
    return config


def llama3_100m_llama2_adamw_dynamic_seq() -> Trainer.Config:
    config = _trainer(
        config_type=DynamicSeqLenTrainer.Config,
        metrics_type=DynamicSeqMetricsProcessor.Config,
    )
    config.dataloader = DynamicSeqDataLoader.Config(dataset="c4", pin_memory=True)
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0, decay_type="linear", min_lr_factor=1.0
    )
    config.training.seq_len = 2048
    config.validator.eval_seq_len = 1024
    return config


def llama3_debug_dynamic_seq() -> Trainer.Config:
    config = llama3_100m_llama2_adamw_dynamic_seq()
    config.model_spec = model_registry("sequence_length_debug")
    config.dataloader.dataset = "sequence_length_smoke"
    config.training.local_batch_size = 2
    config.training.seq_len = 64
    config.training.steps = 8
    config.seq_len_scheduler.kind = "triangular"
    config.seq_len_scheduler.lengths = [16, 32, 64]
    config.seq_len_scheduler.period_steps = 4
    return config


def llama3_100m_llama2_adamw_dynamic_seq_bs() -> Trainer.Config:
    config = _trainer(
        config_type=DynamicSeqBSTrainer.Config,
        metrics_type=DynamicSeqBSMetricsProcessor.Config,
    )
    config.dataloader = DynamicSeqBSDataLoader.Config(dataset="c4", pin_memory=True)
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0, decay_type="linear", min_lr_factor=1.0
    )
    config.training.seq_len = 2048
    config.training.global_batch_size = 512
    config.validator.eval_seq_len = 1024
    return config


def llama3_debug_dynamic_seq_bs() -> Trainer.Config:
    config = llama3_100m_llama2_adamw_dynamic_seq_bs()
    config.model_spec = model_registry("sequence_length_debug")
    config.dataloader.dataset = "sequence_length_smoke"
    config.training.local_batch_size = 2
    config.training.global_batch_size = 8
    config.training.seq_len = 64
    config.training.steps = 8
    config.seq_len_scheduler.kind = "triangular"
    config.seq_len_scheduler.lengths = [16, 32, 64]
    config.seq_len_scheduler.period_steps = 4
    config.batch_size_scheduler.kind = "triangular"
    config.batch_size_scheduler.batch_sizes = [8, 16, 32]
    config.batch_size_scheduler.period_steps = 4
    config.microbatch_scheduler.kind = "triangular"
    config.microbatch_scheduler.batch_sizes = [2, 4]
    config.microbatch_scheduler.period_steps = 4
    return config


def llama3_100m_llama2_adamw_seq_grad() -> Trainer.Config:
    """Compare sequence lengths at each update on a shared long parent batch."""
    config = _trainer(
        config_type=SeqGradTrainer.Config, metrics_type=SeqGradMetricsProcessor.Config
    )
    config.dataloader = DynamicSeqDataLoader.Config(dataset="c4", pin_memory=True)
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0, decay_type="linear", min_lr_factor=1.0
    )
    config.training.seq_len = 16384
    config.training.local_batch_size = 1
    config.training.global_batch_size = 128
    config.optimizer.learning_rate = 0.0005
    config.validator.eval_seq_len = 1024
    return config


def llama3_debug_seq_grad() -> Trainer.Config:
    config = llama3_100m_llama2_adamw_seq_grad()
    config.model_spec = model_registry("sequence_length_debug")
    config.dataloader.dataset = "sequence_length_smoke"
    config.training.seq_len = 64
    config.training.global_batch_size = 8
    config.training.steps = 4
    config.seq_grad.train_length = 16
    config.seq_grad.lengths = [8, 16, 24, 32]
    return config


def _with_1b_llama2(config):
    """Select the upstream 1B backbone adapted to the existing Llama2 tokenizer."""
    config.model_spec = model_registry("llama3_1b_llama2")
    config.hf_assets_path = TOKENIZER
    config.loss.loss_fn.global_vocab_size = config.model_spec.model.vocab_size
    return config


def llama3_1b_llama2_adamw() -> Trainer.Config:
    """Use the 1B backbone with the existing gradient variance Trainer."""
    return _with_1b_llama2(_trainer())


def llama3_1b_llama2_adamw_constant_lr() -> Trainer.Config:
    """Use the native Trainer for a constant-LR 1B baseline."""
    config = llama3_100m_llama2_adamw_constant_lr()
    return _with_1b_llama2(config)


def llama3_1b_llama2_adamw_dynamic_seq() -> Trainer.Config:
    config = llama3_100m_llama2_adamw_dynamic_seq()
    return _with_1b_llama2(config)


def llama3_1b_llama2_adamw_dynamic_seq_bs() -> Trainer.Config:
    config = llama3_100m_llama2_adamw_dynamic_seq_bs()
    return _with_1b_llama2(config)


def llama3_1b_llama2_adamw_seq_grad() -> Trainer.Config:
    config = llama3_100m_llama2_adamw_seq_grad()
    return _with_1b_llama2(config)
