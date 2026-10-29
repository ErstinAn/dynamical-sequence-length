# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Define the models and the default training configuration."""

import os
from pathlib import Path

from datasets import load_dataset
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
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

from torchtitan_dynamic_seq_len.data.fineweb100 import register_fineweb100
from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator
from torchtitan_dynamic_seq_len.grad_var_metrics import GradVarMetricsProcessor
from torchtitan_dynamic_seq_len.grad_var_trainer import GradVarTrainer
from torchtitan_dynamic_seq_len.optimizer import ExperimentOptimizers

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


llama3_configs.setdefault("llama3_100m_llama2_split", _model)
llama3_configs.setdefault(
    "sequence_length_debug",
    lambda attn_backend: _model(attn_backend, dim=64, num_layers=2, num_heads=4, num_kv_heads=2),
)


def _trainer(*, flavor="llama3_100m_llama2_split", method="adamw"):
    config = GradVarTrainer.Config(
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
        metrics=GradVarMetricsProcessor.Config(log_freq=1),
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


def llama3_100m_llama2_muon() -> Trainer.Config:
    return _trainer(method="muon")


def llama3_debug() -> Trainer.Config:
    config = _trainer(flavor="sequence_length_debug")
    config.dataloader.prefetch_factor = None
    return config
