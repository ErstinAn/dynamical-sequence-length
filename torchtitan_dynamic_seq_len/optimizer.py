"""Build AdamW or Muon with the experiment parameter groups."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer


class ExperimentOptimizers(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        implementation: Literal["foreach"] = "foreach"
        method: Literal["adamw", "muon"] = "adamw"
        learning_rate: float = 0.004
        weight_decay: float = 0.1
        vector_weight_decay: float = 0.0
        beta1: float = 0.9
        beta2: float = 0.95
        epsilon: float = 1e-8
        vector_beta1: float = 0.9
        vector_beta2: float = 0.95
        vector_epsilon: float = 1e-8
        momentum: float = 0.9
        nesterov: bool = True
        ns_steps: int = 5
        adjust_lr_fn: Literal["match_rms_adamw", "original"] = "match_rms_adamw"

    def __init__(self, config: Config, *, model_parts: list[nn.Module]):
        if config.method not in ("adamw", "muon"):
            raise ValueError("Set optimizer.method to adamw or muon.")
        if config.param_groups or config.implementation != "foreach":
            raise ValueError(
                "This adapter uses fixed parameter groups and foreach AdamW. "
                "Do not set param_groups or implementation."
            )
        self.model_parts = model_parts
        self.optimizers = []
        all_params = []
        for model in model_parts:
            special_ids = set()
            for name in ("tok_embeddings", "lm_head", "norm"):
                module = getattr(model, name, None)
                if module is not None:
                    special_ids.update(id(p) for p in module.parameters())
            matrices, vectors = [], []
            names = {}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                names[param] = canonical_fqn(name)
                # The vector group also contains the embedding and output matrices.
                group = vectors if id(param) in special_ids or param.ndim < 2 else matrices
                group.append(param)
            if matrices:
                if config.method == "muon":
                    matrix_optimizer = torch.optim.Muon(
                        matrices,
                        lr=config.learning_rate,
                        weight_decay=config.weight_decay,
                        momentum=config.momentum,
                        nesterov=config.nesterov,
                        ns_steps=config.ns_steps,
                        adjust_lr_fn=config.adjust_lr_fn,
                    )
                else:
                    matrix_optimizer = torch.optim.AdamW(
                        matrices,
                        lr=config.learning_rate,
                        betas=(config.beta1, config.beta2),
                        eps=config.epsilon,
                        weight_decay=config.weight_decay,
                        foreach=True,
                    )
                self.optimizers.append(matrix_optimizer)
            if vectors:
                self.optimizers.append(
                    torch.optim.AdamW(
                        vectors,
                        lr=config.learning_rate,
                        betas=(config.vector_beta1, config.vector_beta2),
                        eps=config.vector_epsilon,
                        weight_decay=config.vector_weight_decay,
                        foreach=True,
                    )
                )
            for optimizer in self.optimizers:
                for group in optimizer.param_groups:
                    if "param_names" not in group:
                        group["param_names"] = [names[p] for p in group["params"]]
            all_params.extend(matrices + vectors)
        self._validate_params(all_params)
        self._post_init(all_params)
