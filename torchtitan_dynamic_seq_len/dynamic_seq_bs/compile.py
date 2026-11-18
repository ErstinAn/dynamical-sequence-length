"""Use dynamic shapes for compiled blocks in joint batch and length training."""

import torch
from torchtitan.tools.logging import logger


def compile_dynamic_blocks(model_parts, config):
    if not config.enable or "model" not in config.components:
        return
    # Attention head dimensions are model constants, not schedule dimensions.
    # FlexAttention requires concrete head dimensions when it selects kernels.
    # FSDP bypasses module-class mark_static annotations in the pinned version.
    head_dim_source = r".*\.head_dim"
    sources = torch.compiler.config.static_sources.split(",")
    if head_dim_source not in sources:
        torch.compiler.config.static_sources = ",".join(filter(None, [*sources, head_dim_source]))
    for model in model_parts:
        for block in model.layers.values():
            # Replace the lazy wrapper before its first call. Parameters and
            # activation checkpoint wrappers remain unchanged.
            block.compile(backend=config.backend, fullgraph=True, dynamic=True)
    logger.info("Joint batch and sequence training uses dynamic block compilation")
