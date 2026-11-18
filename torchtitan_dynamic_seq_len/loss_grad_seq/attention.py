"""Select attention boundaries without changing RoPE positions."""

import torch
from torchtitan.models.common.attention import FlexAttention


def attention_positions(positions, document_masking):
    """Use one synthetic document for unmasked causal attention."""
    if document_masking:
        return positions
    return torch.arange(positions.shape[1], device=positions.device).expand_as(positions)


def select_attention(model, positions, document_masking):
    if not isinstance(model.config.first_attention.inner_attention, FlexAttention.Config):
        raise ValueError("Configurable document masking requires FlexAttention")
    return model.get_attention_masks(positions=attention_positions(positions, document_masking))
