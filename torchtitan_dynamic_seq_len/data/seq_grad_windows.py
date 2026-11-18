"""Reproducible continuous crops, with a separate stream for each purpose."""

import hashlib

import torch


def crop_batch_at_offsets(inputs, labels, length, starts):
    """Slice aligned inputs/targets and rebase only the initial document fragment."""
    batch, capacity = labels.shape
    if not 0 < length <= capacity:
        raise ValueError("Crop length must be within the parent sequence")
    starts = torch.as_tensor(starts, dtype=torch.long, device=labels.device).reshape(batch, 1)
    if ((starts < 0) | (starts > capacity - length)).any():
        raise ValueError("Crop start is outside the parent sequence")
    indices = starts + torch.arange(length, device=labels.device)[None, :]
    cropped = {key: value.gather(1, indices) for key, value in inputs.items()}
    if "positions" in cropped:
        positions = cropped["positions"]
        initial_document = positions.eq(0).cumsum(dim=1).eq(0)
        cropped["positions"] = positions - initial_document * positions[:, :1]
    return cropped, labels.gather(1, indices)


def crop_batches(batches, length, *, seed, step, rank, stream):
    # Stateless seeds resume from the optimizer step. Probe order and enabled
    # lengths cannot change the training crops or the global torch RNG.
    key = f"seq-grad-v1:{seed}:{step}:{rank}:{stream}".encode()
    generator = torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63)
    )
    result = []
    for inputs, labels in batches:
        batch, capacity = labels.shape
        if not 0 < length <= capacity:
            raise ValueError("Crop length must be within the parent sequence")
        starts = torch.randint(capacity - length + 1, (batch, 1), generator=generator)
        result.append(crop_batch_at_offsets(inputs, labels, length, starts))
    return result
