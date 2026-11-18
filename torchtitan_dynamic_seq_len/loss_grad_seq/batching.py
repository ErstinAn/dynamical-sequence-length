"""Plan independent phase batches and regroup CPU crops without changing samples."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchPlan:
    loss_length: int
    grad_length: int
    global_batch: int
    loss_microbatch: int
    grad_microbatch: int
    loss_batches: int
    grad_batches: int


def plan_batches(
    *,
    loss_length,
    grad_length,
    global_batch,
    dp,
    loss_limit,
    grad_limit,
    loss_token_budget=0,
    grad_token_budget=0,
):
    if any(
        type(v) is not int or v < 1
        for v in (loss_length, grad_length, global_batch, dp, loss_limit, grad_limit)
    ):
        raise ValueError("Lengths, batch sizes and DP degree must be positive integers")
    if grad_length > loss_length:
        raise ValueError("grad_seq_length must not exceed loss_seq_length")
    if global_batch % dp:
        raise ValueError("Global batch must be divisible by DP degree")
    if any(type(v) is not int or v < 0 for v in (loss_token_budget, grad_token_budget)):
        raise ValueError("Phase token budgets must be nonnegative integers")
    local = global_batch // dp
    loss_cap = (
        min(loss_limit, loss_token_budget // loss_length) if loss_token_budget else loss_limit
    )
    grad_cap = (
        min(grad_limit, grad_token_budget // grad_length) if grad_token_budget else grad_limit
    )
    if loss_cap < 1 or grad_cap < 1:
        raise ValueError("Phase capacity cannot fit one sequence")
    loss_micro = min(local, loss_cap)
    grad_micro = min(local, grad_cap)
    return BatchPlan(
        loss_length,
        grad_length,
        global_batch,
        loss_micro,
        grad_micro,
        (local + loss_micro - 1) // loss_micro,
        (local + grad_micro - 1) // grad_micro,
    )


def regroup_batches(batches, microbatch):
    """Join only short CPU crops, preserving their order and aligned labels/positions."""
    if not batches or microbatch < 1:
        raise ValueError("Regrouping requires data and a positive microbatch")
    keys = tuple(batches[0][0])
    if any(set(inputs) != set(keys) for inputs, _ in batches):
        raise ValueError("All crop batches must have identical input keys")
    inputs = {key: torch.cat([row[key] for row, _ in batches]) for key in keys}
    labels = torch.cat([labels for _, labels in batches])
    return [
        (
            {key: value[start : start + microbatch] for key, value in inputs.items()},
            labels[start : start + microbatch],
        )
        for start in range(0, labels.shape[0], microbatch)
    ]
