"""Calculate sample variance across synchronized accumulation groups.

The caller supplies cumulative gradients before clipping.
The gradients use the token count for the complete optimizer step.
Each group must contain the same number of tokens.
This module does not change gradients or perform distributed reductions."""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch.distributed.tensor import DTensor


@dataclass(kw_only=True, slots=True)
class GradVarMonitorConfig:
    enable: bool = False
    mode: str = "legacy"
    interval: int = 50
    include_initial: bool = True
    group_sequences: int = 128
    num_groups: int = 8
    replicate_groups: int = 0
    """Groups per independent local replicate; zero disables replicate output."""
    local_batch_size: int = 32
    sampling: str = "parent_prefix"
    parent_length: int = 16384
    seed: int = 0
    lengths: list[int] = field(
        default_factory=lambda: [64, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 4096]
    )
    fit_lengths: list[int] = field(default_factory=lambda: list(range(512, 1025, 64)))
    held_out_lengths: list[int] = field(default_factory=lambda: [64, 4096])

    @property
    def total_sequences(self):
        return self.group_sequences * self.num_groups

    def validate(self, dp_degree):
        if self.mode not in {"legacy", "x"}:
            raise ValueError("gradient variance mode must be legacy or x")
        if self.sampling not in {"parent_prefix", "independent_random_window"}:
            raise ValueError("Unknown X sampling mode")
        if (
            self.num_groups < 2
            or min(self.interval, self.group_sequences, self.local_batch_size) <= 0
            or (self.sampling == "parent_prefix" and self.parent_length <= 0)
        ):
            raise ValueError("X probes require num_groups >= 2 and positive interval and sizes")
        if (
            not self.lengths
            or len(set(self.lengths)) != len(self.lengths)
            or any(s <= 0 for s in self.lengths)
        ):
            raise ValueError("X probes require unique positive lengths")
        if self.sampling == "parent_prefix" and max(self.lengths) > self.parent_length:
            raise ValueError("Prefix probes require lengths <= parent_length")
        if dp_degree <= 0 or self.group_sequences % (dp_degree * self.local_batch_size):
            raise ValueError("group_sequences must be divisible by DP degree * local_batch_size")
        if self.replicate_groups and (
            self.replicate_groups < 2
            or self.num_groups % self.replicate_groups
            or self.num_groups // self.replicate_groups < 2
        ):
            raise ValueError(
                "replicate_groups must divide num_groups into at least two blocks of >= 2 groups"
            )
        if set(self.fit_lengths) & set(self.held_out_lengths) or not set(
            self.fit_lengths + self.held_out_lengths
        ) <= set(self.lengths):
            raise ValueError("fit and held-out lengths must be disjoint subsets of lengths")


def full_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Accept replicated tensors and degenerate (size-one) shards only."""
    if isinstance(tensor, DTensor):
        for axis, placement in enumerate(tensor.placements):
            if placement.is_replicate():
                continue
            if placement.is_shard() and tensor.device_mesh.size(axis) == 1:
                continue
            raise ValueError(f"gradient variance requires full replicas: {tensor.placements}")
        local = tensor.to_local()
        if local.shape != tensor.shape:
            raise ValueError("gradient variance received an incomplete local tensor")
        tensor = local
    if tensor.layout != torch.strided:
        raise ValueError("gradient variance requires dense gradients")
    if not tensor.is_floating_point():
        raise ValueError("gradient variance requires real floating-point gradients")
    return tensor.detach()


class GradVarMonitor:
    # FP64 temporaries are bounded independently of model size.
    CHUNK_ELEMENTS = 262144

    def __init__(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]], k: int):
        if k < 2:
            raise ValueError("gradient variance requires gradient_accumulation_steps >= 2")
        self.k = k
        self.parameters = []
        seen, names = set(), set()
        for name, parameter in named_parameters:
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            name = ".".join(
                part
                for part in name.split(".")
                if part
                not in {
                    "_orig_mod",
                    "_checkpoint_wrapped_module",
                }
            )
            if name in names:
                raise ValueError(f"gradient variance parameter name collision: {name}")
            full_local_tensor(parameter)  # Fail before training on actual shards.
            seen.add(id(parameter))
            names.add(name)
            self.parameters.append((name, parameter))
        if not self.parameters:
            raise ValueError("gradient variance requires trainable parameters")
        self.previous: dict[str, torch.Tensor] = {}
        device = full_local_tensor(self.parameters[0][1]).device
        self.m2 = torch.zeros(len(self.parameters), dtype=torch.float64, device=device)
        self.count = 0
        self.active = False

    def begin(self):
        self.count = 0
        self.m2.zero_()
        self.active = True

    def reset(self):
        # Retain allocations, but never carry a statistical state across steps.
        self.count = 0
        self.active = False

    @torch.no_grad()
    def observe(self):
        if not self.active or self.count >= self.k:
            raise RuntimeError("gradient variance capture outside an active accumulation step")
        j = self.count + 1
        for index, (name, parameter) in enumerate(self.parameters):
            grad = None if parameter.grad is None else full_local_tensor(parameter.grad)
            reference = full_local_tensor(parameter) if grad is None else grad
            if not reference.is_contiguous():
                raise ValueError(f"gradient variance requires contiguous gradients: {name}")
            previous = self.previous.get(name)
            # FP32 for ordinary training; retain FP64 inputs for numerical tests.
            dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
            if previous is None:
                previous = torch.empty(reference.numel(), dtype=dtype, device=reference.device)
                self.previous[name] = previous
            if previous.numel() != reference.numel() or previous.device != reference.device:
                raise ValueError(f"gradient variance parameter changed shape/device: {name}")
            flat = None if grad is None else grad.view(-1)
            for start in range(0, previous.numel(), self.CHUNK_ELEMENTS):
                stop = start + self.CHUNK_ELEMENTS
                old = previous[start:stop]
                current = torch.zeros_like(old) if flat is None else flat[start:stop]
                if j == 1:
                    # Detect nonfinite first samples even if later captures cancel them.
                    self.m2[index].add_(
                        torch.where(torch.isfinite(current).all(), 0.0, float("nan"))
                    )
                else:
                    # h_j = K(C_j-C_prev); mean_prev = K*C_prev/(j-1).
                    # Subtract nearby values before scaling, in FP64.
                    delta = current.to(torch.float64) - old.to(torch.float64)
                    delta.sub_(old.to(torch.float64) / (j - 1)).mul_(self.k)
                    self.m2[index].add_(delta.square().sum(), alpha=(j - 1) / j)
                old.copy_(current)
        self.count = j

    def finish(self) -> dict[str, float]:
        if not self.active or self.count != self.k:
            raise RuntimeError(f"gradient variance expected {self.k} captures, got {self.count}")
        values = (self.m2 / (self.k - 1)).tolist()
        for (name, _), value in zip(self.parameters, values, strict=True):
            if not math.isfinite(value):
                raise ValueError(f"nonfinite gradient variance for {name}")
        total = math.fsum(values)
        if not math.isfinite(total):
            raise ValueError("nonfinite global gradient variance")
        metrics = {"global_grad_var": total}
        metrics.update(
            {
                f"grad_var_monitor/grad_var_monitor_{name}_var": value
                for (name, _), value in zip(self.parameters, values, strict=True)
            }
        )
        self.active = False
        return metrics


class XMonitor(GradVarMonitor):
    """Welford trace estimator on complete synchronized group gradients.

    Reuse parameter selection and tensor guards, but never the legacy cumulative
    gradient reconstruction. One mean vector and one scalar M2 are retained.
    """

    def __init__(self, named_parameters, num_groups):
        super().__init__(named_parameters, num_groups)
        self.mean = self.previous
        self.m2 = self.m2.new_zeros(())

    @torch.no_grad()
    def observe_group_gradient(self):
        if not self.active or self.count >= self.k:
            raise RuntimeError("X capture outside an active event")
        j = self.count + 1
        for name, parameter in self.parameters:
            grad = None if parameter.grad is None else full_local_tensor(parameter.grad)
            reference = full_local_tensor(parameter) if grad is None else grad
            if not reference.is_contiguous():
                raise ValueError(f"X requires contiguous gradients: {name}")
            dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
            mean = self.mean.get(name)
            if mean is None:
                mean = torch.empty(reference.numel(), dtype=dtype, device=reference.device)
                self.mean[name] = mean
            if mean.numel() != reference.numel() or mean.device != reference.device:
                raise ValueError(f"X parameter changed shape/device: {name}")
            flat = None if grad is None else grad.view(-1)
            for start in range(0, mean.numel(), self.CHUNK_ELEMENTS):
                old = mean[start : start + self.CHUNK_ELEMENTS]
                current = (
                    torch.zeros_like(old)
                    if flat is None
                    else flat[start : start + self.CHUNK_ELEMENTS]
                )
                if j == 1:
                    self.m2.add_(torch.where(torch.isfinite(current).all(), 0.0, float("nan")))
                    old.copy_(current)
                else:
                    delta = current.to(torch.float64) - old.to(torch.float64)
                    self.m2.add_(delta.square().sum(), alpha=(j - 1) / j)
                    old.copy_(old.to(torch.float64) + delta / j)
        self.count = j

    def observe(self):
        raise RuntimeError("Use observe_group_gradient for independent X samples")

    def finish(self, group_token_count):
        if not self.active or self.count != self.k:
            raise RuntimeError(f"X expected {self.k} groups, got {self.count}")
        if group_token_count <= 0:
            raise ValueError("group_token_count must be positive")
        value = self.m2.item() * group_token_count / (self.k - 1)
        if not math.isfinite(value):
            raise ValueError("nonfinite X gradient variance")
        self.active = False
        return value
