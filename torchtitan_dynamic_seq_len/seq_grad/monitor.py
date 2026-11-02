"""Distances from a fixed, synchronized reference gradient before clipping."""

import math
from dataclasses import dataclass, field

import torch

from torchtitan_dynamic_seq_len.grad_var.monitor import full_local_tensor


@dataclass(kw_only=True, slots=True)
class SeqGradConfig:
    enable: bool = True
    interval: int = 1
    train_length: int = 2048
    lengths: list[int] = field(default_factory=lambda: [512, *range(1024, 2049, 128), 4096])
    seed: int = 12345
    relative_epsilon: float = 1e-12

    def should_monitor(self, step: int, total_steps: int) -> bool:
        return self.enable and (step in (1, total_steps) or step % self.interval == 0)


class SeqGradMonitor:
    CHUNK_ELEMENTS = 262144

    def __init__(self, named_parameters, epsilon=1e-12):
        self.parameters = []
        seen = set()
        for name, parameter in named_parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                full_local_tensor(parameter)
                seen.add(id(parameter))
                self.parameters.append((name, parameter))
        if not self.parameters:
            raise ValueError("Sequence gradient monitoring requires trainable parameters")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("relative_epsilon must be finite and positive")
        self.epsilon = epsilon
        self.reference = []

    @torch.no_grad()
    def capture(self):
        self.reference.clear()
        for _, parameter in self.parameters:
            grad = full_local_tensor(parameter if parameter.grad is None else parameter.grad)
            dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
            reference = torch.zeros_like(grad, dtype=dtype)
            if parameter.grad is not None:
                reference.copy_(grad)
            self.reference.append(reference.flatten())
        return self.compare()["grad_norm"]

    @torch.no_grad()
    def compare(self):
        if len(self.reference) != len(self.parameters):
            raise RuntimeError("Capture the reference gradient first")
        sums = torch.zeros(4, dtype=torch.float64, device=self.reference[0].device)
        for (_, parameter), reference in zip(self.parameters, self.reference, strict=True):
            grad = None if parameter.grad is None else full_local_tensor(parameter.grad).flatten()
            for start in range(0, reference.numel(), self.CHUNK_ELEMENTS):
                ref = reference[start : start + self.CHUNK_ELEMENTS].double()
                current = (
                    torch.zeros_like(ref)
                    if grad is None
                    else grad[start : start + self.CHUNK_ELEMENTS].double()
                )
                sums[0].add_((current - ref).square().sum())
                sums[1].add_(current.square().sum())
                sums[2].add_(ref.square().sum())
                sums[3].add_((current * ref).sum())
        distance_sq, norm_sq, ref_sq, dot = sums.tolist()
        if not all(math.isfinite(x) for x in (distance_sq, norm_sq, ref_sq, dot)):
            raise ValueError("Nonfinite sequence gradient statistics")
        norm, ref_norm = math.sqrt(norm_sq), math.sqrt(ref_sq)
        defined = norm > 0 and ref_norm > 0
        return {
            "distance": math.sqrt(distance_sq),
            "distance_sq": distance_sq,
            "relative_distance": math.sqrt(distance_sq) / (ref_norm + self.epsilon),
            "grad_norm": norm,
            "cosine": max(-1.0, min(1.0, dot / (norm * ref_norm))) if defined else 0.0,
            "cosine_defined": int(defined),
        }

    def reset(self):
        self.reference.clear()
