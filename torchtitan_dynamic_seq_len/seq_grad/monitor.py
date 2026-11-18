"""Raw distances from one complete, synchronized reference gradient."""

import math
from dataclasses import dataclass, field
from typing import Literal

import torch

from torchtitan_dynamic_seq_len.grad_var.monitor import full_local_tensor

LENGTHS = [64, 256, 384, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
           1280, 1536, 2048, 4096]


@dataclass(kw_only=True, slots=True)
class SeqGradConfig:
    enable: bool = True
    interval: int = 50
    include_initial: bool = True
    reference_length: int = 16384
    probe_global_batch_size: int = 256
    probe_local_batch_size: int = 1
    lengths: list[int] = field(default_factory=lambda: list(LENGTHS))
    seed: int = 0
    sampling: Literal["independent", "paired_window"] = "independent"
    parent_population: Literal["packed", "single_document"] = "packed"

    def validate(self, dp):
        if self.sampling not in {"independent", "paired_window"}:
            raise ValueError("Unknown sequence probe sampling protocol")
        if self.parent_population not in {"packed", "single_document"}:
            raise ValueError("Unknown probe parent population")
        if self.parent_population == "single_document" and self.sampling != "paired_window":
            raise ValueError("Single-document parents require paired-window probes")
        for name in ("interval", "reference_length", "probe_global_batch_size",
                     "probe_local_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(dp) is not int or dp <= 0:
            raise ValueError("Probe DP degree must be positive")
        if self.probe_global_batch_size % (dp * self.probe_local_batch_size):
            raise ValueError("Probe global batch must be divisible by DP * local batch")
        if (not self.lengths or len(set(self.lengths)) != len(self.lengths)
                or any(type(s) is not int or not 0 < s <= self.reference_length
                       for s in self.lengths)):
            raise ValueError("Provide unique positive lengths <= reference_length")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Probe seed must be a nonnegative integer")

    def should_monitor(self, completed_updates):
        return self.enable and completed_updates % self.interval == 0 and (
            completed_updates > 0 or self.include_initial
        )


class SeqGradMonitor:
    CHUNK_ELEMENTS = 262144

    def __init__(self, named_parameters):
        self.parameters, seen = [], set()
        for name, parameter in named_parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                full_local_tensor(parameter)
                seen.add(id(parameter))
                self.parameters.append((name, parameter))
        if not self.parameters:
            raise ValueError("Sequence monitoring requires trainable parameters")
        self.reference = []

    @torch.no_grad()
    def capture(self):
        self.reset()
        for _, parameter in self.parameters:
            grad = full_local_tensor(parameter if parameter.grad is None else parameter.grad)
            dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
            reference = torch.zeros_like(grad, dtype=dtype)
            if parameter.grad is not None:
                reference.copy_(grad)
            self.reference.append(reference.flatten())
        self.distance()  # Reject a nonfinite reference immediately.

    @torch.no_grad()
    def distance(self):
        if len(self.reference) != len(self.parameters):
            raise RuntimeError("Capture the reference gradient first")
        total = torch.zeros((), dtype=torch.float64, device=self.reference[0].device)
        for (_, parameter), reference in zip(self.parameters, self.reference, strict=True):
            grad = None if parameter.grad is None else full_local_tensor(parameter.grad).flatten()
            for start in range(0, reference.numel(), self.CHUNK_ELEMENTS):
                ref = reference[start:start + self.CHUNK_ELEMENTS].double()
                current = (torch.zeros_like(ref) if grad is None
                           else grad[start:start + self.CHUNK_ELEMENTS].double())
                total.add_((current - ref).square().sum())
        value = total.item()
        if not math.isfinite(value):
            raise ValueError("Nonfinite sequence gradient distance")
        return math.sqrt(value)

    def reset(self):
        self.reference.clear()
