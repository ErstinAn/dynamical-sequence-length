"""Select sequence lengths from optimizer steps. The first step is 1."""

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Literal


@dataclass(kw_only=True)
class SeqLenSchedulerConfig:
    kind: Literal["constant", "increasing", "decreasing", "triangular", "cyclic"] = "constant"
    length: int = 1024
    lengths: list[int] = field(default_factory=lambda: [512, 1024, 1536, 2048])
    boundaries: list[int] = field(default_factory=list)
    """Each boundary is the last step at one length. An empty list gives equal step counts where possible."""
    period_steps: int = 1000
    """The period includes the first minimum but not the next minimum."""
    min_length: int = 512
    max_length: int = 1536
    """These endpoints and period_steps must give an integer change at each step."""

    def build(self, training_steps: int):
        if training_steps < 1:
            raise ValueError("training_steps must be positive")
        if self.kind == "constant":
            return Constant(self.length)
        if self.kind == "cyclic":
            if self.boundaries:
                raise ValueError("cyclic uses period_steps, not boundaries")
            return Cyclic(self.min_length, self.max_length, self.period_steps)
        lengths = tuple(self.lengths)
        if (
            len(lengths) < 2
            or any(x <= 0 for x in lengths)
            or sorted(set(lengths)) != list(lengths)
        ):
            raise ValueError(
                "lengths must contain at least two strictly increasing positive values"
            )
        if self.kind == "triangular":
            if self.boundaries:
                raise ValueError("triangular uses period_steps, not boundaries")
            return Triangular(lengths, self.period_steps)
        boundaries = tuple(self.boundaries) or tuple(
            i * training_steps // len(lengths) for i in range(1, len(lengths))
        )
        if (
            len(boundaries) != len(lengths) - 1
            or sorted(set(boundaries)) != list(boundaries)
            or boundaries[0] < 1
            or boundaries[-1] >= training_steps
        ):
            raise ValueError("boundaries must be strictly increasing inside the training horizon")
        if self.kind == "increasing":
            return Increasing(lengths, boundaries)
        if self.kind == "decreasing":
            # Reverse the step order to keep the same token budget.
            return Decreasing(Increasing(lengths, boundaries), training_steps)
        raise ValueError(f"Unknown seq_len scheduler: {self.kind}")


def _check_step(step):
    if not isinstance(step, int) or step < 1:
        raise ValueError("optimizer step must be a positive integer (one-based)")


@dataclass(frozen=True)
class Constant:
    length: int

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError("length must be positive")

    @property
    def lengths(self):
        return (self.length,)

    def length_for_step(self, step: int) -> int:
        _check_step(step)
        return self.length


@dataclass(frozen=True)
class Increasing:
    lengths: tuple[int, ...]
    boundaries: tuple[int, ...]

    def length_for_step(self, step: int) -> int:
        _check_step(step)
        return self.lengths[bisect_left(self.boundaries, step)]


@dataclass(frozen=True)
class Decreasing:
    increasing: Increasing
    training_steps: int

    @property
    def lengths(self):
        return self.increasing.lengths

    def length_for_step(self, step: int) -> int:
        _check_step(step)
        return self.increasing.length_for_step(max(1, self.training_steps + 1 - step))


@dataclass(frozen=True)
class Triangular:
    lengths: tuple[int, ...]
    period_steps: int

    def __post_init__(self):
        if self.period_steps < 2 * (len(self.lengths) - 1):
            raise ValueError("period_steps must allow visiting every ascending/descending level")

    def length_for_step(self, step: int) -> int:
        _check_step(step)
        edges = 2 * (len(self.lengths) - 1)
        phase = ((step - 1) % self.period_steps) * edges // self.period_steps
        index = min(phase, edges - phase)
        return self.lengths[index]


@dataclass(frozen=True)
class Cyclic:
    """The length changes linearly at each step. The endpoints do not repeat at a change of direction."""

    min_length: int
    max_length: int
    period_steps: int

    def __post_init__(self):
        if self.min_length <= 0 or self.max_length <= self.min_length:
            raise ValueError("cyclic requires 0 < min_length < max_length")
        if self.period_steps < 2 or self.period_steps % 2:
            raise ValueError("cyclic period_steps must be positive and even")
        if (self.max_length - self.min_length) % (self.period_steps // 2):
            raise ValueError("cyclic endpoints and period must give an integer change per step")

    @property
    def increment(self):
        return (self.max_length - self.min_length) // (self.period_steps // 2)

    @property
    def lengths(self):
        return tuple(range(self.min_length, self.max_length + 1, self.increment))

    def length_for_step(self, step: int) -> int:
        _check_step(step)
        phase = (step - 1) % self.period_steps
        return self.min_length + self.increment * min(phase, self.period_steps - phase)
