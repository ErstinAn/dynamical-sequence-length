"""Select sequence lengths from optimizer steps. The first step is 1."""

from bisect import bisect_left
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal


@dataclass(kw_only=True)
class SeqLenSchedulerConfig:
    kind: Literal[
        "constant", "increasing", "decreasing", "triangular", "cyclic", "piecewise_linear"
    ] = "constant"
    length: int = 1024
    lengths: list[int] = field(default_factory=lambda: [512, 1024, 1536, 2048])
    boundaries: list[int] = field(default_factory=list)
    """Each boundary is the last step at one length. An empty list gives equal step counts where possible."""
    anchor_steps: list[int] = field(default_factory=list)
    """For piecewise_linear, pair each step with the value at the same index in lengths."""
    repeat: bool = False
    """For piecewise_linear, repeat after the last anchor step."""
    period_steps: int = 1000
    """The period includes the first minimum but not the next minimum."""
    phase_offset_steps: int = 0
    """Advance a triangular schedule by this many steps within its period."""
    min_length: int = 512
    max_length: int = 1536
    """Without rounding, the endpoints and period must give an integer change per step."""
    round_to_multiple: int | None = None
    """Round cyclic or piecewise linear lengths to this multiple."""

    def build(self, training_steps: int):
        if training_steps < 1:
            raise ValueError("training_steps must be positive")
        if type(self.phase_offset_steps) is not int or self.phase_offset_steps < 0:
            raise ValueError("phase_offset_steps must be a nonnegative integer")
        if self.phase_offset_steps and self.kind != "triangular":
            raise ValueError("phase_offset_steps requires a triangular schedule")
        if self.kind == "piecewise_linear":
            if self.boundaries:
                raise ValueError("piecewise_linear uses anchor_steps, not boundaries")
            schedule = PiecewiseLinear(
                tuple(self.anchor_steps), tuple(self.lengths), self.round_to_multiple, self.repeat
            )
            if schedule.anchor_steps[-1] > training_steps:
                raise ValueError("The last anchor step must not exceed training_steps")
            return schedule
        if self.anchor_steps or self.repeat:
            raise ValueError("anchor_steps and repeat require a piecewise_linear schedule")
        if self.kind == "constant":
            return Constant(self.length)
        if self.kind == "cyclic":
            if self.boundaries:
                raise ValueError("cyclic uses period_steps, not boundaries")
            return Cyclic(
                self.min_length, self.max_length, self.period_steps, self.round_to_multiple
            )
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
            return Triangular(lengths, self.period_steps, self.phase_offset_steps)
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
class PiecewiseLinear:
    """Interpolate integer values between anchors, with an optional repeat."""

    anchor_steps: tuple[int, ...]
    values: tuple[int, ...]
    round_to_multiple: int | None = None
    repeat: bool = False

    def __post_init__(self):
        if (
            len(self.anchor_steps) < 2
            or len(self.anchor_steps) != len(self.values)
            or any(type(s) is not int or s < 1 for s in self.anchor_steps)
            or self.anchor_steps[0] != 1
            or any(a >= b for a, b in zip(self.anchor_steps, self.anchor_steps[1:]))
        ):
            raise ValueError("anchor_steps must start at 1 and increase, with one step per value")
        if any(type(value) is not int or value < 1 for value in self.values):
            raise ValueError("Anchor values must be positive integers")
        if type(self.repeat) is not bool:
            raise ValueError("repeat must be a boolean")
        multiple = self.round_to_multiple
        if multiple is not None and (type(multiple) is not int or multiple < 1):
            raise ValueError("round_to_multiple must be a positive integer")
        if any(value % (multiple or 1) for value in self.values):
            raise ValueError("Anchor values must be divisible by round_to_multiple")

    @cached_property
    def lengths(self):
        return tuple(sorted({self.length_for_step(s) for s in range(1, self.anchor_steps[-1] + 1)}))

    def length_for_step(self, step: int) -> int:
        _check_step(step)
        if self.repeat:
            step = (step - 1) % self.anchor_steps[-1] + 1
        index = bisect_left(self.anchor_steps, step)
        if index == 0:
            return self.values[0]
        if index == len(self.anchor_steps):
            return self.values[-1]
        start, end = self.anchor_steps[index - 1 : index + 1]
        low, high = self.values[index - 1 : index + 1]
        multiple = self.round_to_multiple or 1
        numerator = low * (end - step) + high * (step - start)
        denominator = (end - start) * multiple
        # Round to the nearest multiple. Exact ties select the larger value.
        return multiple * ((2 * numerator + denominator) // (2 * denominator))


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
    phase_offset_steps: int = 0

    def __post_init__(self):
        if self.period_steps < 2 * (len(self.lengths) - 1):
            raise ValueError("period_steps must allow visiting every ascending/descending level")
        if not 0 <= self.phase_offset_steps < self.period_steps:
            raise ValueError("phase_offset_steps must be within the period")

    def length_for_step(self, step: int) -> int:
        _check_step(step)
        edges = 2 * (len(self.lengths) - 1)
        phase = (
            ((step - 1 + self.phase_offset_steps) % self.period_steps) * edges // self.period_steps
        )
        index = min(phase, edges - phase)
        return self.lengths[index]


@dataclass(frozen=True)
class Cyclic:
    """Use a linear triangle, with optional rounding that preserves its mean."""

    min_length: int
    max_length: int
    period_steps: int
    round_to_multiple: int | None = None

    def __post_init__(self):
        if self.min_length <= 0 or self.max_length <= self.min_length:
            raise ValueError("cyclic requires 0 < min_length < max_length")
        if self.period_steps < 2 or self.period_steps % 2:
            raise ValueError("cyclic period_steps must be positive and even")
        if self.round_to_multiple is not None:
            multiple = self.round_to_multiple
            if not isinstance(multiple, int) or multiple < 1:
                raise ValueError("round_to_multiple must be a positive integer")
            if (
                self.period_steps % 4
                or any(length % multiple for length in (self.min_length, self.max_length))
                or (self.min_length + self.max_length) % (2 * multiple)
            ):
                raise ValueError(
                    "rounded cyclic requires a period divisible by 4 and endpoints "
                    "and midpoint divisible by round_to_multiple"
                )
        elif (self.max_length - self.min_length) % (self.period_steps // 2):
            raise ValueError("cyclic endpoints and period must give an integer change per step")

    @property
    def increment(self):
        if self.round_to_multiple is not None:
            raise ValueError("rounded cyclic does not have a constant integer increment")
        return (self.max_length - self.min_length) // (self.period_steps // 2)

    @property
    def lengths(self):
        if self.round_to_multiple is not None:
            return tuple(
                sorted({self.length_for_step(s) for s in range(1, self.period_steps // 2 + 2)})
            )
        return tuple(range(self.min_length, self.max_length + 1, self.increment))

    def length_for_step(self, step: int) -> int:
        _check_step(step)
        phase = (step - 1) % self.period_steps
        rise = min(phase, self.period_steps - phase)
        if self.round_to_multiple is None:
            return self.min_length + self.increment * rise
        half = self.period_steps // 2
        # Round the lower half, then reflect it about the midpoint.
        # Complementary lengths sum to the two endpoints, including rounding ties.
        offset = min(rise, half - rise)
        denominator = half * self.round_to_multiple
        numerator = (self.max_length - self.min_length) * offset
        units = (2 * numerator + denominator) // (2 * denominator)
        lower = self.min_length + units * self.round_to_multiple
        return lower if 2 * rise <= half else self.min_length + self.max_length - lower
