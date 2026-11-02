"""Select global or per-device batch sizes from optimizer steps."""

from dataclasses import dataclass, field
from typing import Literal

from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig


@dataclass(kw_only=True)
class BatchSizeSchedulerConfig:
    kind: Literal[
        "constant", "increasing", "decreasing", "triangular", "cyclic", "piecewise_linear"
    ] = "constant"
    batch_size: int = 16
    batch_sizes: list[int] = field(default_factory=lambda: [8, 16, 32])
    boundaries: list[int] = field(default_factory=list)
    anchor_steps: list[int] = field(default_factory=list)
    repeat: bool = False
    period_steps: int = 1000
    phase_offset_steps: int = 0
    min_batch_size: int = 8
    max_batch_size: int = 32
    round_to_multiple: int | None = None

    def build(self, training_steps: int):
        # Use the existing integer schedules for both batch settings.
        schedule = SeqLenSchedulerConfig(
            kind=self.kind,
            length=self.batch_size,
            lengths=self.batch_sizes,
            boundaries=self.boundaries,
            anchor_steps=self.anchor_steps,
            repeat=self.repeat,
            period_steps=self.period_steps,
            phase_offset_steps=self.phase_offset_steps,
            min_length=self.min_batch_size,
            max_length=self.max_batch_size,
            round_to_multiple=self.round_to_multiple,
        ).build(training_steps)
        return BatchSizeScheduler(schedule)


@dataclass(frozen=True)
class BatchSizeScheduler:
    schedule: object

    def value_for_step(self, step: int) -> int:
        return self.schedule.length_for_step(step)
