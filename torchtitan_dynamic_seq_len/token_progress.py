"""Plan ten complete checkpoints from cumulative nominal training tokens."""

import json
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path

import torch.distributed as dist
from torchtitan.components.checkpoint import TRAIN_STATE, AsyncMode, CheckpointManager


@dataclass(frozen=True)
class TokenMilestone:
    index: int
    step: int
    threshold: int
    tokens: int
    excess: int


@dataclass(frozen=True)
class TokenPlan:
    budget: int
    cumulative: tuple[int, ...]
    milestones: tuple[TokenMilestone, ...]

    @classmethod
    def smoke(cls, increments, budget):
        increments = list(increments)
        if len(increments) != 8 or any(type(x) is not int or x <= 0 for x in increments):
            raise ValueError("The smoke test requires eight complete optimizer steps.")
        cumulative = [0]
        for value in increments:
            cumulative.append(cumulative[-1] + value)
        if type(budget) is not int or budget != cumulative[-1]:
            raise ValueError("Smoke token_budget must equal its complete eight-step schedule.")
        milestones = tuple(TokenMilestone(i, step, cumulative[step], cumulative[step], 0)
                           for i, step in enumerate((2, 4, 6, 8), 1))
        return cls(budget, tuple(cumulative), milestones)

    @classmethod
    def build(cls, increments, budget):
        if type(budget) is not int or budget < 10:
            raise ValueError("Set token_budget to an integer of at least 10 nominal tokens.")
        cumulative = [0]
        for value in increments:
            if type(value) is not int or value < 1:
                raise ValueError("Each optimizer step must have a positive integer token count.")
            cumulative.append(cumulative[-1] + value)
        if len(cumulative) < 2 or not cumulative[-2] < budget <= cumulative[-1]:
            raise ValueError("training.steps must end at the first complete step reaching token_budget.")
        milestones = []
        for index in range(1, 11):
            threshold = (budget * index + 9) // 10
            step = bisect_left(cumulative, threshold)
            milestones.append(TokenMilestone(index, step, threshold, cumulative[step],
                                             cumulative[step] - threshold))
        if len({item.step for item in milestones}) != 10:
            raise ValueError("One step crosses multiple checkpoint thresholds. Use ten distinct save steps.")
        return cls(budget, tuple(cumulative), tuple(milestones))

    def to_dict(self):
        return {"token_budget": self.budget, "steps": len(self.cumulative) - 1,
                "actual_tokens": self.cumulative[-1],
                "checkpoints": [asdict(item) for item in self.milestones]}


class TokenCheckpointManager(CheckpointManager):
    """Retain the upstream checkpoint payload and change only the save trigger."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        enable: bool = True
        interval: int = 1
        keep_latest_k: int = 0
        enable_first_step_checkpoint: bool = False
        last_save_model_only: bool = False
        initial_load_model_only: bool = False

        def validate_policy(self):
            if (not self.enable or self.load_only or self.keep_latest_k != 0
                    or self.last_save_model_only or self.last_save_in_hf
                    or self.create_seed_checkpoint or self.enable_first_step_checkpoint
                    or self.exclude_from_loading or self.initial_load_in_hf
                    or (self.initial_load_path and self.initial_load_model_only)
                    or self.async_mode != AsyncMode.DISABLED):
                raise ValueError("Token checkpoints require full synchronous saves, full recovery, and retention of all milestone states.")

    def __init__(self, config, **kwargs):
        self.enable = False
        config.validate_policy()
        self.trainer = kwargs["states"][TRAIN_STATE]
        self.milestones = {item.step: item for item in self.trainer.token_plan.milestones}
        super().__init__(config, **kwargs)

    def _should_save(self, curr_step, last_step=False):
        return self.enable and not self.load_only and curr_step in self.milestones

    def save(self, curr_step, last_step=False):
        expected = self.trainer.token_plan.cumulative[curr_step]
        if self.trainer.nominal_tokens_seen != expected:
            raise ValueError("Checkpoint token count differs from the completed schedule prefix.")
        saved = super().save(curr_step, last_step=last_step)
        if saved and (not dist.is_initialized() or dist.get_rank() == 0):
            path = Path(self._create_checkpoint_id(curr_step)) / "token_milestone.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(asdict(self.milestones[curr_step]), indent=2) + "\n")
            temporary.replace(path)
        return saved
