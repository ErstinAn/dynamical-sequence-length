"""Check the data configuration before the trainer starts."""

from dataclasses import dataclass

from torchtitan.trainer import Trainer

from torchtitan_dynamic_seq_len.data.fineweb100 import validation_directory
from torchtitan_dynamic_seq_len.data.fixed_validation import (
    FixedBudgetValidator,
    resolve_artifact_path,
)


@dataclass(kw_only=True, slots=True)
class ExperimentTrainerConfig(Trainer.Config):
    def __post_init__(self):
        super(ExperimentTrainerConfig, self).__post_init__()
        if self.dataloader.dataset != "fineweb_100bt":
            return
        if (
            not isinstance(self.validator, FixedBudgetValidator.Config)
            or self.validator.token_budget <= 0
        ):
            raise ValueError("FineWeb training requires the fixed-budget validator.")
        if (
            resolve_artifact_path(self.validator.artifact_path)
            != validation_directory().expanduser().resolve()
        ):
            raise ValueError(
                "Use the same artifact for training exclusions and validation. "
                "Set DYNAMIC_SEQ_LEN_VALIDATION_DIR. Leave validator.artifact_path empty."
            )
