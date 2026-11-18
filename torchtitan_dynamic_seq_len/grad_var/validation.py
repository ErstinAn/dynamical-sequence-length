"""Evaluate a fixed context length without changing the X training length."""

from dataclasses import dataclass

from torchtitan.models.common.rope import RoPE

from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator


def required_context_length(config):
    validation_length = (
        config.validator.eval_seq_len or config.training.seq_len if config.validator.enable else 0
    )
    return max(config.training.seq_len, *config.grad_var_monitor.lengths, validation_length)


class XValidator(FixedBudgetValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(FixedBudgetValidator.Config):
        pass

    def __init__(self, config, **kwargs):
        # The base Trainer passes the training length. X initialization allocates
        # the larger cache before the first probe or validation forward.
        kwargs["seq_len"] = max(kwargs["seq_len"], config.eval_seq_len)
        super().__init__(config, **kwargs)

    def _validate(self, model_parts, step):
        for model in model_parts:
            for module in model.modules():
                if isinstance(module, RoPE) and module.cache.shape[0] < self.seq_len:
                    raise ValueError("Validation length exceeds the allocated RoPE cache")
        return super()._validate(model_parts, step)
