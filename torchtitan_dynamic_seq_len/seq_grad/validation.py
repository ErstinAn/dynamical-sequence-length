"""Use a fixed evaluation length independent of the training window length."""

from dataclasses import dataclass

from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator


class SeqGradValidator(FixedBudgetValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(FixedBudgetValidator.Config):
        pass

    def __init__(self, config, **kwargs):
        # SeqGradTrainer checks the intrinsic position limit before construction
        # and expands RoPE storage before any training or validation forward.
        kwargs['seq_len'] = max(kwargs['seq_len'], config.eval_seq_len)
        super().__init__(config, **kwargs)
        if self.artifact is not None and config.dataloader.dataset == 'fineweb_edu_long':
            source = self.artifact.manifest['source']
            expected = config.dataloader.expected_manifest_sha256
            if source.get('split') != 'validation' or (expected and source.get('manifest_sha256') != expected):
                raise ValueError('Fixed validation source differs from the configured dataset')
