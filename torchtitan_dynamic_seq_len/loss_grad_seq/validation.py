"""Evaluate held-out sequences at a fixed length or the current long-loss length."""

import hashlib

import numpy as np
from torchtitan.components.loss import IGNORE_INDEX
from dataclasses import dataclass, replace

from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator


class ValidationPrefixView:
    """Keep the same held-out parent sequences and select a prefix of each."""

    def __init__(self, source, length):
        if not 0 < length <= source.manifest["sequence_length"]:
            raise ValueError("Validation length must fit the frozen parent artifact")
        self.source, self.length = source, length
        self.num_sequences = source.num_sequences
        self.token_budget = (
            int(np.count_nonzero(source.arrays["labels"][:, :length] != IGNORE_INDEX))
            if hasattr(source, "arrays") else self.num_sequences * length
        )
        self.manifest_sha256 = hashlib.sha256(
            f"{source.manifest_sha256}:prefix:{length}".encode()
        ).hexdigest()

    def batches(self, **kwargs):
        for inputs, labels in self.source.batches(**kwargs):
            yield (
                {key: value[:, : self.length].contiguous() for key, value in inputs.items()},
                labels[:, : self.length].contiguous(),
            )


class LossLengthValidator(FixedBudgetValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(FixedBudgetValidator.Config):
        document_masking: bool = True
        follow_loss_length: bool = True
        """False evaluates every frozen window at eval_seq_len regardless of the training schedule."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.parent_artifact = self.artifact
        if self.artifact is not None and config.dataloader.dataset == "fineweb_edu_long":
            if self.artifact.manifest["source"].get("split") != "validation":
                raise ValueError("Frozen validation requires the validation split")
            expected = getattr(config.dataloader, "expected_manifest_sha256", "")
            if expected and self.artifact.manifest["source"].get("manifest_sha256") != expected:
                raise ValueError("Frozen validation source manifest differs")
        if getattr(config.dataloader, "packing", "none") == "stream":
            from torchtitan_dynamic_seq_len.data.packed_documents import PACKING_PROTOCOL
            if self.artifact is None or self.artifact.manifest.get("packing_protocol") != PACKING_PROTOCOL:
                raise ValueError("Packed validation requires a matching packed artifact")
            if self.artifact.manifest["source"].get("split") != "validation":
                raise ValueError("Packed validation requires the validation split")
            expected = getattr(config.dataloader, "expected_manifest_sha256", "")
            if expected and self.artifact.manifest["source"].get("manifest_sha256") != expected:
                raise ValueError("Packed validation source manifest differs")
        self.max_eval_length = self.seq_len
        self.local_batch_size = config.eval_local_batch_size

    def post_dataloading_process(self, input_dict, labels, model_parts):
        if getattr(self.config, "document_masking", True):
            return super().post_dataloading_process(input_dict, labels, model_parts)
        from torchtitan_dynamic_seq_len.loss_grad_seq.attention import select_attention
        positions = input_dict["positions"]
        return input_dict["input"], labels, {
            "positions": positions,
            "attention_masks": select_attention(model_parts[0], positions, False),
        }

    def validate(self, model_parts, step):
        length = (self.metrics_processor.phase_metrics["loss_grad/loss_seq_length"]
                  if getattr(self.config, "follow_loss_length", True) else self.max_eval_length)
        if not 0 < length <= self.max_eval_length:
            raise ValueError("Long loss length exceeds validation artifact capacity")
        self.seq_len = length
        self.metrics_processor.validation_seq_length = length
        original_config = self.config
        if self.parent_artifact is not None:
            self.artifact = (
                self.parent_artifact
                if length == self.max_eval_length
                else ValidationPrefixView(self.parent_artifact, length)
            )
            self.config = replace(self.config, token_budget=self.artifact.token_budget)
        try:
            super().validate(model_parts, step)
            if self.last_result is not None:
                self.last_result["sequence_length"] = length
        finally:
            self.config = original_config
