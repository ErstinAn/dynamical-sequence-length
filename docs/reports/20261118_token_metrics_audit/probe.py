"""Inspect current token counters and emitted metrics with CPU fixtures."""

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from tests.test_dynamic_seq_bs import bs_trainer
from tests.test_fixed_validation import _Model, _validator, _write_artifact


def main():
    trainer = bs_trainer(log_freq=4)
    iterator = iter(trainer.dataloader)
    # Independent values for the fixture schedules. Do not derive this oracle
    # from the runtime scheduler or counter under inspection.
    rows = [(4, 2, 2), (4, 2, 2), (4, 4, 2), (8, 4, 1), (8, 4, 1), (8, 4, 1)]
    total = valid_total = 0
    result = {"training": [], "environment_reused": True, "gpu_used": False}
    for step, (global_batch, length, microbatch) in enumerate(rows, 1):
        trainer.step = step
        trainer._prepare_step()
        assert trainer.current_seq_len == length
        assert trainer.current_global_batch_size == global_batch
        assert trainer.current_microbatch_size == microbatch
        batches = [next(iterator) for _ in range(global_batch // microbatch)]
        batches[0][1][0, 0] = -100
        # Independent numerical oracle for the scalar CPU model's loss.
        weight = trainer.model_parts[0].weight.detach().clone()
        denominator = global_batch * length - 1
        expected_loss = sum(
            ((inputs["input"] * weight - labels)[labels != -100]).square().sum()
            for inputs, labels in batches
        ).item() / denominator
        trainer.train_step(trainer.batch_generator(batches))
        total += global_batch * length
        valid_total += denominator
        assert trainer.ntokens_seen == total
        assert trainer.n_valid_tokens_seen == valid_total
        row = dict(step=step, global_batch=global_batch, seq_len=length,
                   microbatch=microbatch, nominal_tokens=total, valid_targets=valid_total)
        if trainer.metrics_processor.should_log(step):
            payload = trainer.metrics_processor.logger.log.call_args.args[0]
            assert payload["n_tokens_seen"] == total
            assert abs(payload["loss_metrics/global_avg_loss"] - expected_loss) < 1e-7
            row["emitted_loss_keys"] = sorted(key for key in payload if "loss" in key)
            row["loss_matches_current_step"] = True
        result["training"].append(row)

        if step == 2:
            before = copy.deepcopy(trainer.state_dict())
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_artifact(root)
                validator = _validator(root)
                validator.metrics_processor = trainer.metrics_processor
                validator.validate([_Model()], step)
            payload = trainer.metrics_processor.logger.log.call_args.args[0]
            assert trainer.state_dict() == before
            assert validator.last_result["evaluated_tokens"] == 96
            result["validation_on_nonlogging_step"] = {
                "step": step,
                "training_nominal_tokens": total,
                "training_counters_unchanged": True,
                "evaluated_targets": validator.last_result["evaluated_tokens"],
                "emitted_loss_keys": sorted(key for key in payload if "loss" in key),
                "has_training_token_axis": "n_tokens_seen" in payload,
            }

    before = copy.deepcopy(trainer.state_dict())
    batches = [({"input": torch.ones(1, 4)}, torch.ones(1, 4)) for _ in range(8)]
    with patch.object(trainer.optimizers, "step", side_effect=RuntimeError("audit: before update")):
        try:
            trainer.train_step(trainer.batch_generator(batches))
        except RuntimeError as error:
            assert str(error) == "audit: before update"
        else:
            raise AssertionError("Expected the injected optimizer failure")
    assert trainer.state_dict() == before
    result["failure_before_optimizer_update_preserves_counters"] = True
    result["logging_contract_complete"] = False
    result["scope"] = "CPU fixtures execute production training and validation methods. No 1B GPU run or W&B upload."
    Path("/tmp/dynamic_seq_token_audit_probe_20261118.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: nominal and valid counts, current-step loss, validation isolation, and failure rollback")
    print("GAP: old loss keys remain and validation has no explicit training-token axis")


if __name__ == "__main__":
    main()
