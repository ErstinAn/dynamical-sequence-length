"""Check loss namespaces and validation contexts against actual frozen data."""

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from tests.test_fixed_validation import _Model, _validator, _write_artifact
from tests.test_loss_grad_seq import dual_trainer
from torchtitan_dynamic_seq_len.loss_grad_seq.validation import LossLengthValidator


class LossGradValidationTest(unittest.TestCase):
    def test_fixed_validation_ignores_longer_and_shorter_training_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(root, n=24, length=8)
            validator = _validator(root, local_batch=5)
            validator.__class__ = LossLengthValidator
            validator.parent_artifact = validator.artifact
            validator.max_eval_length = 8
            validator.config = LossLengthValidator.Config(
                **{field.name: getattr(validator.config, field.name)
                   for field in fields(validator.config) if field.init},
                follow_loss_length=False,
            )
            model = _Model()
            for step, length in enumerate((16, 4, 32), start=1):
                validator.metrics_processor.phase_metrics = {"loss_grad/loss_seq_length": length}
                model.seen.clear()
                validator.validate([model], step)
                self.assertTrue(all(x.shape[1] == 8 for x in model.seen))
                self.assertEqual(validator.last_result["sequence_length"], 8)
                self.assertEqual(validator.last_result["evaluated_tokens"], 192)

    def test_fixed_and_dynamic_validation_lengths_and_exact_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(root, n=24, length=8)
            validator = _validator(root, local_batch=5)
            validator.__class__ = LossLengthValidator
            validator.parent_artifact = validator.artifact
            validator.max_eval_length = 8
            model = _Model()
            for step, length in enumerate((8, 4, 6, 8), start=1):
                validator.metrics_processor.phase_metrics = {"loss_grad/loss_seq_length": length}
                model.seen.clear()
                validator.validate([model], step)
                self.assertTrue(all(x.shape[1] == length for x in model.seen))
                self.assertEqual(validator.last_result["sequence_length"], length)
                self.assertEqual(validator.last_result["evaluated_tokens"], 24 * length)
                self.assertEqual(validator.metrics_processor.validation_seq_length, length)
                self.assertEqual(validator.config.token_budget, 192)

    def test_only_one_additional_training_loss_in_existing_namespace(self):
        trainer = dual_trainer()
        trainer.train_step(trainer.batch_generator(trainer.dataloader))
        values = trainer.metrics_processor.logger.log.call_args.args[0]
        self.assertIn("loss_metrics/long_avg_loss", values)
        self.assertIn("loss_metrics/short_seq_loss", values)
        self.assertNotIn("loss_grad/loss_long", values)
        self.assertNotIn("loss_grad/loss_crop", values)
        self.assertNotIn("long_seq_train_loss", values)
        trainer.metrics_processor.validation_seq_length = 8
        trainer.metrics_processor.log_validation(2.0, 1)
        values = trainer.metrics_processor.logger.log.call_args.args[0]
        self.assertEqual(values["validation_metrics/loss"], 2.0)
        self.assertEqual(values["validation_metrics/seq_len"], 8)
        self.assertNotIn("loss_metrics/short_seq_loss", values)


if __name__ == "__main__":
    unittest.main()
