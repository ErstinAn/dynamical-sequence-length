"""CPU verification of frozen population, identity checks, and DP reductions."""


import hashlib
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import ParallelismConfig

from torchtitan_dynamic_seq_len.data.fixed_validation import (
    FixedBudgetValidator,
    FrozenValidationArtifact,
    file_sha256,
    sequence_batches,
)
from torchtitan_dynamic_seq_len.experiment import ExperimentValidator
from torchtitan_dynamic_seq_len.token_progress import TokenMilestone


def _write_artifact(root: Path, n: int = 24, length: int = 4) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tokenizer / name).write_text("{}")
    values = np.arange(n * length, dtype=np.int64).reshape(n, length) % 7
    arrays = {
        "input": values,
        "labels": (values + 1) % 7,
        "positions": np.broadcast_to(np.arange(length, dtype=np.int64), (n, length)),
    }
    manifest = {
        "schema_version": 1,
        "token_budget": n * length,
        "sequence_length": length,
        "num_sequences": n,
        "arrays": {},
        "tokenizer_files": {
            name: file_sha256(tokenizer / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
    }
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(root / filename, array)
        manifest["arrays"][name] = {"file": filename, "sha256": file_sha256(root / filename)}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 7)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(49).reshape(7, 7) / 11)
        self.seen: list[torch.Tensor] = []

    def forward(self, inputs, positions=None):
        self.seen.append(inputs.detach().cpu())
        return self.embedding(inputs)


def _loss(predictions, labels):
    return torch.nn.functional.cross_entropy(
        predictions.reshape(-1, 7), labels.flatten(), reduction="sum"
    ), {}


def _dims(dp_size=1, mesh=None):
    return SimpleNamespace(
        tp_enabled=False,
        pp_enabled=False,
        cp_enabled=False,
        dp_replicate=dp_size,
        dp_shard=1,
        get_mesh=lambda _: mesh,
    )


def _validator(root: Path, local_batch=7, dp_size=1, dp_rank=0, mesh=None,
               validator_class=FixedBudgetValidator, **config):
    manifest = json.loads((root / "manifest.json").read_text())
    return validator_class.Config(
        token_budget=manifest["token_budget"],
        artifact_path=str(root),
        eval_local_batch_size=local_batch,
        **config,
    ).build(
        parallelism=ParallelismConfig(),
        dp_world_size=dp_size,
        dp_rank=dp_rank,
        tokenizer=SimpleNamespace(tokenizer_path=str(root / "tokenizer")),
        parallel_dims=_dims(dp_size, mesh),
        loss_fn=_loss,
        validation_context=nullcontext,
        metrics_processor=SimpleNamespace(ntokens_since_last_log=0, log_validation=Mock()),
        seq_len=manifest["sequence_length"],
        local_batch_size=999,  # deliberately unrelated training local batch
    )


def _gloo_worker(rank, world_size, root, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("batch",))
        for local_batch in (5, 8):
            validator = _validator(Path(root), local_batch, world_size, rank, mesh)
            validator.validate([_Model()], 12)
            Path(root, f"result-{rank}-{local_batch}.json").write_text(
                json.dumps(validator.last_result)
            )
        point = TokenMilestone(index=1, step=3, threshold=100, tokens=108, excess=8)
        for expected_calls in (1, 0):
            validator = _validator(Path(root), dp_size=world_size, dp_rank=rank, mesh=mesh,
                                   validator_class=ExperimentValidator, token_intervals=50)
            validator.validate_milestone([_Model()], point, root)
            assert validator.metrics_processor.log_validation.call_count == expected_calls
        if rank == 0:
            receipt = json.loads(Path(root, "validation/point-001.json").read_text())
            assert receipt["evaluated_tokens"] == 96 and receipt["tokens"] == 108
    finally:
        dist.destroy_process_group()


class FixedBudgetValidationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = _write_artifact(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def artifact(self):
        return FrozenValidationArtifact(
            self.root,
            token_budget=96,
            sequence_length=4,
            tokenizer_path=self.root / "tokenizer",
        )

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_production_budget_population_identical_dp_and_batch_matrix(self):
        # Hash sequence IDs: membership/order determines exactly which frozen
        # input/label/position rows are consumed, without allocating 500 MB.
        reference = np.arange(936, dtype=np.int64)
        expected_hash = hashlib.sha256(reference.tobytes()).hexdigest()
        for dp in (1, 2, 4, 8):
            for batch in (32, 64, 128):
                with self.subTest(dp=dp, batch=batch):
                    actual = np.concatenate(
                        [
                            reference[indices]
                            for rank in range(dp)
                            for indices in sequence_batches(936, dp, rank, batch)
                        ]
                    )
                    self.assertEqual(hashlib.sha256(actual.tobytes()).hexdigest(), expected_hash)
                    self.assertEqual(actual.size * 16384, 15_335_424)

    def test_frozen_arrays_identical_and_partial_forward_not_padded(self):
        artifact = self.artifact()
        expected = {name: value.tobytes() for name, value in artifact.arrays.items()}
        for dp in (1, 2, 4, 8):
            for batch in (2, 5, 8):
                populations = {name: [] for name in expected}
                for rank in range(dp):
                    for inputs, labels in artifact.batches(
                        dp_world_size=dp,
                        dp_rank=rank,
                        local_batch_size=batch,
                    ):
                        for name, value in {**inputs, "labels": labels}.items():
                            populations[name].append(value.numpy())
                for name in expected:
                    self.assertEqual(np.concatenate(populations[name]).tobytes(), expected[name])

    def test_actual_nll_count_training_state_and_no_gradients(self):
        reference = None
        for batch in (5, 8, 32):
            validator = _validator(self.root, batch, steps=1)
            model = _Model()
            model.train()
            model.embedding.eval()  # deliberately mixed state must survive
            original = model.embedding.weight.detach().clone()
            validator.validate([model], step=42)
            result = validator.last_result
            self.assertEqual(result["evaluated_tokens"], 96)
            self.assertEqual(sum(inputs.shape[0] for inputs in model.seen), 24)
            self.assertTrue(model.training)
            self.assertFalse(model.embedding.training)
            self.assertIsNone(model.embedding.weight.grad)
            torch.testing.assert_close(model.embedding.weight, original, rtol=0, atol=0)
            expected, _ = _loss(
                model(torch.tensor(self.artifact().arrays["input"])),
                torch.tensor(self.artifact().arrays["labels"]),
            )
            self.assertAlmostEqual(result["loss"], expected.item() / 96, places=6)
            if reference is not None:
                self.assertAlmostEqual(result["loss"], reference, places=6)
            reference = result["loss"]
            metric_call = validator.metrics_processor.log_validation.call_args.kwargs
            self.assertEqual(
                metric_call["extra_metrics"]["validation_metrics/evaluated_tokens"], 96
            )
            # Every logger receives these, including TensorBoard.add_scalar.
            self.assertTrue(
                all(
                    isinstance(value, (int, float))
                    for value in metric_call["extra_metrics"].values()
                )
            )

    def test_nonfinite_fails_and_restores_eval_state(self):
        validator = _validator(self.root)
        validator.loss_fn = lambda p, labels: (p.sum() * float("nan"), {})
        model = _Model().eval()
        with self.assertRaises(FloatingPointError):
            validator.validate([model], 1)
        self.assertFalse(model.training)
        self.assertIsNone(validator.last_result)

    def test_real_chunked_loss_matches_full_logits_without_backward(self):
        artifact = self.artifact()
        inputs = torch.tensor(artifact.arrays["input"])
        labels = torch.tensor(artifact.arrays["labels"])
        for num_chunks in (2, 4):
            for local_batch in (5, 8):
                with self.subTest(num_chunks=num_chunks, local_batch=local_batch):
                    # As in the trainer's _skip_lm_head path, forward returns
                    # hidden states; the actual TorchTitan loss applies lm_head.
                    model = _Model()
                    model.lm_head = torch.nn.Linear(7, 7, bias=False)
                    with torch.no_grad():
                        model.lm_head.weight.copy_(torch.arange(49).reshape(7, 7) / 49)
                        full_logits = model.lm_head(model(inputs))
                        expected_nll, _ = _loss(full_logits, labels)
                    model.seen.clear()
                    model.train()
                    model.lm_head.eval()
                    parameters_before = {
                        name: param.detach().clone() for name, param in model.named_parameters()
                    }
                    calls = []

                    def inspect_head(module, args, output):
                        calls.append((args[0].shape, torch.is_grad_enabled()))
                        self.assertFalse(args[0].requires_grad)
                        self.assertFalse(output.requires_grad)

                    hook = model.lm_head.register_forward_hook(inspect_head)
                    loss = ChunkedLossWrapper.Config(num_chunks=num_chunks).build()
                    loss.set_lm_head(model.lm_head)
                    validator = _validator(self.root, local_batch)
                    validator.loss_fn = loss
                    try:
                        validator.validate([model], 23)
                    finally:
                        hook.remove()
                    self.assertEqual(validator.last_result["evaluated_tokens"], 96)
                    # ChunkedLossWrapper sums chunk NLLs in FP32 before the
                    # validator's FP64 accumulator; forward partitioning can
                    # change last-bit roundoff without changing token weights.
                    torch.testing.assert_close(
                        torch.tensor(validator.last_result["loss"]),
                        expected_nll / 96,
                        rtol=1e-6,
                        atol=1e-7,
                    )
                    self.assertEqual(
                        len(calls),
                        validator.last_result["forwards_per_rank"] * num_chunks,
                    )
                    self.assertTrue(all(shape[1] == 4 // num_chunks for shape, _ in calls))
                    self.assertFalse(any(grad_enabled for _, grad_enabled in calls))
                    self.assertEqual(sum(batch.shape[0] for batch in model.seen), 24)
                    self.assertTrue(model.training)
                    self.assertFalse(model.lm_head.training)
                    for name, param in model.named_parameters():
                        self.assertIsNone(param.grad)
                        torch.testing.assert_close(param, parameters_before[name], rtol=0, atol=0)

    def test_checksum_tampering_rejected(self):
        with (self.root / "labels.npy").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.artifact()

    def test_tokenizer_mismatch_rejected(self):
        (self.root / "tokenizer/tokenizer_config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "tokenizer fingerprint"):
            self.artifact()

    def test_insufficient_array_rejected_despite_valid_checksum(self):
        np.save(self.root / "labels.npy", np.zeros((23, 4), dtype=np.int64))
        self.manifest["arrays"]["labels"]["sha256"] = file_sha256(self.root / "labels.npy")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.artifact()

    def test_ignored_labels_rejected_despite_valid_checksum(self):
        labels = np.zeros((24, 4), dtype=np.int64)
        labels[0, 0] = -100
        np.save(self.root / "labels.npy", labels)
        self.manifest["arrays"]["labels"]["sha256"] = file_sha256(self.root / "labels.npy")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "negative/ignored"):
            self.artifact()

    def test_document_provenance_checksum_rejected(self):
        (self.root / "documents.jsonl").write_text("{}\n")
        self.manifest["evaluation_documents"] = {"file": "documents.jsonl", "sha256": "wrong"}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.artifact()

    def test_bad_budget_and_sequence_count_rejected(self):
        self.manifest["token_budget"] = 100
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "budget"):
            self.artifact()
        with self.assertRaisesRegex(ValueError, "divisible"):
            list(sequence_batches(10, 4, 0, 2))

    @unittest.skipUnless(
        os.environ.get("FIXED_VALIDATION_GLOO_TEST") == "1", "opt in to local Gloo sockets"
    )
    def test_real_two_rank_reduction_and_partial_batches(self):
        reference = _validator(self.root, 5)
        reference.validate([_Model()], 12)
        mp.start_processes(
            _gloo_worker,
            args=(2, str(self.root), str(self.root / "rendezvous")),
            nprocs=2,
            join=True,
            start_method="fork",
        )
        for rank in range(2):
            for batch in (5, 8):
                result = json.loads((self.root / f"result-{rank}-{batch}.json").read_text())
                self.assertEqual(result["evaluated_tokens"], 96)
                self.assertAlmostEqual(result["loss"], reference.last_result["loss"], places=6)


if __name__ == "__main__":
    unittest.main()
