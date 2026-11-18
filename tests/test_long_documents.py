"""Check split isolation, token caches, document windows, and checkpoint replay."""
# ruff: noqa: E402

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Match the CLI startup order. SQLite must load its Conda C++ runtime before Arrow.
import build_fineweb_long as pipeline
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rebuild_long_token_cache import rebuild
from tokenizers import Tokenizer
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_dynamic_seq_len.data.long_documents import (
    LongDocumentBatchDataLoader,
    LongDocumentDataLoader,
    LongDocumentDataset,
)


class LongDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "source"
        cls.output = Path(cls.tmp.name) / "artifact"
        (cls.root / "candidates").mkdir(parents=True)
        (cls.root / "extract_status").mkdir()
        cls.tokenizer = Tokenizer.from_file(str(pipeline.TOKENIZER / "tokenizer.json"))
        info = pipeline.tokenizer_provenance()
        plan = {"repo_id": "test/fixture", "revision": "test", "config": "test", "ranks": 1,
                "tokenizer": info, "tokenizer_sha256": info["files_sha256"]["tokenizer.json"],
                "train_token_target": 1200, "validation_minimum_tokens": 1,
                "validation_hash_modulus": 10000, "validation_hash_cutoff": 5000,
                "encoding": "raw", "source_candidate_note": "Synthetic fixture"}
        plan["plan_sha256"] = pipeline.identity(plan)
        pipeline.write_json(cls.root / "extraction_plan.json", plan)
        docs = []
        for i in range(40):
            text = f"Document {i}. " + "A sequence contains ordered tokens. " * (12 + i % 4)
            ids = cls.tokenizer.encode(text, add_special_tokens=False).ids
            url = f"https://example.org/article-{i // 2}"
            docs.append({"id": f"doc-{i}", "text": text, "url": url, "source_tokens": len(ids),
                         "llama3_tokens": len(ids), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "normalized_sha256": pipeline.normalized_hash(text),
                         "url_group": pipeline.url_group(url, str(i)), "source_path": "fixture.parquet",
                         "source_row": i, "dump": "test", "edu_score": 3.0, "input_ids": ids})
        # Same normalized text under another URL must not survive deduplication.
        duplicate = {**docs[0], "id": "copy", "url": "https://other.org/copy", "text": docs[0]["text"].replace(" ", "  ")}
        duplicate["input_ids"] = cls.tokenizer.encode(duplicate["text"], add_special_tokens=False).ids
        duplicate["llama3_tokens"] = len(duplicate["input_ids"])
        duplicate["text_sha256"] = hashlib.sha256(duplicate["text"].encode()).hexdigest()
        duplicate["url_group"] = pipeline.url_group(duplicate["url"], duplicate["id"])
        docs += [duplicate, dict(docs[1])]
        pq.write_table(pa.Table.from_pylist(docs, schema=pipeline.candidate_schema()), cls.root / "candidates/00000.parquet", row_group_size=7)
        pipeline.write_json(cls.root / "extract_status/rank-00.json", {
            "complete": True, "plan_sha256": plan["plan_sha256"], "files": ["candidates/00000.parquet"]})
        args = SimpleNamespace(root=cls.root, output=cls.output)
        original_thresholds = pipeline.THRESHOLDS
        try:
            pipeline.THRESHOLDS = (96, 80, 64)
            pipeline.select_stage(args)
        finally:
            pipeline.THRESHOLDS = original_thresholds
        pipeline.materialize_stage(args)
        pipeline.verify_stage(args)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dedup_split_and_budget(self):
        result = json.loads((self.output / "selection.json").read_text())
        self.assertEqual(result["duplicate_id_or_normalized_text_removed"], 2)
        self.assertGreaterEqual(result["splits"]["train"]["tokens"], 1200)
        self.assertLess(result["splits"]["train"]["tokens"], 1200 + 200)
        self.assertEqual(result["url_group_overlap"], 0)
        self.assertEqual(pipeline.url_group("http://www.example.org/a/?utm_source=x", "x"),
                         pipeline.url_group("https://example.org/a", "y"))

    def test_combined_pools_remove_cross_pool_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            second = Path(temporary) / "second"
            combined = Path(temporary) / "combined"
            shutil.copytree(self.root, second)
            args = SimpleNamespace(root=combined, candidate_root=[self.root, second],
                                   output=Path(temporary) / "artifact")
            pipeline.combine_stage(args)
            with patch.object(pipeline, "THRESHOLDS", (96, 80, 64)):
                pipeline.select_stage(args)
            selected = json.loads((combined / "selection.json").read_text())
            self.assertEqual(selected["duplicate_id_or_normalized_text_removed"], 44)
            pipeline.materialize_stage(args)
            pipeline.verify_stage(args)
            original = json.loads((self.output / "selection.json").read_text())
            self.assertEqual(selected["splits"], original["splits"])

    def test_snapshot_uses_only_completed_files_and_stays_fixed(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ["first", "second"]]
            for root in roots:
                shutil.copytree(self.root, root)
                plan = pipeline.load_plan(root)
                candidate = root / "candidates/00000.parquet"
                pipeline.write_json(candidate.with_suffix(".json"), {
                    "file": "candidates/00000.parquet", "sha256": pipeline.sha256(candidate),
                    "plan_sha256": plan["plan_sha256"], "source_path": "fixture.parquet"})
                (root / "candidates/00001.partial.parquet").write_bytes(b"incomplete")
                (root / "extract_status/rank-00.json").unlink()
            args = SimpleNamespace(root=Path(temporary) / "snapshot", candidate_root=roots, snapshot=True)
            pipeline.combine_stage(args)
            plan = pipeline.load_plan(args.root)
            frozen = pipeline.candidate_files(args.root, plan)
            self.assertEqual(len(frozen), 2)
            with patch.object(pipeline, "THRESHOLDS", (96, 80, 64)):
                pipeline.select_stage(args)
            selected = json.loads((args.root / "selection.json").read_text())
            self.assertGreaterEqual(selected["splits"]["train"]["tokens"], 1200)
            self.assertEqual(selected["duplicate_id_or_normalized_text_removed"], 44)
            # A later completed file must not change the frozen membership.
            shutil.copyfile(roots[0] / "candidates/00000.json", roots[0] / "candidates/00002.json")
            self.assertEqual(pipeline.candidate_files(args.root, plan), frozen)
            record_path = roots[0] / "candidates/00000.json"
            record = json.loads(record_path.read_text())
            record["sha256"] = "changed"
            pipeline.write_json(record_path, record)
            with self.assertRaises(ValueError):
                pipeline.candidate_files(args.root, plan)

    def test_windows_do_not_cross_documents(self):
        dataset = LongDocumentDataset(self.output, split="train", seq_len=32)
        expected = []
        for file in dataset.files:
            for doc in pq.read_table(self.output / file["text"]["path"]).to_pylist():
                ids = [dataset.bos] + self.tokenizer.encode(doc["text"], add_special_tokens=False).ids + [dataset.eos]
                expected += [ids[i:i + 33] for i in range(0, len(ids) - 32, 32)]
        observed = list(dataset)
        self.assertEqual(len(observed), len(expected))
        for (inputs, labels), ids in zip(observed, expected):
            self.assertEqual(inputs["input"].tolist(), ids[:-1])
            self.assertEqual(labels.tolist(), ids[1:])
            self.assertEqual(inputs["positions"].tolist(), list(range(32)))

    def test_resume_and_length_change(self):
        dataset = LongDocumentDataset(self.output, split="train", seq_len=16)
        iterator = iter(dataset)
        next(iterator)
        saved = dataset.state_dict()
        dataset.seq_len = 32
        expected = next(iterator)
        restored = LongDocumentDataset(self.output, split="train", seq_len=32)
        restored.load_state_dict(saved)
        actual = next(iter(restored))
        self.assertTrue(torch.equal(expected[0]["input"], actual[0]["input"]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        bad = {**saved, "manifest_sha256": "wrong"}
        with self.assertRaises(ValueError):
            restored.load_state_dict(bad)

    def test_parallel_partition(self):
        full = list(LongDocumentDataset(self.output, split="train", seq_len=32))
        ranks = [list(LongDocumentDataset(self.output, split="train", seq_len=32,
                                        dp_rank=i, dp_world_size=2)) for i in range(2)]
        self.assertEqual(sum(map(len, ranks)), len(full))

    def test_loader_build_and_checkpoint(self):
        tokenizer = HuggingFaceTokenizer(tokenizer_path=str(pipeline.TOKENIZER))
        config = LongDocumentDataLoader.Config(dataset_path=str(self.output), infinite=True)
        loader = config.build(dp_world_size=1, dp_rank=0, tokenizer=tokenizer,
                              seq_len=32, local_batch_size=2)
        iterator = iter(loader)
        next(iterator)
        state = loader.state_dict()
        expected = next(iterator)
        resumed = config.build(dp_world_size=1, dp_rank=0, tokenizer=tokenizer,
                               seq_len=32, local_batch_size=2)
        resumed.load_state_dict(state)
        actual = next(iter(resumed))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        with self.assertRaises(ValueError):
            loader.set_seq_len(31)

    def test_shared_text_rebuilds_identical_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "downloaded"
            shutil.copytree(self.output, copy)
            shutil.rmtree(copy / "tokens")
            (copy / "manifest.json").unlink()
            rebuild(copy, pipeline.TOKENIZER)
            pipeline.verify_stage(SimpleNamespace(output=copy))

    def test_ordered_package_rebuild_preserves_manifest_and_validation(self):
        from scripts.prepare_final_dataset import prepare
        from torchtitan_dynamic_seq_len.data.fixed_validation import FrozenValidationArtifact
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "final"
            prepare(self.output, final, pipeline.TOKENIZER, length=32, count=8)
            expected_manifest = (final / "manifest.json").read_bytes()
            downloaded = Path(temporary) / "downloaded"
            shutil.copytree(final, downloaded, symlinks=False)
            shutil.rmtree(downloaded / "tokens")
            (downloaded / "manifest.json").unlink()
            rebuild(downloaded, pipeline.TOKENIZER)
            self.assertEqual((downloaded / "manifest.json").read_bytes(), expected_manifest)
            validation = json.loads((downloaded / "validation/manifest.json").read_text())
            frozen = FrozenValidationArtifact(downloaded / "validation", sequence_length=32,
                token_budget=validation["token_budget"], tokenizer_path=pipeline.TOKENIZER)
            self.assertEqual(frozen.num_sequences, 8)
            self.assertEqual(validation["source"]["manifest_sha256"],
                             hashlib.sha256(expected_manifest).hexdigest())

    def test_dynamic_batch_shape_and_resume(self):
        tokenizer = HuggingFaceTokenizer(tokenizer_path=str(pipeline.TOKENIZER))
        config = LongDocumentBatchDataLoader.Config(dataset_path=str(self.output), infinite=True)
        def build():
            return config.build(dp_world_size=1, dp_rank=0, tokenizer=tokenizer,
                                seq_len=64, local_batch_size=1, snapshot_every_n_steps=500)
        loader = build()
        iterator = iter(loader)
        reference = LongDocumentDataset(self.output, split='train', seq_len=16, infinite=True)
        windows = iter(reference)
        for length, batch in [(16, 3), (32, 2), (64, 1), (8, 5)]:
            loader.set_batch_shape(seq_len=length, microbatch_size=batch)
            reference.seq_len = length
            inputs, labels = next(iterator)
            self.assertEqual(labels.shape, (batch, length))
            for row in range(batch):
                expected, expected_labels = next(windows)
                self.assertTrue(torch.equal(inputs['input'][row], expected['input']))
                self.assertTrue(torch.equal(labels[row], expected_labels))
        state = loader.state_dict()
        loader.set_batch_shape(seq_len=32, microbatch_size=3)
        expected = next(iterator)
        resumed = build()
        resumed.load_state_dict(state)
        resumed.set_batch_shape(seq_len=32, microbatch_size=3)
        actual = next(iter(resumed))
        self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_completed_build_resumes_without_rewriting(self):
        before = (self.output / "manifest.json").read_bytes()
        args = SimpleNamespace(root=self.root, output=self.output)
        pipeline.select_stage(args)
        pipeline.materialize_stage(args)
        self.assertEqual((self.output / "manifest.json").read_bytes(), before)

    def test_partial_build_reuses_complete_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            shutil.copytree(self.output, output)
            train = output / "tokens/train-00000.bin"
            modified = train.stat().st_mtime_ns
            (output / "manifest.json").unlink()
            (output / "tokens/validation-00000.bin").unlink()
            pipeline.materialize_stage(SimpleNamespace(root=self.root, output=output))
            self.assertEqual(train.stat().st_mtime_ns, modified)
            pipeline.verify_stage(SimpleNamespace(output=output))

    def test_parallel_shards_preserve_document_order_and_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            original = pipeline.plan_output_tasks
            with patch.object(pipeline, "plan_output_tasks", side_effect=lambda *args:
                              original(*args, shard_tokens=300)):
                pipeline.materialize_stage(SimpleNamespace(root=self.root, output=output))
            pipeline.verify_stage(SimpleNamespace(output=output))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertGreater(len(manifest["files"]), 2)
            for split in ["train", "validation"]:
                expected = list(LongDocumentDataset(self.output, split=split, seq_len=32))
                actual = list(LongDocumentDataset(output, split=split, seq_len=32))
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(actual, expected):
                    self.assertTrue(torch.equal(left[0]["input"], right[0]["input"]))
                    self.assertTrue(torch.equal(left[1], right[1]))


if __name__ == "__main__":
    unittest.main()
