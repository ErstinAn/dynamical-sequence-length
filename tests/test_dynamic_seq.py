"""Do CPU checks of schedules, token packing, checkpoint recovery, and training."""

import copy
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from datasets import Dataset
from torchtitan.config import ConfigManager
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.trainer import Trainer

from tests.test_grad_var_trainer import cpu_metrics, tiny_trainer
from torchtitan_dynamic_seq_len.data.dynamic_seq_loader import DynamicSeqDataLoader
from torchtitan_dynamic_seq_len.dynamic_seq_metrics import DynamicSeqMetricsProcessor
from torchtitan_dynamic_seq_len.dynamic_seq_trainer import DynamicSeqLenTrainer
from torchtitan_dynamic_seq_len.run_config import grid, load_config, resolve_point
from torchtitan_dynamic_seq_len.seq_len_scheduler import SeqLenSchedulerConfig


class IntegerTokenizer:
    def encode(self, text, **kwargs):
        return list(map(int, text.split()))


def loader(*, streaming=False, infinite=False, b=2, documents=None, capacity=8):
    documents = documents or [" ".join(map(str, range(101)))]
    dataset = Dataset.from_dict({"text": documents})
    if streaming:
        dataset = dataset.to_iterable_dataset()
    DATASETS["dynamic_test"] = DatasetConfig(
        path="unused", loader=lambda path: dataset, sample_processor=lambda row: row["text"]
    )
    return DynamicSeqDataLoader(
        DynamicSeqDataLoader.Config(dataset="dynamic_test", infinite=infinite, pin_memory=False),
        dp_world_size=1,
        dp_rank=0,
        tokenizer=IntegerTokenizer(),
        seq_len=capacity,
        local_batch_size=b,
    )


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))

    def forward(self, inputs, **kwargs):
        return inputs * self.weight


def dynamic_trainer(kind="increasing", log_freq=1):
    fixture = tiny_trainer(False, k=2, b=2, seq=8, model=ScalarModel(), log_freq=log_freq)
    trainer = object.__new__(DynamicSeqLenTrainer)
    trainer.__dict__.update(fixture.__dict__)
    trainer.config.training.steps = 6
    trainer.config.seq_len_scheduler = SeqLenSchedulerConfig(
        kind=kind,
        length=4,
        lengths=[2, 4],
        boundaries=[] if kind == "triangular" else [2],
        period_steps=4,
    )
    original_loss = trainer.loss_fn
    trainer.loss_fn = lambda pred, labels, total, **kwargs: original_loss(pred, labels, total)
    trainer.seq_len_scheduler = trainer.config.seq_len_scheduler.build(6)
    trainer.n_valid_tokens_seen = 0
    trainer.flops_by_length = {2: 12, 4: 14}
    metrics = object.__new__(DynamicSeqMetricsProcessor)
    metrics.__dict__.update(cpu_metrics(trainer.parallel_dims, log_freq).__dict__)
    metrics.reset_window()
    trainer.metrics_processor = metrics
    trainer.dataloader = loader(infinite=True)
    return trainer


class SchedulesTest(unittest.TestCase):
    def test_public_profiles_preserve_training_budgets(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "configs/dynamic_seq").glob("*.yaml")):
            if path.stem == "smoke":
                continue
            raw = load_config(path)
            _, points = grid(raw)
            for index in range(len(points)):
                with self.subTest(profile=path.name, index=index):
                    _, _, _, args = resolve_point(path, index)
                    with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
                        cfg = ConfigManager().parse_args(
                            [
                                "--module",
                                "torchtitan_dynamic_seq_len.config_registry",
                                "--config",
                                raw["torchtitan_config"],
                                *args,
                            ]
                        )
                    self.assertIs(cfg._owner, DynamicSeqLenTrainer)
                    scheduler = cfg.validate_dynamic()
                    self.assertEqual(cfg.training.steps, 3840)
                    self.assertEqual(cfg.training.global_batch_size, 512)
                    self.assertEqual(cfg.training.local_batch_size, 16)
                    self.assertEqual(cfg.optimizer.learning_rate, 0.0005)
                    self.assertEqual(cfg.lr_scheduler.warmup_steps, 0)
                    self.assertEqual(cfg.lr_scheduler.min_lr_factor, 1.0)
                    self.assertEqual(cfg.validator.eval_seq_len, 1024)
                    self.assertEqual(
                        sum(scheduler.length_for_step(s) for s in range(1, 3841)) * 512,
                        2013265920,
                    )
                    self.assertEqual(raw["resources"]["account"], "YOUR_ACCOUNT")
                    self.assertFalse(raw["wandb"]["enabled"])
                    self.assertNotIn("/home/kangan", path.read_text())
                    self.assertNotIn("local/", path.read_text())

    def test_public_submission_sweeps_dry_run(self):
        root = Path(__file__).resolve().parents[1]
        for profile, count in (("dynamic_seq/cyclic_sweep.yaml", 2), ("lr_sweep_constant.yaml", 5)):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as folder:
                raw = yaml.safe_load((root / "configs" / profile).read_text())
                raw["resources"]["account"] = "test-account"
                path = Path(folder) / "experiment.yaml"
                path.write_text(yaml.safe_dump(raw))
                env = {**os.environ, "DYNAMIC_SEQ_LEN_OUTPUT": str(Path(folder) / "output")}
                for key in list(env):
                    if key.startswith("SBATCH_") or key == "WANDB_RUN_ID":
                        env.pop(key)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(root / "cluster/vt_arc/submit.py"),
                        str(path),
                        "--dry-run",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn(f"Would submit {count} run(s)", result.stdout)
                self.assertIn(f"--array=0-{count - 1}%", result.stdout)
                self.assertFalse((Path(folder) / "output").exists())

    def test_dynamic_launch_from_checkout_without_local_directory(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder)
            for name in (
                "torchtitan_dynamic_seq_len",
                "scripts",
                "configs",
                "cluster",
                "tests/fixtures",
            ):
                shutil.copytree(
                    root / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__")
                )
            code = """
import os, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
os.environ['WORLD_SIZE'] = '2'
from torchtitan.config import ConfigManager
from torchtitan_dynamic_seq_len.run_config import resolve_point
from torchtitan_dynamic_seq_len.dynamic_seq_trainer import DynamicSeqLenTrainer
raw, _, _, args = resolve_point(root / 'configs/dynamic_seq/cyclic_period64.yaml', 0)
cfg = ConfigManager().parse_args(['--module', 'torchtitan_dynamic_seq_len.config_registry',
                                '--config', raw['torchtitan_config'], *args])
assert cfg._owner is DynamicSeqLenTrainer
assert cfg.validate_dynamic().length_for_step(2) == 544
for name, module in tuple(sys.modules.items()):
    if name.startswith('torchtitan_dynamic_seq_len') and getattr(module, '__file__', None):
        assert Path(module.__file__).resolve().is_relative_to(root), name
"""
            subprocess.run(
                [sys.executable, "-I", "-c", code, str(checkout)],
                cwd=checkout,
                capture_output=True,
                text=True,
                check=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(checkout / "scripts/train.py"),
                    str(checkout / "configs/dynamic_seq/cyclic_period64.yaml"),
                    "--nproc",
                    "2",
                    "--dry-run",
                ],
                cwd=checkout,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("--seq_len_scheduler.period_steps 64", result.stdout)
            self.assertFalse((checkout / "local").exists())

    def test_cyclic_linear_steps_and_exact_budget(self):
        scheduler = SeqLenSchedulerConfig(
            kind="cyclic", min_length=512, max_length=1536, period_steps=256
        ).build(3840)
        values = [scheduler.length_for_step(s) for s in range(1, 3841)]
        self.assertEqual(values[:3], [512, 520, 528])
        self.assertEqual(values[127:130], [1528, 1536, 1528])
        self.assertEqual(values[255:258], [520, 512, 520])
        self.assertTrue(all(abs(b - a) == 8 for a, b in zip(values, values[1:])))
        self.assertTrue(all(v % 8 == 0 for v in values))
        self.assertEqual(values, values[:256] * 15)
        self.assertEqual(set(values), set(scheduler.lengths))
        self.assertEqual(sum(values) * 512, 2013265920)
        with self.assertRaises(ValueError):
            scheduler.length_for_step(0)

    def test_shapes_and_exact_time_reversal(self):
        config = SeqLenSchedulerConfig(lengths=[2, 4, 8], boundaries=[2, 5])
        expected = [2, 2, 4, 4, 4, 8, 8]
        for kind, values in (("increasing", expected), ("decreasing", expected[::-1])):
            config.kind = kind
            scheduler = config.build(7)
            self.assertEqual([scheduler.length_for_step(s) for s in range(1, 8)], values)
            with self.assertRaises(ValueError):
                scheduler.length_for_step(0)
        config.kind, config.boundaries, config.period_steps = "triangular", [], 4
        scheduler = config.build(7)
        self.assertEqual(
            [scheduler.length_for_step(s) for s in range(1, 10)], [2, 4, 8, 4, 2, 4, 8, 4, 2]
        )
        config.kind, config.length = "constant", 3
        self.assertEqual(config.build(7).length_for_step(100), 3)

    def test_bad_parameters(self):
        for kwargs in (
            dict(kind="constant", length=0),
            dict(kind="increasing", lengths=[4, 2]),
            dict(kind="increasing", boundaries=[0, 1, 2]),
            dict(kind="triangular", period_steps=2),
            dict(kind="cyclic", min_length=0),
            dict(kind="cyclic", min_length=1536, max_length=512),
            dict(kind="cyclic", period_steps=0),
            dict(kind="cyclic", period_steps=255),
            dict(kind="cyclic", period_steps=768),
            dict(kind="cyclic", period_steps=256, boundaries=[2]),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SeqLenSchedulerConfig(**kwargs).build(8)

    def test_yaml_cli_dispatch_and_capacity(self):
        root = Path(__file__).resolve().parents[1]
        for filename, points in (
            ("dynamic_seq/cyclic_sweep.yaml", 2),
            ("dynamic_seq/smoke.yaml", 1),
            ("dynamic_seq/cyclic.yaml", 1),
        ):
            for index in range(points):
                raw, _, _, args = resolve_point(root / "configs" / filename, index)
                with patch.dict(
                    os.environ,
                    {
                        "WORLD_SIZE": "1",
                        "DYNAMIC_SEQ_LEN_VALIDATION_DIR": raw["paths"].get("validation_dir", ""),
                    },
                ):
                    config = ConfigManager().parse_args(
                        [
                            "--module",
                            "torchtitan_dynamic_seq_len.config_registry",
                            "--config",
                            raw["torchtitan_config"],
                            *args,
                        ]
                    )
                self.assertIs(config._owner, DynamicSeqLenTrainer)
                config.validate_dynamic()
                self.assertIs(config.dataloader._owner, DynamicSeqDataLoader)
                self.assertIs(config.metrics._owner, DynamicSeqMetricsProcessor)
                config.dataloader.num_workers = 2
                with self.assertRaisesRegex(ValueError, "num_workers=0"):
                    config.validate_dynamic()
                config.dataloader.num_workers = 0
                config.seq_len_scheduler.kind = "constant"
                config.seq_len_scheduler.length = 3
                with self.assertRaisesRegex(ValueError, "loss.num_chunks"):
                    config.validate_dynamic()


class PackingTest(unittest.TestCase):
    def test_cyclic_full_period_packing_and_resume(self):
        schedule = SeqLenSchedulerConfig(kind="cyclic", period_steps=256).build(3840)
        documents = [" ".join(map(str, range(263000)))]
        dl = loader(streaming=True, b=1, documents=documents, capacity=1536)
        iterator = iter(dl)
        offset = 0
        for step in range(1, 258):
            if step == 130:  # Resume after the first step with the maximum length.
                state = copy.deepcopy(dl.state_dict())
                dl = loader(streaming=True, b=1, documents=documents, capacity=1536)
                dl.load_state_dict(state)
                iterator = iter(dl)
            length = schedule.length_for_step(step)
            dl.set_seq_len(length)
            inputs, labels = next(iterator)
            self.assertEqual(tuple(labels.shape), (1, length))
            torch.testing.assert_close(inputs["input"][0], torch.arange(offset, offset + length))
            torch.testing.assert_close(labels[0], torch.arange(offset + 1, offset + length + 1))
            offset += length
        self.assertEqual(offset, 256 * 1024 + 512)

    def test_infinite_resume_after_exact_document_end(self):
        for streaming in (False, True):
            dl = loader(streaming=streaming, infinite=True, b=1, documents=["0 1 2 3 4"])
            dl.set_seq_len(4)
            next(iter(dl))
            state = copy.deepcopy(dl.state_dict())
            restored = loader(streaming=streaming, infinite=True, b=1, documents=["0 1 2 3 4"])
            restored.load_state_dict(state)
            restored.set_seq_len(2)
            self.assertEqual(next(iter(restored))[1].tolist(), [[1, 2]])

    def test_live_lengths_and_resume_at_end_of_document_stream(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                dl = loader(streaming=streaming)
                dl.set_seq_len(2)
                iterator = iter(dl)
                first = next(iterator)
                state = copy.deepcopy(dl.state_dict())
                restored = loader(streaming=streaming)
                restored.load_state_dict(state)
                resumed = iter(restored)
                consumed = first[0]["input"].flatten().tolist()
                for length in [8, 4, 2, 8, 4]:
                    dl.set_seq_len(length)
                    restored.set_seq_len(length)
                    actual, expected = next(resumed), next(iterator)
                    for key in actual[0]:
                        torch.testing.assert_close(actual[0][key], expected[0][key])
                    torch.testing.assert_close(actual[1], expected[1])
                    self.assertTrue(bool((actual[0]["positions"] < length).all()))
                    torch.testing.assert_close(actual[1], actual[0]["input"] + 1)
                    consumed.extend(actual[0]["input"].flatten().tolist())
                self.assertEqual(consumed, list(range(len(consumed))))

    def test_document_labels_and_tail_across_lengths(self):
        dl = loader(b=1, documents=["0 1 2 3", "10 11 12 13 14"])
        dl.set_seq_len(4)
        it = iter(dl)
        inputs, labels = next(it)
        self.assertEqual(inputs["input"].tolist(), [[0, 1, 2, 10]])
        self.assertEqual(labels.tolist(), [[1, 2, 3, 11]])
        self.assertEqual(inputs["positions"].tolist(), [[0, 1, 2, 0]])
        dl.set_seq_len(2)
        self.assertEqual(next(it)[1].tolist(), [[12, 13]])
        with self.assertRaises(StopIteration):
            next(it)
        self.assertEqual(dl.dataset._labels_buffer, [14])


class TrainerTest(unittest.TestCase):
    def test_rope_capacity_matches_short_cache_and_is_reused(self):
        from torchtitan.models.common import ComplexRoPE

        rope = ComplexRoPE.Config(dim=8, max_seq_len=16, theta=500000).build()
        cache = rope.cache
        for length in [2, 8, 4, 16, 2]:
            short = ComplexRoPE.Config(dim=8, max_seq_len=length, theta=500000).build()
            query = torch.randn(2, length, 2, 8)
            positions = torch.arange(length).repeat(2, 1)
            actual = rope(query, query, positions)
            expected = short(query, query, positions)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            self.assertIs(cache, rope.cache)

    def test_fixed_validation_length_is_independent_of_capacity(self):
        from torchtitan.config import ParallelismConfig

        from tests.test_fixed_validation import _dims, _loss, _write_artifact
        from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(root, n=24, length=4)
            validator = FixedBudgetValidator.Config(
                token_budget=96,
                artifact_path=str(root),
                eval_seq_len=4,
            ).build(
                parallelism=ParallelismConfig(),
                dp_world_size=1,
                dp_rank=0,
                tokenizer=SimpleNamespace(tokenizer_path=str(root / "tokenizer")),
                parallel_dims=_dims(),
                loss_fn=_loss,
                validation_context=nullcontext,
                metrics_processor=None,
                seq_len=8,
                local_batch_size=2,
            )
            self.assertEqual(validator.seq_len, 4)
            self.assertEqual(validator.artifact.num_sequences, 24)

    def test_constant_is_native_update(self):
        dynamic = dynamic_trainer("constant")
        native = tiny_trainer(False, k=2, b=2, seq=4, model=ScalarModel())
        native.__class__ = Trainer
        original_loss = native.loss_fn
        native.loss_fn = lambda pred, labels, total, **kwargs: original_loss(pred, labels, total)
        native.dataloader = loader(infinite=True)
        native.dataloader.set_seq_len(4)
        di, ni = (
            dynamic.batch_generator(dynamic.dataloader),
            native.batch_generator(native.dataloader),
        )
        for step in range(1, 4):
            dynamic.step = native.step = step
            dynamic.train_step(di)
            native.train_step(ni)
            torch.testing.assert_close(
                dynamic.model_parts[0].weight, native.model_parts[0].weight, rtol=0, atol=0
            )
            self.assertEqual(dynamic.ntokens_seen, native.ntokens_seen)
        payload = dynamic.metrics_processor.logger.log.call_args.args[0]
        self.assertEqual(payload["n_valid_tokens_seen"], 48)
        self.assertEqual(payload["valid_tokens_per_step"], 16)
        self.assertEqual(list(payload).index("seq_len"), list(payload).index("lr") + 1)

    def test_resume_across_transition_and_nonlogging_steps(self):
        original = dynamic_trainer(log_freq=3)
        iterator = original.batch_generator(original.dataloader)
        for step in (1, 2):
            original.step = step
            original.train_step(iterator)
        resumed = dynamic_trainer(log_freq=3)
        resumed.model_parts[0].load_state_dict(copy.deepcopy(original.model_parts[0].state_dict()))
        resumed.optimizers.optimizer.load_state_dict(
            copy.deepcopy(original.optimizers.optimizer.state_dict())
        )
        resumed.lr_schedulers.scheduler.load_state_dict(
            original.lr_schedulers.scheduler.state_dict()
        )
        resumed.dataloader.load_state_dict(copy.deepcopy(original.dataloader.state_dict()))
        resumed.load_state_dict(original.state_dict())
        ri = resumed.batch_generator(resumed.dataloader)
        for step in (3, 4, 5, 6):
            original.step = resumed.step = step
            original.train_step(iterator)
            resumed.train_step(ri)
            torch.testing.assert_close(
                original.model_parts[0].weight, resumed.model_parts[0].weight, rtol=0, atol=0
            )
            self.assertEqual(original.state_dict(), resumed.state_dict())
        self.assertEqual(original.n_valid_tokens_seen, 80)
        resumed.config.seq_len_scheduler.lengths = [2, 8]
        with self.assertRaisesRegex(ValueError, "differs"):
            resumed.load_state_dict(original.state_dict())

    def test_masked_labels_and_cancelled_accumulation(self):
        trainer = dynamic_trainer("constant")
        data = [({"input": torch.ones(2, 4)}, torch.ones(2, 4)) for _ in range(2)]
        data[0][1][0, 0] = -100
        trainer.train_step(trainer.batch_generator(data))
        payload = trainer.metrics_processor.logger.log.call_args.args[0]
        self.assertEqual(payload["n_tokens_seen"], 16)
        self.assertEqual(payload["n_valid_tokens_seen"], 15)
        before = trainer.state_dict()
        with self.assertRaises(Exception):
            trainer.train_step(trainer.batch_generator(data[:1]))
        self.assertEqual(before, trainer.state_dict())

    def test_partial_final_batch_cancels_whole_step(self):
        from torchtitan.components.dataloader import DataloaderExhaustedError

        trainer = dynamic_trainer("constant")
        trainer.dataloader = loader(b=2, documents=[" ".join(map(str, range(13)))])
        before = trainer.state_dict()
        with self.assertRaises(DataloaderExhaustedError):
            trainer.train_step(trainer.batch_generator(trainer.dataloader))
        self.assertEqual(before, trainer.state_dict())
        self.assertEqual(trainer.optimizers.gradients, [])

    def test_weighted_flops_and_validation_window(self):
        trainer = dynamic_trainer(log_freq=3)
        it = trainer.batch_generator(trainer.dataloader)
        trainer.train_step(it)  # step 1 logs immediately
        trainer.step = 2
        trainer.train_step(it)  # 8 positions at length 2
        metrics = trainer.metrics_processor
        with metrics.validation_scope():
            metrics.ntokens_since_last_log += 999
            metrics.log_validation(0.5, 2)
        trainer.step = 3
        trainer.train_step(it)  # 16 positions at length 4
        payload = metrics.logger.log.call_args.args[0]
        expected_flops = 8 * 12 + 16 * 14
        self.assertAlmostEqual(
            payload["tflops"] / payload["throughput(tps)"] * 1e12, expected_flops / 24
        )
        self.assertEqual(payload["n_tokens_seen"], 32)
        self.assertEqual(payload["n_valid_tokens_seen"], 32)


if __name__ == "__main__":
    unittest.main()
