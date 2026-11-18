"""Do CPU checks of schedules, token packing, checkpoint recovery, and training."""

import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS

from tests.trainer_fixture import cpu_metrics, tiny_trainer
from torchtitan_dynamic_seq_len.data.dynamic_seq_loader import DynamicSeqDataLoader
from torchtitan_dynamic_seq_len.dynamic_seq.metrics import DynamicSeqMetricsProcessor
from torchtitan_dynamic_seq_len.dynamic_seq.trainer import DynamicSeqLenTrainer
from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig


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
    fixture = tiny_trainer(k=2, b=2, seq=8, model=ScalarModel(), log_freq=log_freq)
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

    def test_rounded_cyclic_symmetry_and_exact_budget(self):
        schedule = SeqLenSchedulerConfig(
            kind="cyclic",
            min_length=8,
            max_length=2040,
            period_steps=256,
            round_to_multiple=8,
        ).build(3840)
        values = [schedule.length_for_step(s) for s in range(1, 3841)]
        self.assertEqual(values, values[:256] * 15)
        self.assertEqual((min(values), max(values)), (8, 2040))
        self.assertEqual(values[64], 1024)
        self.assertEqual(sum(values) * 512, 2013265920)
        self.assertEqual(set(values), set(schedule.lengths))
        for phase, actual in enumerate(values[:256]):
            rise = min(phase, 256 - phase)
            ideal = 8 + 2032 * rise / 128
            self.assertLessEqual(abs(actual - ideal), 4)
            self.assertEqual(actual % 8, 0)
            self.assertEqual(actual + values[(phase + 128) % 256], 2048)
            self.assertEqual(actual, values[(-phase) % 256])
        self.assertTrue(all(a < b for a, b in zip(values[:128], values[1:129])))
        self.assertEqual(set(abs(a - b) for a, b in zip(values, values[1:])), {8, 16})

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
            dict(kind="cyclic", round_to_multiple=0),
            dict(kind="cyclic", round_to_multiple=8, period_steps=254),
            dict(kind="cyclic", round_to_multiple=8, min_length=7),
            dict(kind="cyclic", round_to_multiple=8, min_length=8, max_length=2048),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SeqLenSchedulerConfig(**kwargs).build(8)

class PackingTest(unittest.TestCase):
    def test_cyclic_full_period_packing_and_resume(self):
        for minimum, maximum, multiple in ((512, 1536, None), (8, 2040, 8), (512, 8192, None)):
            with self.subTest(minimum=minimum, maximum=maximum):
                schedule = SeqLenSchedulerConfig(
                    kind="cyclic",
                    min_length=minimum,
                    max_length=maximum,
                    period_steps=256,
                    round_to_multiple=multiple,
                ).build(3840)
                documents = [
                    " ".join(
                        map(str, range(sum(schedule.length_for_step(s) for s in range(1, 258)) + 1))
                    )
                ]
                dl = loader(
                    streaming=True, b=1, documents=documents, capacity=max(schedule.lengths)
                )
                iterator = iter(dl)
                offset = 0
                for step in range(1, 258):
                    if step == 130:  # Resume after the first step with the maximum length.
                        state = copy.deepcopy(dl.state_dict())
                        dl = loader(
                            streaming=True, b=1, documents=documents, capacity=max(schedule.lengths)
                        )
                        dl.load_state_dict(state)
                        iterator = iter(dl)
                    length = schedule.length_for_step(step)
                    dl.set_seq_len(length)
                    inputs, labels = next(iterator)
                    self.assertEqual(tuple(labels.shape), (1, length))
                    torch.testing.assert_close(
                        inputs["input"][0], torch.arange(offset, offset + length)
                    )
                    torch.testing.assert_close(
                        labels[0], torch.arange(offset + 1, offset + length + 1)
                    )
                    offset += length
                self.assertEqual(offset, sum(schedule.length_for_step(s) for s in range(1, 258)))

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
        native = tiny_trainer(k=2, b=2, seq=4, model=ScalarModel())
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
