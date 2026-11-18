"""Main entry for the native-vocabulary, randomly initialized Llama3 1B experiment."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch.distributed as dist
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.models.llama3 import model_registry

from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator
from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentBatchDataLoader
from torchtitan_dynamic_seq_len.dynamic_seq_bs.trainer import DynamicSeqBSTrainer
from torchtitan_dynamic_seq_len.optimizer import ExperimentOptimizers
from torchtitan_dynamic_seq_len.token_metrics import TokenMetricsProcessor
from torchtitan_dynamic_seq_len.token_progress import TokenCheckpointManager, TokenPlan

DATASET_SHA256 = "4bf6946f7b7f7cbad0799746519baaa9ef2518f974d2239a8df2f184db004806"
VALIDATION_SHA256 = "eb2faf62a36e324d07e08da73170e1268182cdfea1f38352362b86232f47155b"


class ExperimentValidator(FixedBudgetValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(FixedBudgetValidator.Config):
        token_intervals: int = 0
        """Positive values select equally spaced token targets, including a baseline at zero."""

    def should_validate(self, step):
        # The trainer evaluates token targets before the checkpoint at the same step.
        return not self.config.token_intervals and super().should_validate(step)

    def validate_milestone(self, model_parts, milestone, dump_folder):
        path = Path(dump_folder) / "validation" / f"point-{milestone.index:03d}.json"
        # A restart can replay updates beyond the restored checkpoint. All ranks
        # must make the same decision before entering the validator collectives.
        completed = [path.is_file() if self.dp_rank == 0 else False]
        if self.dp_world_size > 1:
            dist.broadcast_object_list(completed, src=0, group=self._dp_group)
        if completed[0]:
            return
        self.validate(
            model_parts,
            milestone.step,
            extra_metrics={
                "validation_metrics/index": milestone.index,
                "validation_metrics/target_tokens": milestone.threshold,
                "validation_metrics/completed_updates": milestone.step,
            },
        )
        if self.dp_rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt = {**asdict(milestone), **self.last_result}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(path)


class SequenceExperimentTrainer(DynamicSeqBSTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(DynamicSeqBSTrainer.Config):
        token_budget: int = 0
        smoke_test: bool = False

        def validate_dynamic(self):
            if (not self.validator.enable or self.validator.eval_seq_len != 16384
                    or self.validator.token_budget != 15335424):
                raise ValueError("Keep all 936 fixed validation windows at length 16384.")
            if (self.dataloader.packing != "stream" or self.dataloader.infinite
                    or self.dataloader.expected_manifest_sha256 != DATASET_SHA256):
                raise ValueError("Keep the fixed dataset identity, finite ordered stream, and packing protocol.")
            if not self.metrics.enable_wandb or not isinstance(self.metrics, TokenMetricsProcessor.Config):
                raise ValueError("The main experiment requires W&B and TokenMetricsProcessor.")
            if not isinstance(self.checkpoint, TokenCheckpointManager.Config):
                raise ValueError("The main experiment requires ten token checkpoint milestones.")
            self.checkpoint.validate_policy()
            lengths = super(SequenceExperimentTrainer.Config, self).validate_dynamic()
            self.build_token_plan(lengths)
            return lengths

        def build_token_plan(self, lengths=None):
            lengths = lengths or self.seq_len_scheduler.build(self.training.steps)
            batches = self.batch_size_scheduler.build(self.training.steps)
            if self.smoke_test:
                if self.validator.token_intervals:
                    raise ValueError("Smoke uses step-based validation. Set validator.token_intervals to zero.")
                actual_lengths = [lengths.length_for_step(s) for s in range(1, self.training.steps + 1)]
                if actual_lengths != [2048, 2048, 4096, 4096, 8192, 8192, 16384, 16384]:
                    raise ValueError("Smoke lengths must be 2048, 4096, 8192, 16384, each for two steps.")
                if any(batches.value_for_step(s) != 512 for s in range(1, 9)):
                    raise ValueError("Smoke global batch size must remain 512.")
                return TokenPlan.smoke((512 * length for length in actual_lengths), self.token_budget)
            return TokenPlan.build(
                (lengths.length_for_step(step) * batches.value_for_step(step)
                 for step in range(1, self.training.steps + 1)),
                self.token_budget,
                validation_intervals=self.validator.token_intervals,
            )

    def __init__(self, config):
        config.validate_dynamic()
        self.token_plan = config.build_token_plan()
        self.validation_by_step = {point.step: point for point in self.token_plan.validations}
        self.smoke_stop_after = int(os.environ.get("DYNAMIC_SEQ_LEN_SMOKE_STOP_AFTER", "0"))
        if self.smoke_stop_after and (not config.smoke_test or self.smoke_stop_after not in (2, 4, 6)):
            raise ValueError("Controlled smoke interruption requires a checkpoint at step 2, 4, or 6.")
        super().__init__(config)
        if self.validator.artifact.manifest_sha256 != VALIDATION_SHA256:
            raise ValueError("Validation windows differ from the fixed 936-window artifact.")
        self.metrics_processor.set_training_tokens(self.nominal_tokens_seen)

    @property
    def nominal_tokens_seen(self):
        # Full microbatches give equal nominal counts on every replicated DP rank.
        # The inherited schedule signature rejects a changed DP degree on recovery.
        return self.ntokens_seen * self.parallel_dims.dp_replicate

    def train_step(self, data_iterator):
        if self.step == 1 and 0 in self.validation_by_step:
            # TorchTitan loads the checkpoint before entering the first update.
            self.validator.validate_milestone(
                self.model_parts, self.validation_by_step[0], self.config.dump_folder
            )
        try:
            result = super().train_step(data_iterator)
            if self.nominal_tokens_seen != self.token_plan.cumulative[self.step]:
                raise RuntimeError("Training token count differs from the scheduled prefix.")
        finally:
            self.metrics_processor.set_training_tokens(self.nominal_tokens_seen)
        if self.step in self.validation_by_step:
            self.validator.validate_milestone(
                self.model_parts, self.validation_by_step[self.step], self.config.dump_folder
            )
        return result

    def state_dict(self):
        return {**super().state_dict(), "token_budget": self.token_plan.budget,
                "checkpoint_steps": [item.step for item in self.token_plan.milestones],
                "nominal_tokens_seen": self.nominal_tokens_seen}

    def load_state_dict(self, state_dict):
        step = state_dict["step"]
        if (state_dict.get("token_budget") != self.token_plan.budget
                or state_dict.get("checkpoint_steps") != [item.step for item in self.token_plan.milestones]
                or not 0 <= step < len(self.token_plan.cumulative)
                or state_dict.get("nominal_tokens_seen") != self.token_plan.cumulative[step]
                or state_dict["ntokens_seen"] * self.parallel_dims.dp_replicate
                != state_dict.get("nominal_tokens_seen")):
            raise ValueError("Checkpoint token budget or cumulative count differs from this run.")
        super().load_state_dict(state_dict)
        self.metrics_processor.set_training_tokens(self.nominal_tokens_seen)

    def train(self):
        super().train()
        stopped = getattr(self, "smoke_stop_after", 0)
        if self.nominal_tokens_seen < self.token_plan.budget and not (stopped and self.step == stopped):
            raise RuntimeError("Training stopped before the token budget. Check data exhaustion.")

    def should_continue_training(self):
        stop = getattr(self, "smoke_stop_after", 0)
        return super().should_continue_training() and (not stop or self.step < stop)


def llama3_1b() -> SequenceExperimentTrainer.Config:
    """Keep upstream 1B definitions and the data package's fixed training protocol."""
    assets = Path(os.environ.get("DYNAMIC_SEQ_LEN_ASSETS", Path(__file__).resolve().parents[1] / "assets"))
    dataset = assets / "data/fineweb_edu_long_30b_shuffled_v1"
    config = SequenceExperimentTrainer.Config(
        model_spec=model_registry("1B", attn_backend="varlen"),
        hf_assets_path=str(assets / "hf/llama3-tokenizer"),
        loss=ChunkedLossWrapper.Config(num_chunks=32,
                loss_fn=CrossEntropyLoss.Config(global_vocab_size=128256)),
        optimizer=ExperimentOptimizers.Config(method="adamw"),
        dataloader=LongDocumentBatchDataLoader.Config(
            dataset_path=str(dataset), expected_manifest_sha256=DATASET_SHA256,
            packing="stream", infinite=False, pin_memory=True),
        training=TrainingConfig(steps=0, local_batch_size=1, global_batch_size=-1,
                                seq_len=16384, max_norm=1.0),
        lr_scheduler=LRSchedulersContainer.Config(),
        metrics=TokenMetricsProcessor.Config(log_freq=1),
        validator=ExperimentValidator.Config(
            enable=True, artifact_path=str(dataset / "validation"),
            token_budget=15335424, eval_seq_len=16384, eval_local_batch_size=2),
        checkpoint=TokenCheckpointManager.Config(),
        activation_checkpoint=SelectiveAC.Config(),
        debug=DebugConfig(seed=42, deterministic=False),
    )
    config.parallelism.data_parallel_replicate_degree = int(os.environ.get("WORLD_SIZE", "1"))
    config.parallelism.data_parallel_shard_degree = 1
    return config
