"""Evaluate a fixed set of tokens with checked file hashes.

A zero token budget selects the TorchTitan validator.
With a positive budget, each rank evaluates its assigned sequences once.
The validator does not use the training batch size or the steps field.
Data preparation fixes the document boundaries, positions, and token values."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.validate import Validator
from torchtitan.tools.logging import logger

ARRAY_NAMES = ("input", "labels", "positions")
DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "assets/data/fineweb100_validation"


def resolve_artifact_path(artifact_path: str = "") -> Path:
    return (
        Path(
            artifact_path
            or os.environ.get("DYNAMIC_SEQ_LEN_VALIDATION_DIR")
            or DEFAULT_ARTIFACT_PATH
        )
        .expanduser()
        .resolve()
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained_file(root: Path, filename: str) -> Path:
    """Require portable, relative artifact and tokenizer file names."""
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Manifest path must be relative: {filename!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes its root: {filename!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Required validation file is missing: {path}")
    return path


def _checked_file(root: Path, entry: dict[str, str]) -> Path:
    path = _contained_file(root, entry["file"])
    expected = entry["sha256"]
    if file_sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch for validation file: {path}")
    return path


def sequence_batches(
    num_sequences: int, dp_world_size: int, dp_rank: int, local_batch_size: int
) -> Iterator[slice]:
    """Divide the sequence IDs across DP ranks without duplicates or padding.

    Each rank receives the same number of sequences.
    Thus, each rank uses the same FSDP forward shapes.
    The sequence count must be divisible by the DP size."""
    if dp_world_size < 1 or not 0 <= dp_rank < dp_world_size:
        raise ValueError("Invalid validation DP size or rank")
    if local_batch_size < 1 or num_sequences < 1:
        raise ValueError("Validation sequence count and local batch must be positive")
    if num_sequences % dp_world_size:
        raise ValueError("Fixed validation sequence count must be divisible by DP size")
    per_rank = num_sequences // dp_world_size
    start, stop = dp_rank * per_rank, (dp_rank + 1) * per_rank
    for first in range(start, stop, local_batch_size):
        yield slice(first, min(first + local_batch_size, stop))


class FrozenValidationArtifact:
    """Validate identity, shape, and exact effective budget before any forward."""

    def __init__(
        self,
        path: str | Path,
        *,
        token_budget: int,
        sequence_length: int,
        tokenizer_path: str | Path,
    ):
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Frozen validation artifact missing: {manifest_path}. "
                "Build or copy the artifact and set DYNAMIC_SEQ_LEN_VALIDATION_DIR "
                "(or validator.artifact_path); validation never falls back to streaming."
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.manifest_sha256 = file_sha256(manifest_path)
        manifest = self.manifest
        source = manifest.get("source", {})
        if (
            source.get("repo_id") == "HuggingFaceFW/fineweb"
            and source.get("config") == "sample-100BT"
        ):
            from torchtitan_dynamic_seq_len.data.fineweb100_identity import (
                verify_validation_identity,
            )

            verify_validation_identity(manifest)
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported frozen validation manifest schema")
        if manifest.get("token_budget") != token_budget or token_budget <= 0:
            raise ValueError("Frozen validation token budget does not match config")
        if manifest.get("sequence_length") != sequence_length:
            raise ValueError("Frozen validation sequence length does not match training")
        self.num_sequences = manifest.get("num_sequences")
        if (
            not isinstance(self.num_sequences, int)
            or self.num_sequences <= 0
            or self.num_sequences * sequence_length != token_budget
        ):
            raise ValueError("Frozen validation sequence count cannot satisfy token budget")

        tokenizer_files = manifest.get("tokenizer_files", {})
        required = {"tokenizer.json", "tokenizer_config.json"}
        if not required.issubset(tokenizer_files):
            raise ValueError("Manifest must fingerprint tokenizer.json and tokenizer_config.json")
        for filename, digest in tokenizer_files.items():
            path = _contained_file(Path(tokenizer_path), filename)
            if file_sha256(path) != digest:
                raise ValueError(f"Validation tokenizer fingerprint mismatch: {filename}")

        # Bind any document provenance list to the same manifest as the tensors.
        if "evaluation_documents" in manifest:
            _checked_file(self.path, manifest["evaluation_documents"])

        self.arrays: dict[str, np.ndarray] = {}
        entries = manifest.get("arrays", {})
        for name in ARRAY_NAMES:
            if name not in entries:
                raise ValueError(f"Frozen validation manifest missing array: {name}")
            path = _checked_file(self.path, entries[name])
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.dtype("int64"):
                raise ValueError(f"Validation {name} must have int64 dtype")
            if array.shape != (self.num_sequences, sequence_length):
                raise ValueError(f"Validation {name} shape does not satisfy fixed budget")
            if np.any(array < 0):
                raise ValueError(
                    f"Validation {name} contains negative/ignored tokens or positions; "
                    "the frozen budget requires every label position to be valid"
                )
            self.arrays[name] = array
        self.token_budget = token_budget

    def batches(
        self, *, dp_world_size: int, dp_rank: int, local_batch_size: int
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        for indices in sequence_batches(
            self.num_sequences, dp_world_size, dp_rank, local_batch_size
        ):
            # Copy just one forward's slices: read-only mmaps must not be exposed
            # as writable tensors, and the full artifact need not fit in RAM.
            tensors = {
                name: torch.from_numpy(np.array(array[indices], copy=True))
                for name, array in self.arrays.items()
            }
            yield {"input": tensors["input"], "positions": tensors["positions"]}, tensors["labels"]


class FixedBudgetValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        eval_seq_len: int = 0
        """This is the evaluation length. Zero selects the training capacity for compatibility."""

        token_budget: int = 0
        """Zero preserves legacy validation; positive ignores ``steps`` entirely."""

        eval_local_batch_size: int = 32
        """Forward batch size only, independent of both training batch sizes."""

        artifact_path: str = ""
        """Frozen artifact directory; empty uses DYNAMIC_SEQ_LEN_VALIDATION_DIR/default."""

        def __post_init__(self):
            if self.token_budget == 0:
                Validator.Config.__post_init__(self)
            if self.eval_seq_len < 0:
                raise ValueError("eval_seq_len must be nonnegative")
            if self.token_budget < 0:
                raise ValueError("Validation token_budget must be nonnegative")
            if self.eval_local_batch_size <= 0:
                raise ValueError("Validation eval_local_batch_size must be positive")

    def __init__(self, config: Config, **kwargs):
        if config.eval_seq_len:
            if config.eval_seq_len > kwargs["seq_len"]:
                raise ValueError("eval_seq_len exceeds model/RoPE capacity")
            kwargs["seq_len"] = config.eval_seq_len
        # The legacy constructor interprets steps=-1 as an unbounded stream and
        # warns about uneven ranks. Neither applies to this finite artifact.
        super().__init__(replace(config, steps=1) if config.token_budget else config, **kwargs)
        self.config = config
        self.artifact: FrozenValidationArtifact | None = None
        self.last_result: dict[str, Any] | None = None
        if config.token_budget == 0:
            return
        if any(getattr(self.parallel_dims, f"{axis}_enabled") for axis in ("tp", "pp", "cp")):
            raise ValueError("Fixed-budget validation currently supports pure DP only (TP/PP/CP=1)")
        if self.parallelism.spmd_backend != "default":
            raise ValueError("Fixed-budget validation currently requires spmd_backend='default'")
        if self.dp_world_size != self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard:
            raise ValueError("Validator DP size does not match actual DP mesh")
        self._dp_group = None
        if self.dp_world_size > 1:
            mesh = self.parallel_dims.get_mesh("batch")
            if mesh.size() != self.dp_world_size or mesh.get_local_rank() != self.dp_rank:
                raise ValueError("Validator DP rank/size does not match batch mesh")
            # TorchTitan's loss mesh is DP x CP, hence exactly DP here. Use
            # its real reduction group; some revisions use a bookkeeping-only
            # backend for the batch/dataloader mesh.
            reduction_mesh = self.parallel_dims.get_mesh("loss")
            if (
                reduction_mesh.size() != self.dp_world_size
                or reduction_mesh.get_local_rank() != self.dp_rank
            ):
                raise ValueError("Validation loss mesh does not match pure DP topology")
            self._dp_group = reduction_mesh.get_group()
            if dist.get_backend(self._dp_group) == "fake":
                raise ValueError("Validation requires a real DP reduction process group")
        tokenizer_path = getattr(self.tokenizer, "tokenizer_path", None)
        if not tokenizer_path:
            raise ValueError(
                "Fixed validation requires tokenizer_path for fingerprint verification"
            )
        self.artifact = FrozenValidationArtifact(
            resolve_artifact_path(config.artifact_path),
            token_budget=config.token_budget,
            sequence_length=self.seq_len,
            tokenizer_path=tokenizer_path,
        )
        # Force divisibility validation at construction, before training begins.
        next(
            sequence_batches(
                self.artifact.num_sequences,
                self.dp_world_size,
                self.dp_rank,
                config.eval_local_batch_size,
            )
        )
        logger.info(
            "Fixed validation: %d label tokens, %d sequences, DP=%d, eval local=%d, "
            "forwards/rank=%d, manifest SHA256=%s; validator.steps is ignored",
            config.token_budget,
            self.artifact.num_sequences,
            self.dp_world_size,
            config.eval_local_batch_size,
            math.ceil(
                self.artifact.num_sequences / self.dp_world_size / config.eval_local_batch_size
            ),
            self.artifact.manifest_sha256,
        )

    @torch.no_grad()
    def validate(self, model_parts: list[torch.nn.Module], step: int) -> None:
        metrics = getattr(self, "metrics_processor", None)
        scope = getattr(type(metrics), "validation_scope", None)
        with scope(metrics) if scope else nullcontext():
            return self._validate(model_parts, step)

    def _validate(self, model_parts: list[torch.nn.Module], step: int) -> None:
        if self.config.token_budget == 0:
            return super().validate(model_parts, step)
        if len(model_parts) != 1:
            raise ValueError("Fixed-budget validation requires one non-pipelined model")
        assert self.artifact is not None
        model = model_parts[0]
        device = next(model.parameters()).device
        # Check all ranks evaluate the same artifact, even across separate mounts.
        if self.dp_world_size > 1:
            identity = torch.tensor(
                list(bytes.fromhex(self.artifact.manifest_sha256)),
                dtype=torch.uint8,
                device=device,
            )
            identities = [torch.empty_like(identity) for _ in range(self.dp_world_size)]
            dist.all_gather(identities, identity, group=self._dp_group)
            if any(not torch.equal(identity, other) for other in identities):
                raise ValueError("Fixed validation manifests differ across DP ranks")

        # Preserve even intentionally mixed train/eval states of child modules.
        training_states = [(module, module.training) for module in model.modules()]
        local_totals = torch.zeros(2, dtype=torch.float64, device=device)
        forwards = 0
        try:
            model.eval()
            for input_dict, labels in self.artifact.batches(
                dp_world_size=self.dp_world_size,
                dp_rank=self.dp_rank,
                local_batch_size=self.config.eval_local_batch_size,
            ):
                input_dict = {key: value.to(device) for key, value in input_dict.items()}
                labels = labels.to(device)
                local_count = (labels != IGNORE_INDEX).sum()
                self.metrics_processor.ntokens_since_last_log += labels.numel()
                inputs, labels, extra_kwargs = self.post_dataloading_process(
                    input_dict, labels, model_parts
                )
                with self.validation_context():
                    predictions = model(inputs, **extra_kwargs)
                    # Omitting global_valid_tokens requests the unnormalized NLL
                    # sum, including for TorchTitan's ChunkedLossWrapper.
                    loss_sum, _ = self.loss_fn(predictions, labels)
                local_totals[0] += loss_sum.detach().to(dtype=torch.float64)
                local_totals[1] += local_count.to(dtype=torch.float64)
                forwards += 1
                del predictions, loss_sum
            if self.dp_world_size > 1:
                dist.all_reduce(local_totals, op=dist.ReduceOp.SUM, group=self._dp_group)
            if not torch.isfinite(local_totals).all():
                raise FloatingPointError("Non-finite fixed validation NLL/token total")
            actual_tokens = int(local_totals[1].item())
            if actual_tokens != self.config.token_budget:
                raise ValueError(
                    f"Validation evaluated {actual_tokens} tokens, expected {self.config.token_budget}"
                )
            loss = (local_totals[0] / local_totals[1]).item()
            self.last_result = {
                "loss": loss,
                "evaluated_tokens": actual_tokens,
                "num_sequences": self.artifact.num_sequences,
                "forwards_per_rank": forwards,
                "eval_local_batch_size": self.config.eval_local_batch_size,
                "dp_world_size": self.dp_world_size,
                "manifest_sha256": self.artifact.manifest_sha256,
            }
            self.metrics_processor.log_validation(
                loss=loss,
                step=step,
                extra_metrics={
                    "validation_metrics/evaluated_tokens": actual_tokens,
                    "validation_metrics/num_sequences": self.artifact.num_sequences,
                    "validation_metrics/forwards_per_rank": forwards,
                    "validation_metrics/eval_local_batch_size": self.config.eval_local_batch_size,
                },
            )
        finally:
            for module, was_training in training_states:
                module.training = was_training
