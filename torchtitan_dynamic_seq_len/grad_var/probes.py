"""Isolate X events from the optimizer trajectory and training data stream."""

import hashlib
import json
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchtitan.components.loss import IGNORE_INDEX, ChunkedLossWrapper
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.rope import RoPE
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentDataLoader
from torchtitan_dynamic_seq_len.grad_var.monitor import XMonitor
from torchtitan_dynamic_seq_len.grad_var.sampler import ParentWindowSampler, RandomWindowSampler
from torchtitan_dynamic_seq_len.grad_var.validation import required_context_length


def validate_config(config):
    cfg, p = config.grad_var_monitor, config.parallelism
    cfg.validate(p.data_parallel_replicate_degree)
    if (
        any(
            getattr(p, name) != 1
            for name in (
                "data_parallel_shard_degree",
                "tensor_parallel_degree",
                "pipeline_parallel_degree",
                "context_parallel_degree",
                "expert_parallel_degree",
            )
        )
        or p.spmd_backend != "default"
    ):
        raise ValueError("X supports replicated DP only")
    if (
        not isinstance(config.dataloader, LongDocumentDataLoader.Config)
        or config.dataloader.packing != "none"
    ):
        raise ValueError("X requires the unpacked long-document training loader")
    if config.dataloader.split != "train":
        raise ValueError("X source pool must use the training split")
    capacity = required_context_length(config)
    for layer in config.model_spec.model.layers:
        attention = getattr(layer, "attention", None)
        if attention is not None and capacity > attention.rope.max_seq_len:
            raise ValueError(f"X length {capacity} exceeds model position limit")
    if isinstance(config.loss, ChunkedLossWrapper.Config) and (
        config.loss.num_chunks <= 0 or any(s % config.loss.num_chunks for s in cfg.lengths)
    ):
        raise ValueError("Every X candidate length must divide into loss chunks")
    if cfg.seed < 0:
        raise ValueError("X probe seed must be nonnegative")
    if config.metrics.save_for_all_ranks:
        raise ValueError("X metrics require the single existing logging rank")
    validator = config.validator
    if validator.enable and validator.token_budget > 0:
        if not validator.artifact_path:
            raise ValueError("X fixed validation requires an explicit long-document artifact_path")
        manifest = json.loads((Path(validator.artifact_path) / "manifest.json").read_text())
        length = validator.eval_seq_len or config.training.seq_len
        if (
            manifest.get("sequence_length") != length
            or manifest.get("token_budget") != validator.token_budget
        ):
            raise ValueError("X validation artifact length/token budget differs from configuration")
        source = manifest.get("source", {})
        if source.get("split") != "validation" or (
            config.dataloader.expected_manifest_sha256
            and source.get("manifest_sha256") != config.dataloader.expected_manifest_sha256
        ):
            raise ValueError("X validation must use the matching long-document validation split")
        if isinstance(config.loss, ChunkedLossWrapper.Config) and length % config.loss.num_chunks:
            raise ValueError("Validation length must divide into loss chunks")
        if manifest.get("schema_version") == 2:
            from torchtitan_dynamic_seq_len.data.packed_documents import PACKING_PROTOCOL

            if manifest.get("packing_protocol") != PACKING_PROTOCOL:
                raise ValueError("Unsupported packed validation protocol")
            for layer in config.model_spec.model.layers:
                attention = getattr(layer, "attention", None)
                if attention is not None and not isinstance(
                    attention.inner_attention, (FlexAttention.Config, VarlenAttention.Config)
                ):
                    raise ValueError("Packed validation requires document-masked attention")


def initialize(trainer):
    cfg = trainer.config.grad_var_monitor
    dims = trainer.parallel_dims
    cfg.validate(dims.dp_replicate)
    trainer._x_probe_active = False
    trainer._x_dp_degree = dims.dp_replicate
    trainer._grad_var_group = dims.get_mesh("batch").get_group() if dims.dp_enabled else None
    trainer._x_dp_rank = dist.get_rank(trainer._grad_var_group) if dims.dp_enabled else 0
    # The pinned trainer has no GradScaler; fail explicitly if one is introduced.
    if any(getattr(trainer, attr, None) is not None for attr in ("scaler", "grad_scaler")):
        raise ValueError(
            "X probes require an independent AMP scaling path; GradScaler is unsupported"
        )
    trainer.train_loader = trainer.dataloader
    trainer.probe_loader = (
        RandomWindowSampler(trainer.dataloader.dataset, cfg.lengths, cfg.seed)
        if cfg.sampling == "independent_random_window"
        else ParentWindowSampler(trainer.dataloader.dataset, cfg.parent_length, cfg.seed)
    )
    trainer.grad_var_monitor = XMonitor(trainer.model_parts[0].named_parameters(), cfg.num_groups)
    trainer._x_replicate_monitor = (
        XMonitor(trainer.model_parts[0].named_parameters(), cfg.replicate_groups)
        if cfg.replicate_groups
        else None
    )
    # Decoder.update_from_config shrinks runtime RoPE caches to training.seq_len.
    # Extend only the cache allocation, retaining theta and scaling parameters.
    capacity = required_context_length(trainer.config)
    for module in trainer.model_parts[0].modules():
        if isinstance(module, RoPE) and module.cache.shape[0] < capacity:
            module.config = replace(module.config, max_seq_len=capacity)
            with torch.device(module.cache.device):
                module.cache = module._precompute_cache()
    logger.info(
        f"X runtime RoPE capacity: {capacity}; training length: {trainer.config.training.seq_len}"
    )
    from torchtitan_dynamic_seq_len.dynamic_seq_bs.compile import compile_dynamic_blocks

    compile_dynamic_blocks(trainer.model_parts, trainer.config.compile)
    metadata = {
        **asdict(cfg),
        "total_sequences": cfg.total_sequences,
        "sampler": trainer.probe_loader.identity,
        "dataset_manifest_sha256": trainer.dataloader.dataset.manifest_hash,
        "tokenizer_sha256": trainer.dataloader.dataset.tokenizer_sha256,
    }
    if cfg.sampling == "parent_prefix":
        # Keep the metadata signature of existing checkpoints unchanged.
        metadata.pop("sampling")
        metadata["parent_pool_size"] = trainer.probe_loader.pool_size
        logger.info(f"X source pool: {trainer.probe_loader.pool_size} complete parent windows")
    else:
        metadata.pop("parent_length")
        metadata.update(
            valid_start_counts=trainer.probe_loader.pool_sizes,
            cross_length_pairing=False,
            replacement=True,
            positions="reset_to_zero",
        )
        logger.info(f"X independent random windows: {trainer.probe_loader.pool_sizes}")
    trainer._x_signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    trainer.metrics_processor.configure_x(metadata)


@contextmanager
def preserve_training_state(trainer, completed_updates):
    model = trainer.model_parts[0]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [trainer.device.index] if trainer.device.type == "cuda" else []
    trainer._x_probe_active = True

    def restore_buffers():
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)

    try:
        with torch.random.fork_rng(devices=devices):
            seed = (
                trainer.config.grad_var_monitor.seed
                + completed_updates * 100003
                + trainer._x_dp_rank
            )
            # Seed only this rank's device; do not alter other CUDA RNG streams.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            yield restore_buffers
    finally:
        restore_buffers()
        trainer.optimizers.zero_grad()
        trainer.grad_var_monitor.reset()
        if getattr(trainer, "_x_replicate_monitor", None) is not None:
            trainer._x_replicate_monitor.reset()
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        trainer._x_probe_active = False


def collective_check(trainer, error):
    failed = torch.tensor(int(error is not None), device=trainer.device)
    if trainer._grad_var_group is not None:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=trainer._grad_var_group)
    if failed.item():
        raise RuntimeError(error or "Another DP rank rejected X probe inputs/statistics")


def compute_group_gradient(trainer, anchors, length):
    cfg = trainer.config.grad_var_monitor
    trainer.optimizers.zero_grad()
    denominator = torch.tensor(cfg.group_sequences * length, device=trainer.device)
    for inputs, labels in trainer.probe_loader.microbatches(
        anchors, length, trainer._x_dp_rank, trainer._x_dp_degree, cfg.local_batch_size
    ):
        error = None
        if tuple(labels.shape) != (cfg.local_batch_size, length) or not bool(
            (labels != IGNORE_INDEX).all()
        ):
            error = "X requires exactly s valid labels per sequence, with no padding"
        collective_check(trainer, error)
        # Bypass the legacy capture hook and batch_generator training counters.
        loss = Trainer.forward_backward_step(
            trainer,
            input_dict={k: v.to(trainer.device) for k, v in inputs.items()},
            labels=labels.to(trainer.device),
            global_valid_tokens=denominator,
        )
        collective_check(trainer, None if bool(torch.isfinite(loss)) else "Nonfinite X probe loss")
    error = None
    try:
        trainer._check_gradient_sync()
        trainer.grad_var_monitor.observe_group_gradient()
    except (ValueError, RuntimeError) as exc:
        error = str(exc)
    collective_check(trainer, error)


def maybe_monitor(trainer, completed_updates):
    cfg = trainer.config.grad_var_monitor
    if (
        completed_updates <= trainer._x_last_event
        or completed_updates % cfg.interval
        or (completed_updates == 0 and not cfg.include_initial)
    ):
        return
    started = time.perf_counter()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    anchors = (
        trainer.probe_loader.sample_groups(completed_updates, cfg.num_groups, cfg.group_sequences)
        if cfg.sampling == "parent_prefix"
        else None
    )
    payload = {"grad_var/step": completed_updates}
    replicate_monitor = getattr(trainer, "_x_replicate_monitor", None)
    replicates = (
        [{} for _ in range(cfg.num_groups // cfg.replicate_groups)]
        if replicate_monitor is not None
        else []
    )
    logger.info(f"X event starts at completed update {completed_updates}")
    with preserve_training_state(trainer, completed_updates) as restore_buffers:
        for length in cfg.lengths:
            if cfg.sampling == "independent_random_window":
                anchors = trainer.probe_loader.sample_groups(
                    completed_updates, cfg.num_groups, cfg.group_sequences, length=length
                )
            trainer.grad_var_monitor.begin()
            for group_index, group in enumerate(anchors):
                restore_buffers()
                compute_group_gradient(trainer, group, length)
                if replicate_monitor is not None:
                    error = None
                    try:
                        if group_index % cfg.replicate_groups == 0:
                            replicate_monitor.begin()
                        replicate_monitor.observe_group_gradient()
                        if (group_index + 1) % cfg.replicate_groups == 0:
                            replicates[group_index // cfg.replicate_groups][
                                f"grad_var/x_s{length}"
                            ] = replicate_monitor.finish(cfg.group_sequences * length)
                    except (ValueError, RuntimeError) as exc:
                        error = str(exc)
                    collective_check(trainer, error)
            error = None
            try:
                payload[f"grad_var/x_s{length}"] = trainer.grad_var_monitor.finish(
                    cfg.group_sequences * length
                )
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
            collective_check(trainer, error)
            logger.info(
                f"X update {completed_updates}, s={length}: {payload[f'grad_var/x_s{length}']:.8g}"
            )
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    seconds = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(trainer.device) / 2**30
        if trainer.device.type == "cuda"
        else 0
    )
    trainer.metrics_processor.emit_probe_metrics(payload)
    if trainer._x_dp_rank == 0 and hasattr(trainer.config, "dump_folder"):
        path = Path(trainer.config.dump_folder) / "x_events.jsonl"
        with path.open("a") as output:
            output.write(
                json.dumps({"metrics": payload, "seconds": seconds, "peak_allocated_gib": peak})
                + "\n"
            )
        if replicates:
            with (path.parent / "x_replicates.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "step": completed_updates,
                            "group_sequences": cfg.group_sequences,
                            "groups_per_replicate": cfg.replicate_groups,
                            "cross_length_pairing": cfg.sampling == "parent_prefix",
                            "replicates": replicates,
                        }
                    )
                    + "\n"
                )
    trainer._x_last_event = completed_updates
    # Exclude probe wall time from the normal training throughput window.
    trainer.metrics_processor.time_last_log += seconds
    logger.info(
        f"X event completed update={completed_updates} seconds={seconds:.3f} peak_allocated_gib={peak:.3f}"
    )
