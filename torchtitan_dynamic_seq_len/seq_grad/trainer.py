"""Independent packed context-gradient probes at completed optimizer updates."""

import hashlib
import json
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchtitan.components.loss import IGNORE_INDEX, ChunkedLossWrapper
from torchtitan.models.common.attention import (
    FlexAttention, VarlenAttention, create_attention_mask, get_causal_mask_mod,
)
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.rope import RoPE
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from torchtitan_dynamic_seq_len.data.dynamic_seq_loader import DynamicSeqDataLoader
from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentDataLoader
from torchtitan_dynamic_seq_len.data.packed_documents import PackedDocumentDataset
from torchtitan_dynamic_seq_len.dynamic_seq_bs.compile import compile_dynamic_blocks
from torchtitan_dynamic_seq_len.grad_var.trainer import GradVarTrainer
from torchtitan_dynamic_seq_len.seq_grad.metrics import SeqGradMetricsProcessor
from torchtitan_dynamic_seq_len.seq_grad.monitor import SeqGradConfig, SeqGradMonitor
from torchtitan_dynamic_seq_len.seq_grad.validation import SeqGradValidator
from torchtitan_dynamic_seq_len.seq_grad.sampler import (
    PackedProbeLoader, PairedWindowProbeLoader, SingleDocumentProbeLoader, TextProbeLoader, stream_seed,
)
from torchtitan_dynamic_seq_len.training_config import ExperimentTrainerConfig


class SeqGradTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(ExperimentTrainerConfig):
        seq_grad: SeqGradConfig = field(default_factory=SeqGradConfig)

        def validate_seq_grad(self):
            cfg, p = self.seq_grad, self.parallelism
            cfg.validate(p.data_parallel_replicate_degree)
            if any(getattr(p, name) != 1 for name in (
                "data_parallel_shard_degree", "tensor_parallel_degree", "pipeline_parallel_degree",
                "context_parallel_degree", "expert_parallel_degree",
            )) or p.spmd_backend != "default":
                raise ValueError("Sequence probes support single GPU or replicated DP only")
            if self.training.global_batch_size <= 0 or self.training.local_batch_size <= 0:
                raise ValueError("Training requires explicit positive batch sizes")
            loader = self.dataloader
            if cfg.parent_population == "single_document" and not isinstance(loader, LongDocumentDataLoader.Config):
                raise ValueError("Single-document probe parents require the indexed long-document corpus")
            if not isinstance(loader, (DynamicSeqDataLoader.Config, LongDocumentDataLoader.Config)):
                raise ValueError("Sequence probes require a supported production packing loader")
            if loader.num_workers or loader.persistent_workers or loader.prefetch_factor is not None:
                raise ValueError("Sequence probes require synchronous production packing")
            if isinstance(loader, LongDocumentDataLoader.Config) and (
                loader.packing != "stream" or loader.split != "train"
            ):
                raise ValueError("Sequence probes require stream packing on the training split")
            if not isinstance(self.metrics, SeqGradMetricsProcessor.Config):
                raise ValueError("Sequence probes require SeqGradMetricsProcessor")
            if self.metrics.save_for_all_ranks:
                raise ValueError("Probe metrics use the existing single logging rank")
            if self.lr_scheduler.warmup_steps != 0 or self.lr_scheduler.min_lr_factor != 1.0:
                raise ValueError("Sequence experiment requires constant LR: warmup=0, min_lr_factor=1")
            lengths = [self.training.seq_len, cfg.reference_length, *cfg.lengths]
            if self.validator.enable:
                lengths.append(self.validator.eval_seq_len or self.training.seq_len)
            for layer in self.model_spec.model.layers:
                attention = layer.attention
                if max(lengths) > attention.rope.max_seq_len:
                    raise ValueError("Requested probe exceeds the model position limit")
                if not isinstance(attention.inner_attention, (FlexAttention.Config, VarlenAttention.Config)):
                    raise ValueError("Packed probes require document-blocked Flex/Varlen attention")
                if cfg.parent_population == "single_document" and not isinstance(attention.inner_attention, FlexAttention.Config):
                    raise ValueError("Single-document probes currently require FlexAttention")
            if isinstance(self.loss, ChunkedLossWrapper.Config) and (
                self.loss.num_chunks <= 0 or any(s % self.loss.num_chunks for s in lengths)
            ):
                raise ValueError("All training, probe and validation lengths must divide into loss chunks")

    def __init__(self, config):
        config.validate_seq_grad()
        if config.validator.enable:
            config.validator = SeqGradValidator.Config(**{
                item.name: getattr(config.validator, item.name) for item in fields(config.validator)
            })
        self._probe_active, self._last_event = False, -1
        self.n_valid_tokens_seen = 0
        super().__init__(config)
        if self.num_pipeline_parallel_microbatches != 1 or len(self.model_parts) != 1:
            raise ValueError("Sequence probes require one microbatch per backward and one model")
        self._probe_degree = self.parallel_dims.dp_replicate
        self._probe_rank = dist.get_rank(self._group()) if self.parallel_dims.dp_enabled else 0
        config.seq_grad.validate(self._probe_degree)
        source = self.dataloader.dataset
        self.probe_loader = (SingleDocumentProbeLoader(source, config.seq_grad.reference_length)
                             if config.seq_grad.parent_population == "single_document" else
                             PackedProbeLoader(source) if isinstance(source, PackedDocumentDataset)
                             else TextProbeLoader(source))
        if config.seq_grad.sampling == "paired_window":
            self.probe_loader = PairedWindowProbeLoader(self.probe_loader, config.seq_grad.reference_length)
        self.seq_grad_monitor = SeqGradMonitor(self.model_parts[0].named_parameters())
        # Upstream shrinks runtime RoPE storage to training.seq_len. Restore only
        # cache capacity within the already validated intrinsic model limit.
        capacity = max(config.training.seq_len, config.seq_grad.reference_length,
                       config.validator.eval_seq_len if config.validator.enable else 0)
        self._ensure_rope_capacity(capacity)
        compile_dynamic_blocks(self.model_parts, config.compile)
        metadata = {**asdict(config.seq_grad), "sampler": self.probe_loader.identity,
                    "normalization": "SUM gradients; local loss sum / global valid targets",
                    "learning_rate": config.optimizer.learning_rate,
                    "lr_scheduler": asdict(config.lr_scheduler),
                    "training_length": config.training.seq_len,
                    "training_global_batch": config.training.global_batch_size,
                    "training_local_batch": config.training.local_batch_size,
                    "dp_degree": self._probe_degree}
        self._signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        self.metrics_processor.configure_probe(metadata)

    def _ensure_rope_capacity(self, capacity):
        for module in self.model_parts[0].modules():
            if isinstance(module, RoPE) and module.cache.shape[0] < capacity:
                module.config = replace(module.config, max_seq_len=capacity)
                with torch.device(module.cache.device):
                    module.cache = module._precompute_cache()

    def _group(self):
        return self.parallel_dims.get_mesh("batch").get_group() if self.parallel_dims.dp_enabled else None

    def post_dataloading_process(self, input_dict, labels):
        if not self._probe_active or self.config.seq_grad.parent_population != "single_document":
            return super().post_dataloading_process(input_dict, labels)
        # Skip native packed-document mask construction only for these probes.
        # Training and validation still use their original packed masks.
        positions = input_dict["positions"]
        inputs, labels, kwargs = super().post_dataloading_process(
            {key: value for key, value in input_dict.items() if key != "positions"}, labels)
        kwargs["positions"] = positions
        attention = self.model_config.first_attention.inner_attention
        kwargs["attention_masks"] = create_attention_mask(
            get_causal_mask_mod(), positions.shape[0], None, positions.shape[1], positions.shape[1],
            device=positions.device, BLOCK_SIZE=attention.block_size,
            separate_full_blocks=not is_in_batch_invariant_mode())
        return inputs, labels, kwargs

    def _collective_check(self, error):
        failed = torch.tensor(int(error is not None), device=self.device)
        if self.parallel_dims.dp_enabled:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self._group())
        if failed.item():
            raise RuntimeError(error or "Another DP rank rejected sequence probe inputs/statistics")

    def _statistics(self, capture=False):
        error, result = None, None
        try:
            GradVarTrainer._check_gradient_sync(self)
            result = self.seq_grad_monitor.capture() if capture else self.seq_grad_monitor.distance()
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        self._collective_check(error)
        return result

    @contextmanager
    def _preserve_state(self):
        buffers = [(buffer, buffer.detach().clone()) for buffer in self.model_parts[0].buffers()]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        devices = [self.device.index] if self.device.type == "cuda" else []
        self._probe_active = True

        def restore_buffers():
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)

        try:
            with torch.random.fork_rng(devices=devices):
                yield restore_buffers
        finally:
            restore_buffers()
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            self.optimizers.zero_grad()
            self.seq_grad_monitor.reset()
            if isinstance(self.probe_loader, PairedWindowProbeLoader):
                self.probe_loader.clear_event()
            self._probe_active = False

    def _probe(self, length, completed_updates):
        cfg = self.config.seq_grad
        error, batches = None, None
        try:
            batches = self.probe_loader.sample_batch(
                length=length, global_sequences=cfg.probe_global_batch_size,
                local_batch_size=cfg.probe_local_batch_size,
                rank=self._probe_rank, degree=self._probe_degree, seed=cfg.seed, step=completed_updates,
            )
            if len(batches) != cfg.probe_global_batch_size // (self._probe_degree * cfg.probe_local_batch_size):
                raise ValueError("Incomplete probe batch")
            if any(tuple(labels.shape) != (cfg.probe_local_batch_size, length)
                   or "input" not in inputs for inputs, labels in batches):
                raise ValueError("Probe labels or sequence count differ from configuration")
            if cfg.parent_population == "single_document" and any(
                (labels == IGNORE_INDEX).any() or not torch.equal(
                    inputs["positions"], torch.arange(length).expand_as(inputs["positions"]))
                for inputs, labels in batches
            ):
                raise ValueError("Single-document probes require contiguous positions and no ignored targets")
        except Exception as exc:
            error = str(exc)
        self._collective_check(error)
        count = sum(int((labels != IGNORE_INDEX).sum()) for _, labels in batches)
        valid_tokens = torch.tensor(count, dtype=torch.int64, device=self.device)
        if self.parallel_dims.dp_enabled:
            dist.all_reduce(valid_tokens, group=self._group())
        self._collective_check(None if valid_tokens.item() > 0 else "No valid probe targets")
        self.optimizers.zero_grad()
        seed = stream_seed(cfg.seed, completed_updates, length, self._probe_rank, "model")
        torch.random.default_generator.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % 2**32)
        loss = torch.zeros((), device=self.device)
        for inputs, labels in batches:
            # Match native Trainer transfer and post_dataloading_process. Preserve
            # non-tensor metadata; production code builds masks from positions.
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            loss += self.forward_backward_step(input_dict=inputs, labels=labels.to(self.device),
                                               global_valid_tokens=valid_tokens).detach()
        self._collective_check(None if torch.isfinite(loss).item() else "Nonfinite probe loss")
        return int(valid_tokens)

    def maybe_monitor(self, completed_updates):
        cfg = self.config.seq_grad
        if completed_updates <= self._last_event or not cfg.should_monitor(completed_updates):
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        payload, counts = {"context_bias/step": completed_updates}, {}
        logger.info(f"Sequence distance event starts at completed update {completed_updates}")
        with self._preserve_state() as restore_buffers:
            for index, length in enumerate([cfg.reference_length, *cfg.lengths]):
                restore_buffers()
                counts[str(length)] = self._probe(length, completed_updates)
                if index == 0:
                    self._statistics(capture=True)
                else:
                    payload[f"context_bias/distance_s{length}"] = self._statistics()
                logger.info(f"Sequence probe update={completed_updates} length={length} valid_targets={counts[str(length)]}")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0
        cost = torch.tensor([seconds, peak], dtype=torch.float64, device=self.device)
        if self.parallel_dims.dp_enabled:
            dist.all_reduce(cost, op=dist.ReduceOp.MAX, group=self._group())
        max_seconds, peak = cost.tolist()
        self.metrics_processor.emit_probe_metrics(payload)
        if self._probe_rank == 0 and hasattr(self.config, "dump_folder"):
            path = Path(self.config.dump_folder) / "seq_grad_events.jsonl"
            with path.open("a") as output:
                output.write(json.dumps(dict(metrics=payload, seconds=max_seconds,
                                             peak_allocated_gib=peak, valid_targets=counts,
                                             sampling=cfg.sampling)) + "\n")
        self._last_event = completed_updates
        self.metrics_processor.time_last_log += seconds
        logger.info(f"Sequence event completed update={completed_updates} seconds={max_seconds:.3f} peak_allocated_gib={peak:.3f}")

    def train_step(self, data_iterator):
        self.maybe_monitor(self.step - 1)
        self._training_calls = 0
        result = super().train_step(data_iterator)
        self.maybe_monitor(self.step)
        return result

    def forward_backward_step(self, *, input_dict, labels, global_valid_tokens):
        previous_tokens = self.ntokens_seen
        try:
            loss = super().forward_backward_step(input_dict=input_dict, labels=labels,
                                                 global_valid_tokens=global_valid_tokens)
        finally:
            if self._probe_active:
                self.ntokens_seen = previous_tokens
        if not self._probe_active:
            self._training_calls += 1
            if self._training_calls == 1:
                self.n_valid_tokens_seen += int(global_valid_tokens)
        return loss

    def state_dict(self):
        return {**super().state_dict(), "n_valid_tokens_seen": self.n_valid_tokens_seen,
                "seq_grad_signature": self._signature, "seq_grad_last_event": self._last_event}

    def load_state_dict(self, state_dict):
        if state_dict.get("seq_grad_signature") != self._signature:
            raise ValueError("Checkpoint sequence probe protocol or batch topology differs")
        super().load_state_dict(state_dict)
        self.n_valid_tokens_seen = state_dict["n_valid_tokens_seen"]
        self._last_event = state_dict["seq_grad_last_event"]
