"""Calculate metrics for dynamic training with the TorchTitan logger."""

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

from torchtitan.components.metrics import MetricsProcessor, WandBLogger


class DynamicSeqMetricsProcessor(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.reset_window()

    def reset_window(self):
        self.window_flops = 0
        self.window_tokens = 0
        self.window_steps = 0
        self.window_seconds = 0.0
        self._saved_memory = {}
        self.ntokens_since_last_log = 0
        self.data_loading_times = []

    def close(self):
        # TorchTitan calls close while handling a training or validation error.
        # Its default W&B close omits the exit code and reports success.
        if sys.exception() is not None:
            for logger in getattr(self.logger, "_loggers", ()):
                if isinstance(logger, WandBLogger) and logger.wandb.run is not None:
                    logger.wandb.finish(exit_code=1)
        super().close()

    def begin_step(self):
        self.step_started = time.perf_counter()
        self._logged = False

    def record_step(self, *, length, tokens, flops_per_token):
        self.current_seq_len = length
        self.window_steps += 1
        self.window_tokens += tokens
        self.window_flops += tokens * flops_per_token

    def finish_step(self):
        if not self._logged:
            self.window_seconds += time.perf_counter() - self.step_started

    def _capture_memory(self):
        stats = self.device_memory_monitor.get_peak_stats()
        for key, attr in (
            ("memory/max_active(GiB)", "max_active_gib"),
            ("memory/max_active(%)", "max_active_pct"),
            ("memory/max_reserved(GiB)", "max_reserved_gib"),
            ("memory/max_reserved(%)", "max_reserved_pct"),
        ):
            self._saved_memory[key] = max(self._saved_memory.get(key, 0), getattr(stats, attr))

    def _step_metrics(self):
        return dict(seq_len=self.current_seq_len)

    def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        elapsed = self.window_seconds + time.perf_counter() - self.step_started
        merged = dict(extra_metrics or {})  # TorchTitan adds n_tokens_seen before LR.
        merged.update(self._step_metrics())
        merged.update(
            valid_tokens_per_step=self.global_valid_tokens,
            n_valid_tokens_seen=self.global_valid_tokens_seen,
        )
        merged["time_metrics/end_to_end(s)"] = elapsed / self.window_steps
        self._capture_memory()
        merged.update(self._saved_memory)
        # Weight the FLOP count by the token count for each sequence length.
        # The measurement interval can include different sequence lengths.
        self.num_flops_per_token = self.window_flops / self.window_tokens
        self.ntokens_since_last_log = self.window_tokens
        self.time_last_log = time.perf_counter() - elapsed
        if not self.data_loading_times:
            self.data_loading_times = [0.0]
        super().log(step, global_avg_loss, global_max_loss, grad_norm, merged)
        self.reset_window()
        self._logged = True

    @contextmanager
    def validation_scope(self):
        """Keep the training counters and timer separate from validation."""
        self._capture_memory()
        tokens, times, clock = (
            self.ntokens_since_last_log,
            self.data_loading_times,
            self.time_last_log,
        )
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()
        try:
            yield
        finally:
            self.ntokens_since_last_log = tokens
            self.data_loading_times = times
            self.time_last_log = clock
            self.device_memory_monitor.reset_peak_stats()
