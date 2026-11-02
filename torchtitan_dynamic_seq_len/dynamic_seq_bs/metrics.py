"""Record sequence and batch schedules with actual token counts."""

from dataclasses import dataclass

from torchtitan_dynamic_seq_len.dynamic_seq.metrics import DynamicSeqMetricsProcessor


class DynamicSeqBSMetricsProcessor(DynamicSeqMetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(DynamicSeqMetricsProcessor.Config):
        pass

    def _step_metrics(self):
        return {
            **super()._step_metrics(),
            "global_batch_size": self.current_global_batch_size,
            "microbatch_size": self.current_microbatch_size,
            "gradient_accumulation_steps": self.current_accumulation_steps,
        }
