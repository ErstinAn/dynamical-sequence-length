"""Run a Llama GPU test with TorchTitan. Save parameters and counters for comparison."""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--constant", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--load-step", type=int, default=-1)
    args = parser.parse_args()
    init_logger()
    name = "llama3_100m_llama2_adamw_constant_lr" if args.native else "llama3_debug_dynamic_seq"
    cli = ["--module", "torchtitan_dynamic_seq_len.config_registry", "--config", name]
    config = ConfigManager().parse_args(cli)
    from torchtitan_dynamic_seq_len.config_registry import model_registry

    config.model_spec = model_registry("sequence_length_debug")
    config.dataloader.dataset = "sequence_length_smoke"
    config.dataloader.pin_memory = False
    config.training.steps = 8
    config.training.seq_len = 64
    config.training.local_batch_size = 2
    config.training.global_batch_size = 8
    config.training.mixed_precision_param = "float32"
    config.training.mixed_precision_reduce = "float32"
    config.compile.enable = args.compile
    config.debug.seed = 123
    config.debug.deterministic = True
    config.loss.num_chunks = 2
    config.validator.enable = False
    config.metrics.enable_wandb = False
    config.metrics.log_freq = 1
    config.checkpoint.enable = True
    config.checkpoint.interval = 2
    config.checkpoint.keep_latest_k = 10
    config.checkpoint.last_save_model_only = False
    config.checkpoint.load_step = args.load_step
    config.dump_folder = str(args.dump)
    if not args.native and args.constant:
        config.seq_len_scheduler.kind = "constant"
        config.seq_len_scheduler.length = 64
    trainer = config.build()
    try:
        trainer.train()
        state = {}
        for name, parameter in trainer.model_parts[0].named_parameters():
            if hasattr(parameter, "full_tensor"):
                parameter = parameter.full_tensor()
            state[name] = parameter.detach().cpu()
        if dist.get_rank() == 0:
            torch.save({"model": state, "trainer": trainer.state_dict()}, args.dump / "result.pt")
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
