"""Real FSDP2 state isolation with monitoring on/off; run with torchrun on GPUs."""

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchtitan.config import ConfigManager

from torchtitan_dynamic_seq_len.run_config import resolve_point


def capture_fixture_state(trainer):
    """Test-only baseline; production probes never copy model/optimizer state."""
    return {
        "model": copy.deepcopy(trainer.model_parts[0].state_dict()),
        "buffers": {k: v.detach().clone() for k, v in trainer.model_parts[0].named_buffers()},
        "optimizer": copy.deepcopy(trainer.optimizers.state_dict()),
        "scheduler": copy.deepcopy(trainer.lr_schedulers.state_dict()),
        "data": copy.deepcopy(trainer.dataloader.state_dict()),
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(trainer.device).clone()
        if trainer.device.type == "cuda" else None,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }


def restore_fixture_state(trainer, baseline):
    trainer.model_parts[0].load_state_dict(copy.deepcopy(baseline["model"]))
    with torch.no_grad():
        for name, buffer in trainer.model_parts[0].named_buffers():
            buffer.copy_(baseline["buffers"][name])
    # Optimizer.load_state_dict can retain same-device tensor storage; the
    # StatefulDataLoader can retain dataset packing lists. Never hand either
    # loader the reusable baseline: the first trajectory would overwrite it.
    trainer.optimizers.load_state_dict(copy.deepcopy(baseline["optimizer"]))
    trainer.lr_schedulers.load_state_dict(copy.deepcopy(baseline["scheduler"]))
    trainer.dataloader.load_state_dict(copy.deepcopy(baseline["data"]))
    trainer.ntokens_seen = trainer.n_valid_tokens_seen = 0
    trainer._last_event = -1
    torch.set_rng_state(baseline["torch_rng"])
    if baseline["cuda_rng"] is not None:
        torch.cuda.set_rng_state(baseline["cuda_rng"], trainer.device)
    random.setstate(baseline["python_rng"])
    np.random.set_state(baseline["numpy_rng"])


def assert_identical(a, b, path="state"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_identical(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            assert_identical(left, right, f"{path}[{index}]")
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda message: f"{path}: {message}")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b, (path, a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling", choices=("independent", "paired_window"), default="independent")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw, _, _, cli = resolve_point(Path("legacy/configs/seq_grad/smoke.yaml"), 0)
    cfg = ConfigManager().parse_args(["--module", "torchtitan_dynamic_seq_len.config_registry",
                                     "--config", raw["torchtitan_config"], *cli])
    cfg.dump_folder = str(args.output)
    cfg.parallelism.data_parallel_replicate_degree = int(os.environ["WORLD_SIZE"])
    cfg.seq_grad.probe_local_batch_size = 1
    cfg.seq_grad.sampling = args.sampling
    cfg.metrics.enable_wandb = False
    cfg.metrics.log_freq = 1
    trainer = cfg.build()
    try:
        baseline = capture_fixture_state(trainer)
        baseline_check = copy.deepcopy(baseline)
        results = []
        for enabled in (False, True):
            restore_fixture_state(trainer, baseline)
            assert_identical(trainer.optimizers.state_dict(), baseline["optimizer"], "restored_optimizer")
            assert_identical(trainer.model_parts[0].state_dict(), baseline["model"], "restored_model")
            cfg.seq_grad.enable = enabled
            iterator = trainer.batch_generator(trainer.dataloader)
            for step in range(1, 5):
                trainer.step = step
                trainer.train_step(iterator)
            results.append(capture_fixture_state(trainer))
            results[-1]["tokens"] = (trainer.ntokens_seen, trainer.n_valid_tokens_seen)
            assert_identical(baseline, baseline_check, "immutable_baseline")
        assert_identical(*results)
        if dist.get_rank() == 0:
            report = {"passed": True, "world_size": dist.get_world_size(),
                      "exact_equality": ["model", "buffers", "optimizer", "scheduler", "data",
                                         "CPU/CUDA/Python/NumPy RNG", "training tokens"],
                      "baseline_immutable": True, "sampling": args.sampling}
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print("PASS: actual FSDP2 parameters, optimizer, buffers, RNG, dataloader and training tokens unchanged", flush=True)
    finally:
        trainer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
