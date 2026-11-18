"""Replay all scheduled shapes through one real Llama block on a B200."""

import argparse
import json
import time
from pathlib import Path

import torch
from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.parallel_dims import ParallelDims

from torchtitan_dynamic_seq_len.config_registry import llama3_100m_llama2_adamw_dynamic_seq_bs
from torchtitan_dynamic_seq_len.dynamic_seq_bs.compile import compile_dynamic_blocks
from torchtitan_dynamic_seq_len.run_config import resolve_point
from torchtitan_dynamic_seq_len.schedulers.batch_size import BatchSizeSchedulerConfig
from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "dynamic"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1)
    torch.cuda.set_device(0)
    c = llama3_100m_llama2_adamw_dynamic_seq_bs()
    c.model_spec.model.layers = c.model_spec.model.layers[:1]
    model = c.model_spec.model.build().to(device="cuda", dtype=torch.bfloat16)
    model.init_states(buffer_device=torch.device("cuda"))
    SelectiveAC.Config().build().apply(model)
    parallel = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1)
    apply_compile(
        model, compile_config=CompileConfig(enable=True, backend="inductor"), parallel_dims=parallel
    )
    block = model.layers["0"]
    if args.mode == "dynamic":
        compile_dynamic_blocks([model], CompileConfig(enable=True, backend="inductor"))
    raw, _, _, _ = resolve_point(
        ROOT / "legacy/configs/dynamic_seq_bs/linear_bs_cyclic_seq_constant_lr.yaml", 0
    )
    fixed = raw["fixed"]

    def schedule(prefix, cls):
        return cls(
            **{k[len(prefix) + 1 :]: v for k, v in fixed.items() if k.startswith(prefix + ".")}
        ).build(3840)

    seq = schedule("seq_len_scheduler", SeqLenSchedulerConfig)
    micro = schedule("microbatch_scheduler", BatchSizeSchedulerConfig)
    started = time.perf_counter()
    completed = 0

    def record(status, **extra):
        data = dict(
            mode=args.mode,
            status=status,
            completed=completed,
            seconds=time.perf_counter() - started,
            gpu=torch.cuda.get_device_name(),
            graphs=dict(torch._dynamo.utils.counters["stats"]),
            peak_active_gib=torch.cuda.max_memory_allocated() / 2**30,
            **extra,
        )
        (folder / f"{args.mode}-result.json").write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data), flush=True)

    try:
        for step in range(1, 3841):
            b, s = micro.value_for_step(step), seq.length_for_step(step)
            block.train()
            x = torch.randn(b, s, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            pos = torch.arange(s, device="cuda").expand(b, -1).contiguous()
            mask = model.get_attention_masks(positions=pos)
            y = block(x, mask, pos)
            loss = y.float().square().mean()
            loss.backward()
            assert torch.isfinite(loss)
            block.zero_grad(set_to_none=True)
            del x, pos, y, loss
            if step % 20 == 0:
                block.eval()
                with torch.no_grad():
                    x = torch.randn(32, 2048, 768, device="cuda", dtype=torch.bfloat16)
                    pos = torch.arange(2048, device="cuda").expand(32, -1).contiguous()
                    mask = model.get_attention_masks(positions=pos)
                    y = block(x, mask, pos)
                    assert torch.isfinite(y).all()
                    del x, pos, y
            completed = step
            if step == 1 or step % 100 == 0 or step in (1528, 1529, 1530):
                torch.cuda.synchronize()
                record("running", current_shape=[b, s])
        record("passed")
    except Exception as error:
        record("failed", exception=type(error).__name__, message=str(error), failed_shape=[b, s])
        raise


if __name__ == "__main__":
    main()
