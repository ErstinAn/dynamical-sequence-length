"""Record ten training steps for a source comparison."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import torch
    from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss

    from torchtitan_dynamic_seq_len.grad_var_monitor import GradVarMonitor

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rank, world = 0, 1
    device = torch.device(args.device)
    if args.device == "cuda":
        import os

        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from torchtitan.distributed.fsdp import disable_fsdp_gradient_division

        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=("dp_replicate", "dp_shard"))
    results = {}
    for method in ("adamw", "muon"):
        torch.manual_seed(42)
        if args.legacy:
            from torchtitan_dynamic_seq_len.container import RotationOptimizersContainer

            optimizer_config = RotationOptimizersContainer.Config(
                config_path=str(
                    args.source_root / "torchtitan_dynamic_seq_len/configs" / f"{method}.yaml"
                ),
            )
            optimizer_type = RotationOptimizersContainer
        else:
            from torchtitan_dynamic_seq_len.optimizer import ExperimentOptimizers

            optimizer_config = ExperimentOptimizers.Config(method=method)
            optimizer_type = ExperimentOptimizers

        class LanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_embeddings = torch.nn.Embedding(32, 8)
                self.body = torch.nn.Linear(8, 8)
                self.norm = torch.nn.LayerNorm(8)
                self.lm_head = torch.nn.Linear(8, 32, bias=False)
                self.lm_head.weight = self.tok_embeddings.weight

            def forward(self, tokens_BS):
                return self.norm(self.body(self.tok_embeddings(tokens_BS)).tanh())

        model = LanguageModel().to(device)
        if args.device == "cuda":
            fully_shard(model, mesh=mesh)
            disable_fsdp_gradient_division(model)
        optimizer = optimizer_type(optimizer_config, model_parts=[model])
        loss_fn = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(
                num_chunks=2,
                loss_fn=CrossEntropyLoss.Config(global_vocab_size=32),
            )
        )
        loss_fn.set_lm_head(model.lm_head)
        monitor = GradVarMonitor(model.named_parameters(), 4)
        generator = torch.Generator().manual_seed(42 + rank)
        trace = []
        for step in range(10):
            optimizer.zero_grad()
            monitor.begin()
            loss_sum = 0.0
            for _ in range(4):
                tokens_BS = torch.randint(0, 32, (2, 16), generator=generator).to(device)
                labels_BS = torch.randint(0, 32, (2, 16), generator=generator).to(device)
                loss, _ = loss_fn(
                    model(tokens_BS), labels_BS, torch.tensor(128 * world, device=device)
                )
                loss.backward()
                loss_sum += loss.item()
                monitor.observe()
            metrics = monitor.finish()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            trace.append({"step": step + 1, "loss": loss_sum, "grad_norm": grad_norm, **metrics})
            if step == 4:
                state = optimizer.state_dict()
                optimizer = optimizer_type(optimizer_config, model_parts=[model])
                optimizer.load_state_dict(state)
        results[method] = {
            "trace": trace,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

    def local_state(value):
        from torch.distributed.tensor import DTensor

        if isinstance(value, DTensor):
            return value.to_local().detach().cpu()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: local_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [local_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(local_state(item) for item in value)
        return value

    if rank == 0:
        torch.save(local_state(results), args.output)
    if args.device == "cuda":
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
