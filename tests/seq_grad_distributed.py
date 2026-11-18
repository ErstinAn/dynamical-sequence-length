"""Two CPU or four GPU ranks: compare complete probes with an explicit token mean."""

import copy
import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from tests.test_seq_grad import explicit_gradient, probe_trainer
from torchtitan_dynamic_seq_len.seq_grad.sampler import PairedWindowProbeLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling", choices=("independent", "paired_window"), default="independent")
    args = parser.parse_args()
    cuda = os.environ.get("SEQ_GRAD_TEST_CUDA") == "1"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(seconds=180))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        mesh = init_device_mesh(device.type, (world,))
        trainer = probe_trainer()
        trainer.device = device
        trainer.model_parts[0].to(device)
        trainer.parallel_dims.dp_replicate = world
        trainer.parallel_dims.world_size = world
        trainer.parallel_dims.get_mesh = lambda name: mesh
        trainer._probe_rank, trainer._probe_degree = rank, world
        cfg = trainer.config.seq_grad
        cfg.sampling = args.sampling
        if cfg.sampling == "paired_window":
            trainer.probe_loader = PairedWindowProbeLoader(trainer.probe_loader, cfg.reference_length)
            cfg.lengths.append(cfg.reference_length)
        cfg.probe_global_batch_size = 16
        model = trainer.model_parts[0]
        # SUM each microbatch, as in the pinned replicated FSDP path. The real
        # FSDP2 path is separately exercised by the production smoke run.
        def sum_gradient(gradient):
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            return gradient
        for parameter in model.parameters():
            parameter.register_hook(sum_gradient)
        for size in (1, 2, 4):
            cfg.probe_local_batch_size = size
            trainer._last_event = -1
            trainer.maybe_monitor(0)
            payload = trainer.metrics_processor.logger.log.call_args.args[0]
            refs = {}
            for length in [cfg.reference_length, *cfg.lengths]:
                data = trainer.probe_loader.sample_batch(length=length, global_sequences=16,
                    local_batch_size=16, rank=0, degree=1, seed=cfg.seed, step=0)
                reference_model = copy.deepcopy(model).cpu()
                refs[length] = explicit_gradient(reference_model, data)
            for length in cfg.lengths:
                expected = (refs[length] - refs[cfg.reference_length]).norm().item()
                torch.testing.assert_close(payload[f"context_bias/distance_s{length}"],
                                           expected, rtol=1e-12, atol=1e-12)
            assert trainer.ntokens_seen == 0
        # A rank-local malformed sample must fail all ranks before backward.
        original = trainer.probe_loader.sample_batch
        if rank == 0:
            trainer.probe_loader.sample_batch = lambda **kwargs: []
        try:
            trainer._probe(4, 2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Collective input validation did not fail")
        trainer.probe_loader.sample_batch = original
        if rank == 0:
            print(f"PASS: DP={world} {device.type}, sampling={args.sampling}, SUM token mean, unequal labels, local batches 1/2/4, collective error", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
