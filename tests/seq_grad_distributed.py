"""Validate synchronized sequence distances and checkpoint state on two CPU ranks.

Run: torchrun --standalone --nproc_per_node=2 -m tests.seq_grad_distributed
"""

import copy
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh

from tests.test_seq_grad import parents, probe_trainer


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    mesh = init_device_mesh("cpu", (world,))
    trainer = probe_trainer()
    trainer.parallel_dims.dp_replicate = world
    trainer.parallel_dims.world_size = world
    trainer.parallel_dims.get_mesh = lambda name: mesh
    trainer.parallel_dims.get_optional_mesh = lambda name: (
        mesh if name in ("loss", "batch") else None
    )
    trainer.config.training.global_batch_size *= world

    def sum_gradient(gradient):
        dist.all_reduce(gradient)
        return gradient

    trainer.model_parts[0].weight.register_hook(sum_gradient)

    def explicit(data):
        count = torch.tensor(sum(int((labels != -100).sum()) for _, labels in data))
        grad = torch.zeros((), dtype=torch.float64)
        weight = trainer.model_parts[0].weight.detach()
        for inputs, labels in data:
            x, valid = inputs["input"], labels != -100
            grad += (2 * x * (weight * x - labels))[valid].sum()
        dist.all_reduce(count)
        dist.all_reduce(grad)
        return grad / count

    for step in (1, 2):
        trainer.step = step
        data = parents()
        for inputs, _ in data:
            inputs["input"].add_(rank * 0.7)
        if rank:
            data[0][1][0, 1] = -100
        ref = explicit(data)
        expected = {}
        for length in (2, 4, 8):
            cropped = trainer._crop(data, length, "train" if length == 4 else f"probe:{length}")
            expected[length] = abs(float(explicit(cropped) - ref))
        trainer.train_step(trainer.batch_generator(data))
        payload = trainer.metrics_processor.logger.log.call_args.args[0]
        for length, distance in expected.items():
            torch.testing.assert_close(
                payload[f"seq_grad_distance/s{length:05d}"], distance, rtol=1e-12, atol=1e-12
            )
        assert payload["n_tokens_seen"] == 32 * step
        assert payload["seq_grad_reference/valid_tokens"] == 127
        assert payload["seq_grad_cost/probe_tokens"] == 208

    # All ranks stop before entering backward if a rank cannot stage a complete step.
    from torchtitan.components.dataloader import DataloaderExhaustedError

    saved = copy.deepcopy(trainer.state_dict())
    try:
        trainer.train_step(trainer.batch_generator([] if rank else parents()))
    except DataloaderExhaustedError:
        pass
    else:
        raise AssertionError("All ranks must stop on a partial parent batch")
    assert trainer.state_dict() == saved
    directory = [tempfile.mkdtemp(prefix="seq-grad-dcp-") if rank == 0 else None]
    dist.broadcast_object_list(directory)
    dcp.save({"train_state": trainer}, checkpoint_id=directory[0])
    trainer.n_valid_tokens_seen = trainer.ntokens_seen = 0
    dcp.load({"train_state": trainer}, checkpoint_id=directory[0])
    assert trainer.state_dict() == saved
    dist.barrier()
    if rank == 0:
        shutil.rmtree(directory[0])
        print(
            "PASS: sequence distances, SUM normalization, token counts, collective stop and DCP restore"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
