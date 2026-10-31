"""Do checks of Trainer reductions and replicated DCP state with two CPU ranks.

Run: torchrun --standalone --nproc_per_node=2 -m tests.dynamic_seq_distributed
"""

import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh

from tests.test_dynamic_seq import dynamic_trainer


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    mesh = init_device_mesh("cpu", (world,))
    trainer = dynamic_trainer("constant")
    trainer.parallel_dims.dp_replicate = world
    trainer.parallel_dims.world_size = world
    trainer.parallel_dims.get_mesh = lambda name: mesh
    trainer.parallel_dims.get_optional_mesh = lambda name: (
        mesh if name in ("loss", "batch") else None
    )

    def sum_gradient(gradient):
        dist.all_reduce(gradient)
        return gradient

    trainer.model_parts[0].weight.register_hook(sum_gradient)
    for step in (1, 2, 3):
        trainer.step = step
        batches = [({"input": torch.ones(2, 4)}, torch.ones(2, 4)) for _ in range(2)]
        # The ranks have different valid token counts: 16 and 15.
        if rank:
            batches[0][1][0, 0] = -100
        trainer.train_step(trainer.batch_generator(batches))
        payload = trainer.metrics_processor.logger.log.call_args.args[0]
        assert payload["n_tokens_seen"] == 32 * step, payload
        assert payload["valid_tokens_per_step"] == 31, payload
        assert payload["n_valid_tokens_seen"] == 31 * step, payload
    from torchtitan.components.dataloader import DataloaderExhaustedError

    saved = trainer.state_dict()
    try:
        data = [] if rank else [({"input": torch.ones(2, 4)}, torch.ones(2, 4))] * 2
        trainer.train_step(trainer.batch_generator(data))
    except DataloaderExhaustedError:
        pass
    else:
        raise AssertionError("All ranks must cancel when one shard is exhausted")
    assert trainer.state_dict() == saved
    directory = [tempfile.mkdtemp(prefix="dynamic-dcp-") if rank == 0 else None]
    dist.broadcast_object_list(directory)
    dcp.save({"train_state": trainer}, checkpoint_id=directory[0])
    trainer.n_valid_tokens_seen = 0
    trainer.ntokens_seen = 0
    dcp.load({"train_state": trainer}, checkpoint_id=directory[0])
    assert trainer.n_valid_tokens_seen == 93
    assert trainer.ntokens_seen == 48
    assert trainer.step == 3
    if rank == 0:
        print(
            "PASS: two-rank native loss/token reductions and DCP restore with unequal valid counts"
        )
        for path in Path(directory[0]).iterdir():
            path.unlink()
        os.rmdir(directory[0])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
