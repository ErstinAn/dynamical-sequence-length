"""Check ten token checkpoints and loss axes with two CPU ranks."""

import json
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from tests.test_dynamic_seq_bs import batch_loader
from tests.test_token_progress import checkpoint_manager, token_trainer


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    mesh = init_device_mesh("cpu", (world,))
    directory = [tempfile.mkdtemp(prefix="token-milestones-") if rank == 0 else None]
    dist.broadcast_object_list(directory)
    root = Path(directory[0])

    def build():
        trainer = token_trainer(log_freq=4)
        trainer.parallel_dims.dp_replicate = world
        trainer.parallel_dims.world_size = world
        trainer.parallel_dims.get_mesh = lambda name: mesh
        trainer.parallel_dims.get_optional_mesh = lambda name: mesh if name in ("batch", "loss") else None
        trainer.dataloader = batch_loader(dp_rank=rank, dp_world_size=world,
            documents=[" ".join(map(str, range(start, start + 2001))) for start in (0, 2001)])
        def sum_gradient(gradient):
            dist.all_reduce(gradient)
            return gradient
        trainer.model_parts[0].weight.register_hook(sum_gradient)
        trainer.checkpointer = checkpoint_manager(trainer, root)
        return trainer

    trainer = build()
    iterator = trainer.batch_generator(trainer.dataloader)
    for step in range(1, 11):
        trainer.step = step
        trainer.train_step(iterator)
        assert trainer.nominal_tokens_seen == trainer.token_plan.cumulative[step]
        trainer.checkpointer.save(step)
    resumed = build()
    assert resumed.checkpointer.load()
    assert resumed.metrics_processor.training_tokens == 256
    restored = resumed.batch_generator(resumed.dataloader)
    for step in range(11, 21):
        trainer.step = resumed.step = step
        trainer.train_step(iterator)
        resumed.train_step(restored)
        torch.testing.assert_close(trainer.model_parts[0].weight, resumed.model_parts[0].weight, rtol=0, atol=0)
        assert trainer.state_dict() == resumed.state_dict()
        resumed.metrics_processor.log_validation(0.5, step)
        payload = resumed.metrics_processor.logger.log.call_args.args[0]
        assert payload["n_tokens_seen"] == resumed.token_plan.cumulative[step]
        assert payload["n_tokens_seen"] == resumed.ntokens_seen * world
        resumed.checkpointer.save(step, last_step=step == 20)
    dist.barrier()
    if rank == 0:
        assert len(list((root / "checkpoint").glob("step-*/.metadata"))) == 10
        for item in resumed.token_plan.milestones:
            receipt = json.loads((root / f"checkpoint/step-{item.step}/token_milestone.json").read_text())
            assert receipt["tokens"] == item.tokens
        shutil.rmtree(root)
        print("PASS: two-rank nominal counts, ten complete checkpoints, exact recovery, and validation token axes")
    trainer.checkpointer.close()
    resumed.checkpointer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
