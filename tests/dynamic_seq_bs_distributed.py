"""Check variable batches and checkpoint recovery with two CPU ranks.

Run: torchrun --standalone --nproc_per_node=2 -m tests.dynamic_seq_bs_distributed
"""

import copy
import shutil
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.components.dataloader import DataloaderExhaustedError

from tests.test_dynamic_seq_bs import batch_loader, bs_trainer


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    mesh = init_device_mesh("cpu", (world,))

    def build():
        trainer = bs_trainer()
        trainer.parallel_dims.dp_replicate = world
        trainer.parallel_dims.world_size = world
        trainer.parallel_dims.get_mesh = lambda name: mesh
        trainer.parallel_dims.get_optional_mesh = lambda name: (
            mesh if name in ("batch", "loss") else None
        )
        trainer.dataloader = batch_loader(
            dp_rank=rank,
            dp_world_size=world,
            documents=[" ".join(map(str, range(start, start + 2001))) for start in (0, 2001)],
        )

        def sum_gradient(gradient):
            dist.all_reduce(gradient)
            return gradient

        trainer.model_parts[0].weight.register_hook(sum_gradient)
        return trainer

    trainer = build()
    iterator = iter(trainer.dataloader)
    total = valid_total = 0
    for step in (1, 2, 3):
        trainer.step = step
        length = trainer._prepare_step()
        batches = [next(iterator) for _ in range(trainer.gradient_accumulation_steps)]
        if rank:
            batches[0][1][0, 0] = -100
        local_valid = sum(int((labels != -100).sum()) for _, labels in batches)
        denominator = torch.tensor(local_valid)
        dist.all_reduce(denominator)
        reference_weight = trainer.model_parts[0].weight.detach().clone().requires_grad_()
        loss = (
            sum(
                ((inputs["input"] * reference_weight - labels)[labels != -100].square().sum())
                for inputs, labels in batches
            )
            / denominator
        )
        reference_gradient = torch.autograd.grad(loss, reference_weight)[0]
        dist.all_reduce(reference_gradient)
        trainer.train_step(trainer.batch_generator(batches))
        torch.testing.assert_close(
            trainer.optimizers.gradients[-1][0], reference_gradient, rtol=1e-12, atol=1e-8
        )
        total += trainer.current_global_batch_size * length
        valid_total += int(denominator)
        payload = trainer.metrics_processor.logger.log.call_args.args[0]
        assert payload["n_tokens_seen"] == total, payload
        assert payload["n_valid_tokens_seen"] == valid_total, payload
        assert payload["valid_tokens_per_step"] == int(denominator), payload

    directory = [tempfile.mkdtemp(prefix="dynamic-seq-bs-dcp-") if rank == 0 else None]
    dist.broadcast_object_list(directory)

    def state(tr):
        return {
            "train_state": tr,
            "dataloader": tr.dataloader,
            "model": tr.model_parts[0],
            "optimizer": tr.optimizers.optimizer,
            "lr": tr.lr_schedulers.scheduler,
        }

    dcp.save(state(trainer), checkpoint_id=directory[0])
    resumed = build()
    # Initialize optimizer tensors before the checkpoint load.
    resumed.optimizers.optimizer.load_state_dict(
        copy.deepcopy(trainer.optimizers.optimizer.state_dict())
    )
    dcp.load(state(resumed), checkpoint_id=directory[0])
    restored = iter(resumed.dataloader)
    for step in (4, 5, 6):
        trainer.step = resumed.step = step
        trainer.train_step(trainer.batch_generator(iterator))
        resumed.train_step(resumed.batch_generator(restored))
        torch.testing.assert_close(
            trainer.model_parts[0].weight, resumed.model_parts[0].weight, rtol=0, atol=0
        )
        assert trainer.state_dict() == resumed.state_dict()
        expected = trainer.current_global_batch_size * trainer.current_seq_len
        total += expected
        valid_total += expected
        payload = resumed.metrics_processor.logger.log.call_args.args[0]
        assert payload["n_tokens_seen"] == total
        assert payload["n_valid_tokens_seen"] == valid_total

    saved = copy.deepcopy(trainer.state_dict())
    good = [({"input": torch.ones(1, 4)}, torch.ones(1, 4)) for _ in range(4)]
    try:
        trainer.train_step(trainer.batch_generator([] if rank else good))
    except DataloaderExhaustedError:
        pass
    else:
        raise AssertionError("All ranks must stop when one rank has no data")
    assert trainer.state_dict() == saved
    dist.barrier()
    if rank == 0:
        shutil.rmtree(directory[0])
        print(
            "PASS: variable batches, global gradients, token counts, DCP recovery, and coordinated exhaustion"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
