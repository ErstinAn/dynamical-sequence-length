"""Two-rank token normalization, independent batching, restoration and collective stop."""

import copy
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.components.dataloader import DataloaderExhaustedError

from tests.test_loss_grad_seq import dual_trainer
from torchtitan_dynamic_seq_len.loss_grad_seq.batching import regroup_batches


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh("cpu", (world,))
    trainer = dual_trainer(dynamic_batch=True)
    # Local sample counts 5 and 9 exercise unequal final microbatches on both phases.
    trainer.config.batch_size_scheduler.batch_sizes = [10, 18]
    trainer.config.loss_microbatch_scheduler.batch_sizes = [3, 5]
    trainer.schedules = trainer.config.schedules()
    trainer.parallel_dims.dp_replicate = world
    trainer.parallel_dims.world_size = world
    trainer.parallel_dims.get_mesh = lambda name: mesh
    trainer.parallel_dims.get_optional_mesh = lambda name: (
        mesh if name in ("batch", "loss") else None
    )

    def reduce_gradient(grad):
        dist.all_reduce(grad)
        return grad

    trainer.model_parts[0].weight.register_hook(reduce_gradient)
    total = 0
    for step in range(1, 13):
        trainer.step = step
        plan = trainer._prepare_step()
        data = []
        for i in range(plan.loss_batches):
            size = min(plan.loss_microbatch, plan.global_batch // world - i * plan.loss_microbatch)
            x = torch.arange(size * plan.loss_length, dtype=torch.float64)
            x = x.reshape(size, plan.loss_length) / 100 + rank + i
            y = x + 0.1
            if rank and i == 0:
                y[0, -1] = -100
            data.append(({"input": x}, y))
        crops = regroup_batches(trainer._crop(data), plan.grad_microbatch)
        weight = trainer.model_parts[0].weight.detach()
        numerator = sum(
            (2 * x["input"] * (weight * x["input"] - y))[y != -100].sum() for x, y in crops
        )
        count = torch.tensor(sum(int((y != -100).sum()) for _, y in crops))
        parent_count = torch.tensor(sum(int((y != -100).sum()) for _, y in data))
        parent_loss = sum(((weight * x["input"] - y)[y != -100]).square().sum() for x, y in data)
        for value in (numerator, count, parent_count, parent_loss):
            dist.all_reduce(value)
        trainer.train_step(trainer.batch_generator(copy.deepcopy(data)))
        torch.testing.assert_close(
            trainer.optimizers.gradients[-1][0], numerator / count, rtol=1e-12, atol=1e-10
        )
        payload = trainer.metrics_processor.logger.log.call_args.args[0]
        torch.testing.assert_close(
            payload["loss_metrics/long_avg_loss"],
            (parent_loss / parent_count).item(),
            rtol=1e-12,
            atol=1e-12,
        )
        total += plan.global_batch * plan.grad_length
        assert payload["n_tokens_seen"] == total
        assert payload["valid_tokens_per_step"] == int(count)
    saved = copy.deepcopy(trainer.state_dict())
    for failure in ("empty", "invalid_crop"):
        data = [
            (
                {"input": torch.ones(size, plan.loss_length)},
                torch.ones(size, plan.loss_length),
            )
            for size in [plan.loss_microbatch, plan.global_batch // world - plan.loss_microbatch]
        ]
        if failure == "invalid_crop" and rank:
            for _, labels in data:
                labels.fill_(-100)
        try:
            trainer.train_step(trainer.batch_generator([] if rank and failure == "empty" else data))
        except (DataloaderExhaustedError, ValueError):
            pass
        else:
            raise AssertionError("Every rank must stop before backward")
        assert trainer.state_dict() == saved
    trainer.parent_tokens_seen = 0
    trainer.load_state_dict(saved)
    assert trainer.state_dict() == saved
    dist.barrier()
    if rank == 0:
        print(
            "PASS: independent phase batches, distributed gradients/loss, dynamic schedules, collective stop"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
