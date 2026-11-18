# Independent batch and sequence schedules

These three profiles use 3,840 optimizer steps and 2,013,265,920 training tokens each.
Their token counts can change between steps.
They use `DynamicSeqBSTrainer` and three independent scalar schedules.
They do not use a fixed-token mode.

Validation uses length 2,048 every 20 optimizer steps, with 20,971,520 tokens per evaluation.
All three profiles use microbatch equal to global batch divided by eight.
Two GPUs require four accumulation groups. Four GPUs require two accumulation groups.
The proposed combinations require GPU memory and throughput checks before submission.

## Profiles 1 and 2

- [Profile 1: cosine LR](../legacy/configs/dynamic_seq_bs/linear_bs_cyclic_seq_cosine_lr.yaml)
- [Profile 2: constant LR](../legacy/configs/dynamic_seq_bs/linear_bs_cyclic_seq_constant_lr.yaml)

Both profiles increase global batch linearly from 256 to 768 over the full training horizon.
The schedule rounds batch sizes to multiples of eight.
Microbatch increases from 32 to 96 with the same interpolation.
Sequence length cycles from 8 to 2,040 and back, with 15 periods of 256 steps.

The sequence anchors within each period are:

| Step | Length |
| ---: | ---: |
| 1 | 8 |
| 128 | 2040 |
| 129 | 2040 |
| 256 | 8 |

Each slope changes length by 16 tokens per step.
Each endpoint occurs twice across a complete repeated period.
This differs from the older cyclic schedule, which visits each endpoint once per period.
The duplicated endpoints make the sequence trace symmetric under reversal.

Paired batch values from the start and end sum to 1,024.
Paired sequence lengths are equal. Each period has a mean length of 1,024.
These properties give exactly 2,013,265,920 training tokens for both profiles.

Profile 1 uses the native TorchTitan cosine LR scheduler, without warmup.
The first optimizer update uses 0.0005. The final optimizer update uses zero.
The YAML sets `lr_scheduler.total_steps: 3839` because the 3,840 updates use LR indices zero through 3,839.
The training horizon remains 3,840 steps.
Profile 2 uses a constant LR of 0.0005.

## Profile 3: four-stage cycle

[Profile 3](../legacy/configs/dynamic_seq_bs/four_stage_cycle.yaml) repeats a 960-step cycle four times.
Each phase has 240 steps.

| Steps within a cycle | Global batch | Sequence length | Action |
| --- | ---: | ---: | --- |
| 1 to 240 | 256 | 512 | Hold the small batch and short sequence. |
| 241 to 480 | 256 to 1024 | 512 | Increase batch linearly, rounded to multiples of eight. |
| 481 to 720 | 1024 | 1472 | Hold the large batch and long sequence. |
| 721 to 960 | 256 | 512 | Reset both values and hold them. |

This profile switches sequence length at phase boundaries.
It also resets batch size at the start of phase four.
The LR stays at 0.0005.

The large sequence length is 1,472 to match the training token budget exactly:

```text
tokens_per_cycle = 240 * (256 * 512 + 640 * 512 + 1024 * 1472 + 256 * 512)
                 = 503316480
total_tokens     = 4 * 503316480 = 2013265920
```

The second phase has an exact mean global batch of 640 after rounding.

## Scheduler configuration

`piecewise_linear` is shared by the sequence, global batch, and microbatch schedulers.
Use `lengths` for sequence values and `batch_sizes` for either batch setting.
The first anchor step is one. Anchor steps must increase strictly.
Adjacent equal values form a plateau. Consecutive steps can represent a jump.

With `repeat: true`, the final anchor step defines the period.
For example, the global batch anchors for profile 3 are:

```yaml
batch_size_scheduler.kind: piecewise_linear
batch_size_scheduler.anchor_steps: [1, 240, 241, 480, 481, 720, 721, 960]
batch_size_scheduler.batch_sizes: [256, 256, 256, 1024, 1024, 1024, 256, 256]
batch_size_scheduler.repeat: true
batch_size_scheduler.round_to_multiple: 8
```

The schedules depend only on the optimizer step.
Checkpoint signatures include the anchors and repetition setting.
Existing profiles retain their previous checkpoint signatures when these new fields have default values.

## Checks

```bash
python -m unittest tests.test_piecewise_schedules tests.test_dynamic_seq tests.test_dynamic_seq_bs -v
```

The checks cover interpolation, rounding, jumps, repetition, exact budgets, and batch divisibility for two and four GPUs.
They verify the LR used by each optimizer update, including the final zero and checkpoint recovery.
