# Dynamic sequence length and batch sizes

`DynamicSeqBSTrainer` uses three independent schedules.
Each schedule reads the optimizer step, starting at one.
The sequence schedule selects the number of tokens in each sequence.
The global batch schedule selects the number of sequences across all data parallel ranks for one optimizer update.
The microbatch schedule selects the number of sequences per device for one forward and backward pass.

The Trainer calculates the number of accumulation groups:

```text
accumulation_steps = global_batch_size / (data_parallel_degree * microbatch_size)
```

Every scheduled combination must give a positive integer result.
The Trainer checks all steps before model construction.
An invalid combination causes an error that identifies the step and settings.
The Trainer does not round a batch size to make the combination valid.

The Trainer and metrics modules are in `torchtitan_dynamic_seq_len/dynamic_seq_bs/`.
Shared schedules are in `torchtitan_dynamic_seq_len/schedulers/`.
`batch_size.py` supplies separate scheduler instances for global batch and microbatch.
`sequence_length.py` supplies the sequence scheduler.

## Configuration

Select `llama3_100m_llama2_adamw_dynamic_seq_bs` in the YAML `torchtitan_config` field.
Use `configs/dynamic_seq_bs/increasing.yaml` as a complete example.
The existing `dynamic_seq` entry keeps its fixed batch behavior.

The three configuration sections are `seq_len_scheduler`, `batch_size_scheduler`, and `microbatch_scheduler`.
Both batch schedules use the same parameter names and calculation rules.
They have separate configuration objects and can use different periods and boundaries.

| Kind | Batch schedule parameters |
| --- | --- |
| `constant` | `batch_size` |
| `increasing`, `decreasing` | `batch_sizes`, optional `boundaries` |
| `triangular` | `batch_sizes`, `period_steps`, optional `phase_offset_steps` |
| `cyclic` | `min_batch_size`, `max_batch_size`, `period_steps`, optional `round_to_multiple` |
| `piecewise_linear` | `anchor_steps`, `batch_sizes`, optional `repeat` and `round_to_multiple` |

Enter `batch_sizes` in increasing order for the existing level-based schedules.
For `piecewise_linear`, enter values in anchor order. Values can increase, decrease, or stay constant.

For `decreasing`, the scheduler reverses the increasing schedule in time.
A boundary identifies the last step at a level in the increasing schedule.
Empty boundaries divide the training steps between levels as equally as possible.
A cyclic schedule uses the same integer and rounding rules as the [sequence schedule](dynamic_seq_len.md#scheduler-parameters).
The divisibility check still applies after explicit cyclic rounding.

For `triangular`, `phase_offset_steps` advances the schedule within its period.
The default is zero. The offset must be smaller than the period.

For `piecewise_linear`, the first anchor step must be one.
Each anchor has one value. Interpolation selects the nearest integer, or the nearest specified multiple.
Exact ties select the larger value. Anchor values must be divisible by the specified multiple.

Without repetition, the schedule holds the final value after the last anchor.
With `repeat: true`, the last anchor step defines the period.
The [independent schedule profiles](joint_schedule_proposals.md) combine linear batch growth, cyclic lengths, and a four-stage cycle.
They keep the same total training token budget as the fixed-token profiles below.

The example uses these settings with two GPUs:

| Optimizer steps | Sequence length | Global batch | Microbatch | Accumulation groups |
| --- | ---: | ---: | ---: | ---: |
| 1 to 1280 | 512 | 128 | 32 | 2 |
| 1281 to 2560 | 1024 | 256 | 16 | 8 |
| 2561 to 3840 | 2048 | 512 | 8 | 32 |

Set `training.global_batch_size` and `training.local_batch_size` to their scheduled values at step one.
These fields supply the initialization values for TorchTitan.
They do not change during training.
The Trainer stores the current values separately.
Set `training.seq_len` to the model capacity, which must cover all training and validation lengths.

## Fixed token budget profiles

These profiles change global batch size and keep 524,288 tokens per optimizer step:

| Profile | Global batch schedule |
| --- | --- |
| [Increasing](../configs/dynamic_seq_bs/fixed_tokens_bs_increasing.yaml) | 256 to 65,536 across nine stages |
| [Decreasing](../configs/dynamic_seq_bs/fixed_tokens_bs_decreasing.yaml) | The increasing profile in reverse step order |
| [Cyclic](../configs/dynamic_seq_bs/fixed_tokens_bs_cyclic.yaml) | 256 to 65,536 and back, with a period of 256 steps |

Each profile uses 3,840 steps and 2,013,265,920 training tokens.
The learning rate is constant at 0.0005, with no warmup.
Validation runs every 20 optimizer steps with a fixed sequence length of 2,048.
Its 10,240 sequences contain the same 20,971,520 evaluation tokens.
The profiles target the W&B project `dynamic_seq_length/finweb-llama-100m-seqbs-joint-scheduler`.

Global batch levels are powers of two from 256 through 65,536.
The sequence schedule selects `524288 / global_batch_size`, from 2,048 down to 8.
The microbatch schedule selects `global_batch_size / 4`, from 64 through 16,384.
These are three independently configured schedules. The Trainer does not enforce a fixed token mode.

Model capacity is 2,048 for both training and validation.

Two GPUs use two accumulation groups. Four GPUs use one accumulation group.
The YAML files include two-GPU B200 resource examples.
Set the account and resources for the target server before submission.
W&B is disabled in shared configurations.

The increasing and decreasing profiles use these stage durations:
`240, 480, 480, 480, 480, 480, 480, 480, 240` steps.
All three profiles allocate the same total steps and tokens to each batch and sequence pair.
The cyclic profile divides those steps between repeated visits.

The cyclic profile uses 15 periods, each with 16 phases of 16 steps.
It uses the discrete `triangular` scheduler, not the linear `cyclic` scheduler.
The sequence schedule has a 128-step offset. Both batch schedules have zero offset.
Thus, sequence length starts at its maximum when batch size starts at its minimum.

The CPU tests check every step for two and four GPUs.
The new microbatch values still require a GPU memory and throughput test before a full run.

## Data and loss

The Trainer sets the sequence length and microbatch before it reads data for an optimizer step.
`DynamicSeqBSDataLoader` uses the existing document reader and token buffers.
Its dataset collects the scheduled number of sequences into each batch.
The underlying DataLoader does not apply fixed batch grouping.
The Trainer does not rebuild the dataset or iterator at a schedule change.

This mode requires `num_workers=0`, no prefetch, and no persistent workers.
It supports one GPU or replicated data parallelism.
All ranks use the same three schedules.
A partial batch or an exhausted rank cancels the optimizer step on all ranks.
The loader records its state at each batch instead of using a fixed accumulation interval.

TorchTitan calculates loss with the actual valid token count across all accumulation groups and ranks.
The Trainer does not divide the loss by the accumulation count again.
Changing microbatch size changes the calculation groups while preserving the selected global batch size.

## Metrics and validation

Training records use this order after the learning rate:

```text
seq_len
global_batch_size
microbatch_size
gradient_accumulation_steps
```

These values describe the current optimizer step.
Use `metrics.log_freq: 1` to record every schedule change.
For a longer log interval, throughput and FLOPs use the actual tokens from all steps in that interval.
Loss values retain the TorchTitan meaning for the current logged step.

Without masked labels, each optimizer step consumes `global_batch_size * seq_len` tokens.
The total budget is the sum of these products across steps.
The example processes 1,761,607,680 tokens.
`n_tokens_seen` counts all label positions, and `n_valid_tokens_seen` excludes ignored labels.
The two counts can differ when the data contains masked labels.

Validation has separate fixed settings.
The example uses `validator.eval_seq_len: 1024` and `validator.eval_local_batch_size: 32`.
Training schedules do not change these values or the validation token budget.

## Training commands and recovery

The existing training and submission scripts support the new configuration entry.
No separate submission pipeline is necessary.

1. Prepare the tokenizer and environment as specified in the project README.
2. Check the smoke command.

   ```bash
   python scripts/train.py configs/dynamic_seq_bs/smoke.yaml --nproc 2 --dry-run
   ```

3. Run the smoke test inside a GPU allocation.

   ```bash
   python scripts/train.py configs/dynamic_seq_bs/smoke.yaml --nproc 2
   ```

4. Prepare the fixed validation artifact before a FineWeb experiment.
5. Copy the example YAML to an experiment file.
6. Set the account and W&B settings in the experiment file.
7. Submit the experiment through the [existing submission procedure](../cluster/vt_arc/README.md).

A checkpoint stores the data state and a signature of all three schedules.
The signature also includes the training horizon, model capacity, and data parallel degree.
It does not use the current accumulation count as a fixed setting.
Recovery restores the saved counters and calculates the settings from the optimizer step.
A changed schedule or topology causes a recovery error.

## Checks

Run the CPU tests:

```bash
python -m unittest tests.test_dynamic_seq_bs -v
```

Run the distributed test in an environment with local sockets:

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m tests.dynamic_seq_bs_distributed
```

The tests check all schedule kinds, invalid intermediate combinations, token continuity, and changes to batch shape.
They compare gradients and optimizer updates against a full batch reference.
They also check weighted metrics and checkpoint recovery across batch changes.
The distributed test checks unequal valid token counts and cancellation when one rank has no data.

For a short GPU test, use `configs/dynamic_seq_bs/gpu_increasing_cyclic.yaml`.
It uses the 100M model and a local text fixture for 24 optimizer steps.
Global batch size increases from 8 to 16 to 32.
Sequence length cycles between 128 and 512 with a period of eight steps.
Microbatch size decreases from 4 to 2 to 1.

The expected total is 143,360 training tokens.
The test saves a complete checkpoint at step 24.

```bash
python scripts/train.py configs/dynamic_seq_bs/gpu_increasing_cyclic.yaml --nproc 1
```

On A30, this TorchTitan version uses A100 peak FLOPs for the MFU calculation.
Do not use that MFU value for performance comparisons.

CPU checks do not establish GPU memory limits.
Test the largest scheduled combinations on the target GPU before a full experiment.
