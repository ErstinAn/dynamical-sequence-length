# constant_lr-constant-bs

Use the [experiment launch prompt](experiment_launch_prompt.md) with this specification after the target server passes smoke.

## Reference config

```yaml
reference_config: configs/smoke.yaml
```

Resolve this path from the repository root. Use the model, initialization, data, tokenizer, optimizer, seeds, and runtime settings from this config. Use its fixed validation set. Apply the experiment settings below as overrides. Keep all other settings the same across runs.

Use `constant_lr-constant-bs` as the experiment group name. Include this name in config names, run names, and output paths.

Prepare training configurations for five sequence length schedules. Use the same supplied 1B model for all runs. Use the exact stage tables below. Do not include the real-length solution as a training run.

## Default hardware

Use the same allocation for smoke and each formal run. Change it only at the user's request.

```yaml
num_nodes: 2
gpus_per_node: 8
num_gpus: 16
gpu_type: H200
```

Get all other training settings from the reference config. Select the micro-batch size on this hardware with the tests below.

## Submission settings

| Field | Value | Use |
|---|---|---|
| `max_concurrent_runs` | `null` | Optional limit across this batch. Omitted or null means no user limit. |
| `lr_sweep` | **Pending** | Positive learning rates, or a minimum, maximum, count, and spacing (linear or logarithmic). |
| `wandb_project` | **Pending** | Required formal W&B destination in `entity/project` format. |
| `wandb_run_name` | **Pending** | Required display name or pattern. Distinguish the schedule, LR, and seed. |

To limit the batch, replace `null` with a positive integer. For example, `2` allows two runs to train at once.
Pending jobs do not count toward this limit. Slurm site limits still apply.

Split `wandb_project` into the YAML fields `wandb.entity` and `wandb.project`.
Set the top-level YAML field `name` to the resolved W&B run name for each combination.

Use the same initial weights and data order for each paired seed. Sequence boundaries can differ between schedules.

## Common settings

- Use a global batch size of **256 sequences per optimizer update**.
- Use **20,132,659,200 input tokens per run**.
- Keep the learning rate constant within each run. Do not use warmup or decay.
- Use each learning rate and each seed with all five schedules.
- Get the learning rate from `lr_sweep`. Ignore learning rate values in old schedule CSV files.
- Use `loss.num_chunks = 32` for all five schedules.
- Do not use a sequence length greater than **16384**.

`T` is the total number of optimizer updates. It does not count gradient accumulation steps.

Count non-padding model input tokens. Each update uses `256 * sequence_length` tokens. Use full sequences of the specified length. Do not count copies on other GPU ranks as extra tokens. If some targets are masked, record their count separately.

## Run settings

| Schedule ID | `T` | Mean sequence length | Global batch size |
|---|---:|---:|---:|
| `staged_5` | 19200 | 4096 | 256 sequences |
| `staged_50` | 19200 | 4096 | 256 sequences |
| `heuristic_4` | 10240 | 7680 | 256 sequences |
| `fixed_4096` | 19200 | 4096 | 256 sequences |
| `fixed_16384` | 4800 | 16384 | 256 sequences |

All five runs have the same token budget. Update counts differ. Compute and training time can also differ.

## Stage tables

Use a zero-based update index `u`. Select the row where:

```text
start_update <= u < end_update
```

The start is included. The end is excluded. Keep the same length for all gradient accumulation steps in one update. Increase `u` after the optimizer update. Stop at `u == T`.

### Fixed lengths

| Schedule ID | `start_update` | `end_update` | Sequence length |
|---|---:|---:|---:|
| `fixed_4096` | 0 | 19200 | 4096 |
| `fixed_16384` | 0 | 4800 | 16384 |

### Heuristic four-stage schedule

Use 2560 updates per stage. Here, `T/4` uses this schedule's `T=10240`.

| Stage | `start_update` | `end_update` | Sequence length |
|---:|---:|---:|---:|
| 1 | 0 | 2560 | 2048 |
| 2 | 2560 | 5120 | 4096 |
| 3 | 5120 | 7680 | 8192 |
| 4 | 7680 | 10240 | 16384 |

### 5-stage schedule

Use schedule ID `staged_5` and `T=19200`.

| Stage | `start_update` | `end_update` | Sequence length |
|---:|---:|---:|---:|
| 1 | 0 | 7734 | 2368 |
| 2 | 7734 | 13003 | 3008 |
| 3 | 13003 | 16289 | 4256 |
| 4 | 16289 | 18057 | 6656 |
| 5 | 18057 | 19200 | 16384 |

### 50-stage schedule

Use schedule ID `staged_50` and `T=19200`.

| Stage | `start_update` | `end_update` | Sequence length |
|---:|---:|---:|---:|
| 1 | 0 | 737 | 1952 |
| 2 | 737 | 1457 | 1984 |
| 3 | 1457 | 2162 | 2016 |
| 4 | 2162 | 2850 | 2080 |
| 5 | 2850 | 3507 | 2112 |
| 6 | 3507 | 4147 | 2176 |
| 7 | 4147 | 4772 | 2208 |
| 8 | 4772 | 5380 | 2272 |
| 9 | 5380 | 5973 | 2336 |
| 10 | 5973 | 6549 | 2400 |
| 11 | 6549 | 7110 | 2432 |
| 12 | 7110 | 7654 | 2496 |
| 13 | 7654 | 8178 | 2560 |
| 14 | 8178 | 8679 | 2656 |
| 15 | 8679 | 9164 | 2720 |
| 16 | 9164 | 9640 | 2784 |
| 17 | 9640 | 10104 | 2848 |
| 18 | 10104 | 10544 | 2944 |
| 19 | 10544 | 10969 | 3040 |
| 20 | 10969 | 11385 | 3104 |
| 21 | 11385 | 11786 | 3200 |
| 22 | 11786 | 12167 | 3296 |
| 23 | 12167 | 12537 | 3392 |
| 24 | 12537 | 12891 | 3520 |
| 25 | 12891 | 13235 | 3616 |
| 26 | 13235 | 13564 | 3744 |
| 27 | 13564 | 13884 | 3872 |
| 28 | 13884 | 14188 | 4000 |
| 29 | 14188 | 14476 | 4128 |
| 30 | 14476 | 14755 | 4256 |
| 31 | 14755 | 15021 | 4416 |
| 32 | 15021 | 15277 | 4576 |
| 33 | 15277 | 15517 | 4768 |
| 34 | 15517 | 15754 | 4928 |
| 35 | 15754 | 15977 | 5120 |
| 36 | 15977 | 16183 | 5344 |
| 37 | 16183 | 16383 | 5568 |
| 38 | 16383 | 16574 | 5792 |
| 39 | 16574 | 16755 | 6048 |
| 40 | 16755 | 16926 | 6336 |
| 41 | 16926 | 17086 | 6624 |
| 42 | 17086 | 17240 | 6944 |
| 43 | 17240 | 17382 | 7264 |
| 44 | 17382 | 17517 | 7648 |
| 45 | 17517 | 17642 | 8032 |
| 46 | 17642 | 17759 | 8480 |
| 47 | 17759 | 17867 | 8928 |
| 48 | 17867 | 17966 | 9472 |
| 49 | 17966 | 18057 | 10016 |
| 50 | 18057 | 19200 | 16384 |

## Micro-batch size

Use large micro-batches to improve training throughput without an out-of-memory (OOM) error. Measure throughput in input tokens per second. A larger micro-batch is not always faster. Select the fastest stable configuration from the tested candidates.

Use these definitions:

- `D`: number of independent data-parallel replicas.
- `m`: sequences per micro-batch on each replica.
- `A`: gradient accumulation steps per optimizer update.

For equal local batches, keep:

```text
D * m * A = 256
```

For pure data parallelism, `D = num_gpus`. With model parallelism, get `D` from the process layout.

1. Read the GPU type, available memory, and runtime settings.
2. Check previous `kickoff-smoke` results from the environment setup tests. Reuse results only when the hardware, model, length, precision, and distributed settings match.
3. Run new smoke tests where results are missing or larger micro-batches need testing.
4. Increase `m` through values that give an integer `A`. Keep the global batch at 256 sequences.
5. Test complete optimizer updates with the training runtime settings. Include optimizer-state allocation. Measure peak memory and throughput after startup.
6. If an OOM error occurs, reduce `m`. Select the fastest stable candidate. Keep memory headroom for normal variation.
7. Write the selected micro-batch size and accumulation count into each generated training config before the full run.
8. Restore the initial training state after the smoke tests.

Select values for each sequence length. They can change at stage boundaries. For stage-based configs, store both values in each stage entry or in an explicit length-to-batch map. Use the framework's supported field names. Do not leave these values only in logs or temporary launch arguments.

Record the tested values, peak memory, throughput, and smoke-test reference. Use the same selected values for the same length across runs with the same hardware and runtime settings.

If no valid configuration fits, report the problem. Do not change the schedule or global batch size. Normalize gradients for the full global batch. Do not apply the accumulation divisor twice.

## Stage changes and restart

Change sequence length only between optimizer updates. Keep the model state, optimizer state, learning rate, and data position at a stage change.

Save the completed update count, token count, model state, optimizer state, data position, and random-number state in each checkpoint. On restart, use the completed update count as the next index `u`.

Do not advance `u` after a skipped optimizer update. Record retries and their extra tokens separately from the tokens in accepted updates.

## Run checks

Before training, check these conditions:

- Intervals start at zero and end at `T`, with no gaps or overlaps.
- Every length is a positive multiple of 32 and is at most 16384.
- Every update uses 256 sequences.
- The token total equals the value below.

```text
sum((end_update - start_update) * sequence_length * 256)
    == 20_132_659_200
```

Do not round lengths or rescale boundaries. Check the update and token counts during training and after restart.

## Evaluation and output

Run 51 validations per run. Use all 936 fixed windows of length 16384 each time.
Each validation measures loss on 15,335,424 prediction targets.
Use cumulative nominal training tokens to divide the budget into 50 equal intervals.
Adjacent targets differ by 402,653,184 nominal training tokens (2% of the budget).

| Validation index | Target | When to validate |
|---|---|---|
| 0 | 0 tokens (0%) | Evaluate the initial model before the first optimizer update. |
| 1 through 49 | `index * 402_653_184` tokens (2% through 98%) | Evaluate after the first complete update that reaches or exceeds the target. |
| 50 | 20,132,659,200 tokens (100%) | Evaluate once after the final optimizer update. |

Record the validation index, target token count, completed update count, and actual training token count.
Plot validation loss in W&B against the actual training token count.
Exclude validation tokens from the training token count.
On restart, count each validation index once across all segments of the run.
Use the W&B run group and validation index to match records across segments.

Save a checkpoint at every fifth validation index from 5 through 50 (10% through 100%).
Keep all ten checkpoints. Index 0 records the initial validation only.

Set `validator.token_intervals: 50` under `fixed` in the formal YAML.
The dry run lists all 51 points under `validations` and the ten saves under `checkpoints`.
When both occur at the same update, validation finishes before the checkpoint save.
Each completed point has a receipt in `validation/point-<index>.json` inside the run directory.
The receipt uses a three-digit index from `000` through `050`.

Report all learning rate results. Use the same validation rule to select a learning rate for each schedule.

Produce five schedule configurations under `constant_lr-constant-bs`. Include the selected micro-batch and accumulation values. Produce a launcher for all schedule, learning rate, and seed combinations. Record each run's input settings, batch configuration, update count, token count, validation results, and training time.

Prepare the configurations and hardware tests first. Start full training only when the user requests it.
