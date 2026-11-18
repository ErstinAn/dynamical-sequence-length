# constant_lr-constant-bs

Use the [experiment launch prompt](experiment_launch_prompt.md) with this specification after the target server passes smoke.

## Reference config

```yaml
reference_config: configs/smoke.yaml
```

Resolve this path from the repository root. Get the model, initial weights, data, tokenizer, optimizer, seeds, runtime settings, and evaluation settings from this config. Apply the experiment settings below as overrides. Keep all other settings the same across runs.

Use `constant_lr-constant-bs` as the experiment group name in config filenames and output paths.
Use the W&B run name from the submission settings below.

For phase 1, compare only `staged_5` and `fixed_16384`. Use the same supplied 1B model for all runs. Keep both sequence length schedules fixed during the learning rate sweep. Use the exact stage tables below.

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
| `wandb_project` | **Llama3-1B-20B-LongSeq_Fineweb-edu** | Formal W&B project name. |
| `wandb_run_name` | **{sl}-cons_lr-cons_bs-lr+{x}** | Replace `{sl}` with the schedule ID and `{x}` with the LR as written in `lr_sweep`. |

To limit the batch, replace `null` with a positive integer. For example, `2` allows two runs to train at once.
Pending jobs do not count toward this limit. Slurm site limits still apply.

## Phase 1 learning rate sweep

Use this exact list for both settings:

```yaml
lr_sweep: [5e-4, 1e-3, 2e-3, 4e-3, 8e-3]
```

Keep the selected learning rate constant within each run. Do not add values or change the sequence length schedule for a learning rate.

Run both settings at all five learning rates. This gives **10 runs per seed**. Use the seed list from the reference config.

Use the same initial weights and data order for each paired seed. Sequence boundaries can differ between schedules.

## Common settings

- Use a global batch size of **256 sequences per optimizer update**.
- Use **20,132,659,200 input tokens per run**.
- Keep the learning rate constant within each run. Do not use warmup or decay.
- Use each learning rate and each seed with both settings.
- Get the learning rate from `lr_sweep`. Ignore learning rate values in old schedule CSV files.
- Use sequence lengths that are multiples of **32**. Do not exceed **16384**.

`T` is the total number of optimizer updates. It does not count gradient accumulation steps.

Count non-padding model input tokens. Each update uses `256 * sequence_length` tokens. Use full sequences of the specified length. Do not count copies on other GPU ranks as extra tokens. If some targets are masked, record their count separately.

## Run settings

| Schedule ID | `T` | Mean sequence length | Global batch size |
|---|---:|---:|---:|
| `staged_5` | 19200 | 4096 | 256 sequences |
| `fixed_16384` | 4800 | 16384 | 256 sequences |

Both settings have the same token budget. Update counts differ. Compute and training time can also differ.

## Stage tables

Use a zero-based update index `u`. Select the row where:

```text
start_update <= u < end_update
```

The start is included. The end is excluded. Keep the same length for all gradient accumulation steps in one update. Increase `u` after the optimizer update. Stop at `u == T`.

### Fixed sequence length 16384

| Schedule ID | `start_update` | `end_update` | Sequence length |
|---|---:|---:|---:|
| `fixed_16384` | 0 | 4800 | 16384 |

### 5-stage schedule

Use schedule ID `staged_5` and `T=19200`.

| Stage | `start_update` | `end_update` | Sequence length |
|---:|---:|---:|---:|
| 1 | 0 | 7734 | 2368 |
| 2 | 7734 | 13003 | 3008 |
| 3 | 13003 | 16289 | 4256 |
| 4 | 16289 | 18057 | 6656 |
| 5 | 18057 | 19200 | 16384 |

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

Use the same validation data, input lengths, and metrics for all runs. The model must support evaluation at length 16384.

Use common token-count targets for intermediate evaluation. Evaluate after the first complete update that reaches each target. Record the actual count. Do not change `T` to align evaluation points.

Report all learning rate results. Use the same validation rule to select a learning rate for each schedule.

Produce two schedule configurations under `constant_lr-constant-bs`. Mark this experiment as `phase_1`. Include the selected micro-batch and accumulation values. Produce a launcher for the 10 setting and learning rate combinations per seed. Record each run's input settings, batch configuration, update count, token count, validation results, and training time.

Prepare the configurations and hardware tests first. Start full training only when the user requests it.
