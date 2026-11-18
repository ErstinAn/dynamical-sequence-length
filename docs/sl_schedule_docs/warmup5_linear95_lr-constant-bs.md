# warmup5_linear95_lr-constant-bs

Use the [experiment launch prompt](experiment_launch_prompt.md) with this specification after the target server passes smoke.

## Reference config

```yaml
reference_config: configs/smoke.yaml
```

Use the 1B model, initial weights, tokenizer, data, optimizer, precision, process layout, and seeds from the reference config. Use `warmup5_linear95_lr-constant-bs` as the experiment group. Apply the settings below to both runs in each pair.

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
| `wandb_run_name` | **{sl}-warmup_lineardecay_lr-cons_bs-lr+{x}** | Replace `{sl}` with the schedule ID and `{x}` with the LR as written in `lr_sweep`. |

To limit the batch, replace `null` with a positive integer. For example, `2` allows two runs to train at once.
Pending jobs do not count toward this limit. Slurm site limits still apply.

## Run settings

Use two nodes with eight NVIDIA H200 GPUs per node. Use a global batch size of **256 sequences per optimizer update**. Count full sequences of non-padding input tokens. Each run uses **20,132,659,200 input tokens**.

| Schedule ID | Optimizer updates (`T`) | Warmup updates (`W`) | Decay updates | Mean sequence length |
|---|---:|---:|---:|---:|
| `staged_5` | 19200 | 960 | 18240 | 4096 |
| `fixed_16384` | 4800 | 240 | 4560 | 16384 |

## Learning rate

Use 5% linear warmup and 95% linear decay. Use each of these **peak** learning rates with both settings. This gives ten runs per seed. Keep the sequence length table fixed across the sweep.

```yaml
lr_sweep: [5e-4, 1e-3, 2e-3, 4e-3, 8e-3]
```

## Sequence length

Select the row for `start_update <= u < end_update`. Change length between complete optimizer updates. Continue the LR schedule across every length change.

| Setting | Stage | Start update | End update | Updates | Sequence length |
|---|---:|---:|---:|---:|---:|
| `staged_5` | 1 | 0 | 539 | 539 | 1792 |
| `staged_5` | 2 | 539 | 11658 | 11119 | 2688 |
| `staged_5` | 3 | 11658 | 16063 | 4405 | 3648 |
| `staged_5` | 4 | 16063 | 17904 | 1841 | 5696 |
| `staged_5` | 5 | 17904 | 19200 | 1296 | 16384 |
| `fixed_16384` | 1 | 0 | 4800 | 4800 | 16384 |

Use the exact integer boundaries on restart.

## Micro-batch size and run checks

1. Reuse matching `kickoff-smoke` results for each length. Run a smoke test for missing lengths or larger candidate batches. Match the hardware, precision, model, and process layout.
2. Test complete optimizer updates. Measure peak memory and input tokens per second after startup. Select the fastest stable micro-batch size that fits in memory.
3. Write the selected micro-batch size and accumulation count into the training config for each length. Keep `data_parallel_replicas × sequences_per_replica_per_micro_batch × accumulation_steps = 256`.
4. Check that the intervals cover each update once. Check that all lengths are multiples of 32. Verify `sum(updates × sequence_length × 256) = 20,132,659,200` for each setting.
5. At each checkpoint, save the model, optimizer, data position, random state, completed update count, and token count. Use the completed update count to restore the LR and length. Advance the count only after an accepted update.
6. Use the same initial weights, data order, and evaluation rules for each paired seed. Evaluate at common token targets. Record the actual token count at evaluation, validation metrics, and training time for all ten runs.
