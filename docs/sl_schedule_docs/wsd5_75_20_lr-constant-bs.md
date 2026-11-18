# wsd5_75_20_lr-constant-bs

Use the [experiment launch prompt](experiment_launch_prompt.md) with this specification after the target server passes smoke.

## Reference config

```yaml
reference_config: configs/smoke.yaml
```

Use the 1B model, initial weights, tokenizer, data, optimizer, precision, process layout, and seeds from the reference config. Use `wsd5_75_20_lr-constant-bs` as the experiment group.

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
| `wandb_run_name` | **{sl}-wsd_lr-cons_bs-lr+{x}** | Replace `{sl}` with the schedule ID and `{x}` with the LR as written in `lr_sweep`. |

To limit the batch, replace `null` with a positive integer. For example, `2` allows two runs to train at once.
Pending jobs do not count toward this limit. Slurm site limits still apply.

## Run settings

Use two nodes with eight NVIDIA H200 GPUs per node. Use a global batch size of **256 sequences per optimizer update**. Each run uses **20,132,659,200 non-padding input tokens**.

| Schedule ID | Total updates (`T`) | Warmup updates | Stable updates | Decay updates | Mean sequence length |
|---|---:|---:|---:|---:|---:|
| `staged_5` | 19200 | 960 | 14400 | 3840 | 4096 |
| `fixed_16384` | 4800 | 240 | 3600 | 960 | 16384 |

## Learning rate

Use Warmup-Stable-Decay schedule with 5% linear warmup, 75% at the peak learning rate, and 20% linear decay. Run both length settings at each **peak** learning rate. This gives ten runs per seed. Use the same lr schedule and sequence length table for all five rates.

```yaml
lr_sweep: [5e-4, 1e-3, 2e-3, 4e-3, 8e-3]
```


## Sequence length

Use the row where `start_update <= u < end_update`. Change length only between optimizer updates. Continue the learning rate schedule across each length change.
At zero-based index 15360, the staged run uses the peak LR. The next update uses a lower LR. Both use length 6144.

| Setting | Stage | Start update | End update | Updates | Sequence length |
|---|---:|---:|---:|---:|---:|
| `staged_5` | 1 | 0 | 638 | 638 | 1056 |
| `staged_5` | 2 | 638 | 9290 | 8652 | 2112 |
| `staged_5` | 3 | 9290 | 14316 | 5026 | 3232 |
| `staged_5` | 4 | 14316 | 17887 | 3571 | 6144 |
| `staged_5` | 5 | 17887 | 19200 | 1313 | 16384 |
| `fixed_16384` | 1 | 0 | 4800 | 4800 | 16384 |

## Micro-batch size and run checks

1. Reuse matching `kickoff-smoke` results for each length. Match the GPU, model, precision, and process layout. Test missing lengths or larger candidate batches with complete optimizer updates.
2. Measure peak memory and input tokens per second after startup. Select the fastest stable micro-batch size that fits in memory. Write this size and the accumulation count into the training config for each length.
3. Keep `data_parallel_replicas × sequences_per_replica_per_micro_batch × accumulation_steps = 256`.
4. Check that the intervals cover each update once. Check that every length is a multiple of 32. Verify `sum(updates × sequence_length × 256) = 20,132,659,200` for each run.
5. On restart, restore the model, optimizer, data position, random state, completed update count, and token count. Use the completed update count to select the next LR and sequence length.
6. Use the same initial weights and data order for each paired seed. Evaluate at common token targets. Record the actual token count, validation metrics, and training time.
