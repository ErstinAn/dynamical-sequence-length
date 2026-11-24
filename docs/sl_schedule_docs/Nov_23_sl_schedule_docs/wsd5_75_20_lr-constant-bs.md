# wsd5_75_20_lr-constant-bs

Use the [experiment launch prompt](experiment_launch_prompt.md) with this specification after the target server passes smoke.

## Reference config

Use the complete configuration of this finished W&B run as the reference config:

```yaml
reference_run:
  entity: dynamic_seq_length
  project: Llama3-1B-20B_Fineweb-edu-240909-val240909
  run_name: staged_5-wsd_lr-cons_bs-lr_2e-3
  run_id: 858175ff26e34e89b7ff5282a05b08ba
  url: https://wandb.ai/dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/runs/858175ff26e34e89b7ff5282a05b08ba
```

Read it with `wandb.Api().run("dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/858175ff26e34e89b7ff5282a05b08ba").config`. It uses the same LR schedule and LR as this experiment. Take the 1B Llama 3 model, initialization, dataset, tokenizer, optimizer (AdamW, betas, weight decay, gradient clipping), precision, process layout, seeds, and evaluation settings from it. Change only the settings below: update count `steps`, sequence-length schedule, LR warmup/decay lengths for each `T`, micro-batch schedule, activation checkpointing (`activation_checkpoint.mode = "none"`), removal of `validator.freq` (validation by `validator.token_intervals = 50`, as in the launch prompt), W&B run name, and output folder. Keep all other settings the same across runs. Before generating configs, list every field you changed relative to the reference run.

**The W&B reference run is the authority** for the model, initialization, dataset, tokenizer, validation data, optimizer, and precision. If the smoke config on the target server, the launch prompt, or any local default gives a different value, use the W&B value and report the difference. Take only the environment, job file, Slurm settings, shared storage path, and W&B credentials from the smoke record. The W&B config is a resolved record (it contains object representations such as `<function ...>` and sharding reprs); map each field to the repository's YAML format instead of copying it verbatim.

Use `wsd5_75_20_lr-constant-bs` as the experiment group name in config filenames and output paths. This experiment compares **three sequence-length settings** at one fixed learning-rate schedule: `fixed_4096`, `fixed_2048`, and `staged_20`.

## Dataset

**Use exactly the same training and validation data as the W&B reference run. This is the most important requirement of this specification.** Check every dataset field below against the W&B reference run config, not against the smoke config, a local file, an older config, or memory. This section overrides the data instructions of the launch prompt, including the smoke data and the public-dataset restore. Do not substitute another dataset, a re-tokenized copy, a different split, or a different packing mode. The reference run uses:

| Item | Reference value (`config` field) |
|---|---|
| Training dataset | `dataloader.dataset = fineweb_edu_long`, `split = train`, `packing = stream` |
| Training data path | `dataloader.dataset_path = /checkpoint/optim/zhiqibu/data/seqlen_assets/data/fineweb_edu_240909_30b_v1` |
| Training manifest SHA-256 | `dataloader.expected_manifest_sha256 = 2ca374652bc4c4f0c8649d58672f30d17cac83a32d915d3d99c4c1e3a0a47da3` |
| Tokenizer | `hf_assets_path = /checkpoint/optim/zhiqibu/data/seqlen_assets/hf/llama3-tokenizer` |
| Validation data | `validator.artifact_path = /checkpoint/optim/zhiqibu/data/seqlen_assets/data/fineweb_edu_240909_val_v1/validation_packed_16384` |
| Validation manifest SHA-256 | `validator.expected_manifest_sha256 = 8652035304766d8edc00943b899d04a90ac0fb85d2322f807d4d24141950e831` |
| Validation settings | `validator.eval_seq_len = 16384`, `validator.token_budget = 15320140`, `validator.token_intervals = 50`, `validator.eval_local_batch_size = 1` |

`validator.dataloader.dataset` shows `c4_validation`, but the validation data come from `validator.artifact_path`. Verify which one the code actually reads.

**Confirm the dataset with the user before the micro-batch tests and before any formal run.** Show the user the resolved training path, validation path, both manifest hashes (expected from W&B and computed on the server), the tokenizer path, and the first few token IDs of one training batch. Do not start any run until the user explicitly confirms. If a path is missing on this server, a hash differs, or any field is unclear or cannot be matched to the W&B config, stop and ask the user. Do not use a fallback dataset, and do not restore or rebuild the public dataset (`Erstin/fineweb-edu-long-30b`) unless the user asks for it; a rebuilt copy must still match both manifest hashes above.

## Model initialization and model checkpoints

**Match the model initialization and model-checkpoint settings of the W&B reference run exactly.** All three reference runs in the project use identical values:

| Item | Reference value (`config` field) |
|---|---|
| Model | `model_spec.name = llama3`, `model_spec.flavor = 1B` (all `model_spec.model` fields unchanged) |
| Initialization | `checkpoint.initial_load_path = null`: train from random initialization, no pretrained or seed checkpoint (`create_seed_checkpoint = false`) |
| Seed and determinism | `debug.seed = 42`, `debug.deterministic = true` |
| Checkpoint saving | `checkpoint.enable = true`, `folder = checkpoint`, `export_dtype = float32`, `last_save_model_only = false`, `last_save_in_hf = false`; schedule: 10 token-based checkpoints (below) |
| Checkpoint loading | `load_step = -1`, `exclude_from_loading = []`, `initial_load_model_only = false` |

**Save 10 complete checkpoints at 10% increments of the token budget** (every 2,013,265,920 nominal tokens), including the final state, and keep all ten. Save each one after the first complete optimizer update whose cumulative tokens reach the target. Each checkpoint contains the model, optimizer state, data position, random-number state, completed update count, and token count. The table gives the completed update count at each checkpoint:

| Schedule ID | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_4096` | 1920 | 3840 | 5760 | 7680 | 9600 | 11520 | 13440 | 15360 | 17280 | 19200 |
| `fixed_2048` | 3840 | 7680 | 11520 | 15360 | 19200 | 23040 | 26880 | 30720 | 34560 | 38400 |
| `staged_20` | 4590 | 8229 | 11203 | 13522 | 15239 | 16536 | 17640 | 18240 | 18720 | 19200 |

Do not change the initialization or checkpoint settings on your own. If a field is unclear in the code or the saved checkpoints differ from the table, ask the user. Record the actual update and token count of every checkpoint in the run report.

## Default hardware

Use the same allocation for smoke and each formal run. Change it only at the user's request.

```yaml
num_nodes: 1
gpus_per_node: 8
num_gpus: 8
gpu_type: H200
```

Get all other training settings from the reference config. Select the micro-batch size on this hardware with the tests below.

## Submission settings

| Field | Value | Use |
|---|---|---|
| `max_concurrent_runs` | `null` | Optional limit across this batch. Omitted or null means no user limit. |
| `wandb_entity` | **dynamic_seq_length** | Confirmed W&B entity for the formal runs. |
| `wandb_project` | **Llama3-1B-20B_Fineweb-edu-240909-val240909** | Formal W&B project name, the same project as the reference run. |
| `wandb_run_name` | **{sl}-wsd_lr-cons_bs-lr_2e-3** | Replace `{sl}` with the schedule ID. |

To limit the batch, replace `null` with a positive integer. For example, `2` allows two runs to train at once.
Pending jobs do not count toward this limit. Slurm site limits still apply.

## Run settings

Every run uses a global batch size of **256 sequences per optimizer update** and exactly **20,132,659,200 nominal tokens**: count `256 * sequence_length` per optimizer update, which is W&B `n_tokens_seen` (the reference run ends at `n_tokens_seen = 20,132,659,200`). Record the valid-token count (`n_valid_tokens_seen`) separately; do not add updates to make it reach the budget. Use nominal tokens for the checkpoint and validation targets. Each update uses `256 * sequence_length` tokens, so the update count `T` differs between settings. Compute and training time also differ.

| Schedule ID | Optimizer updates (`T`) | Warmup updates | Stable updates | Decay updates | First decay update | Stages | Mean sequence length |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_4096` | 19200 | 960 | 14400 | 3840 | 15360 | 1 | 4096 |
| `fixed_2048` | 38400 | 1920 | 28800 | 7680 | 30720 | 1 | 2048 |
| `staged_20` | 19200 | 960 | 14400 | 3840 | 15360 | 20 | 4096 |

Use **seed 42** (`debug.seed = 42`, as in the reference run) for every run. This gives **3 runs** in total. All runs use the same initial weights and the same data order. Sequence boundaries can differ between schedules.

`T` is the total number of optimizer updates. It does not count gradient accumulation steps.

## Learning rate

Use a Warmup-Stable-Decay schedule: 5% linear warmup to a **peak learning rate of 2e-3**, 75% at the peak, and 20% linear decay to zero. Compute the phase lengths from each run's own `T` (table above). At the zero-based index in the column "First decay update", the run still uses the peak LR; the next update uses a lower LR.

```yaml
learning_rate: 2e-3
```

Use this single learning rate for all three settings. Do not run a learning-rate sweep. Continue the LR schedule across every sequence-length change.

## Sequence length

Use a zero-based update index `u`. Select the row where `start_update <= u < end_update`. The start is included and the end is excluded. Keep the same length for all gradient accumulation steps in one update. Change length only between optimizer updates. Increase `u` after the optimizer update. Stop at `u == T`.

- `fixed_4096`, `fixed_2048`: one length for the whole run.
- `staged_20`: 20-stage discretization of the token-budget optimal schedule for this LR schedule (Section 4 of the paper, fixed LR and batch size). Lengths are between 1024 and 16384 and are multiples of 32. Source: `stage20_schedules.csv`, schedule `wsd`.

| Schedule ID | Stage | `start_update` | `end_update` | Updates | Sequence length |
|---|---:|---:|---:|---:|---:|
| `fixed_4096` | 1 | 0 | 19200 | 19200 | 4096 |
| `fixed_2048` | 1 | 0 | 38400 | 38400 | 2048 |
| `staged_20` | 1 | 0 | 656 | 656 | 1024 |
| `staged_20` | 2 | 656 | 875 | 219 | 1408 |
| `staged_20` | 3 | 875 | 3164 | 2289 | 1792 |
| `staged_20` | 4 | 3164 | 5151 | 1987 | 1952 |
| `staged_20` | 5 | 5151 | 7034 | 1883 | 2112 |
| `staged_20` | 6 | 7034 | 8597 | 1563 | 2336 |
| `staged_20` | 7 | 8597 | 9948 | 1351 | 2560 |
| `staged_20` | 8 | 9948 | 11156 | 1208 | 2816 |
| `staged_20` | 9 | 11156 | 12109 | 953 | 3104 |
| `staged_20` | 10 | 12109 | 12962 | 853 | 3424 |
| `staged_20` | 11 | 12962 | 13687 | 725 | 3808 |
| `staged_20` | 12 | 13687 | 14301 | 614 | 4224 |
| `staged_20` | 13 | 14301 | 14818 | 517 | 4704 |
| `staged_20` | 14 | 14818 | 15251 | 433 | 5248 |
| `staged_20` | 15 | 15251 | 16131 | 880 | 5920 |
| `staged_20` | 16 | 16131 | 16790 | 659 | 6400 |
| `staged_20` | 17 | 16790 | 17272 | 482 | 7008 |
| `staged_20` | 18 | 17272 | 17629 | 357 | 7744 |
| `staged_20` | 19 | 17629 | 17894 | 265 | 8640 |
| `staged_20` | 20 | 17894 | 19200 | 1306 | 16384 |

Use the exact integer boundaries on restart. Do not round lengths or rescale boundaries.

## Micro-batch size and throughput

**Turn activation checkpointing off** in every run. Set `activation_checkpoint.mode = "none"` (no activation recomputation in the backward pass; also called gradient checkpointing). Use neither full nor selective recomputation. The 1B model fits on H200 without it, and recomputation costs throughput.

Keep `D * m * A = 256`, where `D = 8` data-parallel replicas (`parallelism.data_parallel_replicate_degree = 8` in the reference run), `m` is sequences per micro-batch on each GPU, and `A` is gradient accumulation steps per optimizer update.

**Match the micro-batch sizes of the reference config**, not earlier `kickoff-smoke` or smoke-server measurements. This overrides the launch prompt's instruction to reuse earlier measurements. The three reference runs in the project used `microbatch_scheduler` values of 32 at 1056; 16 at 1792 to 3232; 8 at 3648 to 6656; and 2 at 16384. The largest per-GPU micro-batch among them is 53,248 tokens. Start from the values below. A length used by a reference run keeps its reference value. For other lengths, the starting value is the largest `m` in 32, 16, 8, 4, 2, 1 whose tokens do not exceed 53,248.

| Sequence length | Used by | Starting `m` (sequences per GPU) | `A` (accumulation) | Tokens per GPU micro-batch | Source |
|---:|---|---:|---:|---:|---|
| 1024 | `staged_20` | 32 | 1 | 32,768 | derived (32 x 1024 = 32,768 <= 53,248 tokens) |
| 1408 | `staged_20` | 32 | 1 | 45,056 | derived (32 x 1408 = 45,056 <= 53,248 tokens) |
| 1792 | `staged_20` | 16 | 2 | 28,672 | reference run |
| 1952 | `staged_20` | 16 | 2 | 31,232 | derived (16 x 1952 = 31,232 <= 53,248 tokens) |
| 2048 | `fixed_2048` | 16 | 2 | 32,768 | derived (16 x 2048 = 32,768 <= 53,248 tokens) |
| 2112 | `staged_20` | 16 | 2 | 33,792 | reference run |
| 2336 | `staged_20` | 16 | 2 | 37,376 | derived (16 x 2336 = 37,376 <= 53,248 tokens) |
| 2560 | `staged_20` | 16 | 2 | 40,960 | derived (16 x 2560 = 40,960 <= 53,248 tokens) |
| 2816 | `staged_20` | 16 | 2 | 45,056 | derived (16 x 2816 = 45,056 <= 53,248 tokens) |
| 3104 | `staged_20` | 16 | 2 | 49,664 | derived (16 x 3104 = 49,664 <= 53,248 tokens) |
| 3424 | `staged_20` | 8 | 4 | 27,392 | derived (8 x 3424 = 27,392 <= 53,248 tokens) |
| 3808 | `staged_20` | 8 | 4 | 30,464 | derived (8 x 3808 = 30,464 <= 53,248 tokens) |
| 4096 | `fixed_4096` | 8 | 4 | 32,768 | derived (8 x 4096 = 32,768 <= 53,248 tokens) |
| 4224 | `staged_20` | 8 | 4 | 33,792 | derived (8 x 4224 = 33,792 <= 53,248 tokens) |
| 4704 | `staged_20` | 8 | 4 | 37,632 | derived (8 x 4704 = 37,632 <= 53,248 tokens) |
| 5248 | `staged_20` | 8 | 4 | 41,984 | derived (8 x 5248 = 41,984 <= 53,248 tokens) |
| 5920 | `staged_20` | 8 | 4 | 47,360 | derived (8 x 5920 = 47,360 <= 53,248 tokens) |
| 6400 | `staged_20` | 8 | 4 | 51,200 | derived (8 x 6400 = 51,200 <= 53,248 tokens) |
| 7008 | `staged_20` | 4 | 8 | 28,032 | derived (4 x 7008 = 28,032 <= 53,248 tokens) |
| 7744 | `staged_20` | 4 | 8 | 30,976 | derived (4 x 7744 = 30,976 <= 53,248 tokens) |
| 8640 | `staged_20` | 4 | 8 | 34,560 | derived (4 x 8640 = 34,560 <= 53,248 tokens) |
| 16384 | `staged_20` | 2 | 16 | 32,768 | reference run |

1. Test each length with activation checkpointing off and complete optimizer updates, including optimizer-state allocation. Measure peak memory and input tokens per second after startup.
2. If the starting `m` runs out of memory, halve it. If it leaves clear memory headroom, also test the next larger `m` (at most 32, because `D * m <= 256`). Select the fastest stable value. Keep headroom for normal variation.
3. Write the selected `m` and `A` for each length into the generated config (`microbatch_scheduler` with anchors at the stage boundaries, as in the reference run). Use the same value for the same length in every run.
4. Normalize gradients for the full global batch of 256 sequences. Do not apply the accumulation divisor twice. Do not change the schedule or the global batch size to fit memory; if nothing fits, report it.

**Performance comes from two settings only: activation checkpointing off and a carefully selected micro-batch for each length.** Keep every other runtime setting exactly as in the reference run, including `dataloader.num_workers`, `pin_memory`, and `prefetch_factor`, `compile.enable`, `optimizer.implementation`, `metrics.log_freq`, and the parallelism settings. Record the tokens per second and the estimated wall-clock time of each run, and report them to the user before starting formal runs.

## Stage changes and restart

Keep the model state, optimizer state, learning rate, and data position at a stage change. Save the completed update count, token count, model state, optimizer state, data position, and random-number state in each checkpoint. On restart, use the completed update count as the next index `u` and to restore the LR and length. Do not advance `u` after a skipped optimizer update. Record retries and their extra tokens separately.

## Run checks

Before training, check for every run:

- Intervals start at zero and end at `T`, with no gaps or overlaps.
- Every length is a positive multiple of 32, at least 1024, and at most 16384.
- Every update uses 256 sequences.
- The token total equals the budget:

```text
sum((end_update - start_update) * sequence_length * 256)
    == 20_132_659_200
```

Check the update and token counts during training and after restart.

## Evaluation and output

Use the same validation data, input lengths, and metrics for all runs (section "Dataset"). The model must support evaluation at length 16384. Use `validator.token_intervals = 50` and remove `validator.freq`: evaluate 51 times, once before training, at 49 intermediate token targets, and once after the final update. Evaluate after the first complete update that reaches each target and record the actual count. Do not change `T` to align evaluation points.

Produce three schedule configurations under `wsd5_75_20_lr-constant-bs`, with the selected micro-batch and accumulation values, and a launcher for the 3 runs. Record each run's input settings, batch configuration, update count, token count, validation results, and training time.

Prepare the configurations and hardware tests first. Start full training only when the user requests it.
