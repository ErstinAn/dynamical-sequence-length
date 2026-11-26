# llama3_1b-5stages-valaligned-wsd5_75_20_lr-constant-bs

Use the [experiment launch prompt](Nov25_dense_experiment_launch_prompt.md) with this specification after the target server passes smoke.

This specification repeats the dense **Llama 3 1B** experiment with the 5-stage sequence-length schedule, under the `wsd5_75_20_lr-constant-bs` learning-rate schedule. Model, data, token budget, global batch, micro-batches, and checkpoint rules are those of the dense reference runs. **The one change: the validation sequence length follows the training sequence-length schedule** (section "Evaluation and output"); the dense reference runs validated at the fixed length 16384.

It trains two sequence-length settings, `staged_5` and `fixed_4096`, at each learning rate in the grid below. **`fixed_16384` is not trained again**: its training length is always 16384, so its aligned validation is the same as in the finished dense runs. Those runs serve as the alignment check (section "Alignment check with `fixed_16384`").

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

Read it with `wandb.Api().run("dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/858175ff26e34e89b7ff5282a05b08ba").config`. It uses the same LR schedule as this experiment and the same 5-stage sequence-length schedule as `staged_5` here. Take the 1B Llama 3 model, initialization, dataset, tokenizer, validation data, optimizer (AdamW, betas, weight decay, gradient clipping), precision, process layout, seeds, and evaluation settings from it. Change only these settings:

- update count `steps`, sequence-length schedule, and LR warmup/decay lengths for each `T`;
- learning rate (grid below);
- micro-batch schedule for each length (section "Micro-batch size and throughput");
- activation checkpointing off (`activation_checkpoint.mode = "none"`);
- removal of `validator.freq` (validation by `validator.token_intervals = 50`, as in the launch prompt);
- validation sequence length: follows the training schedule instead of the fixed `validator.eval_seq_len = 16384`;
- W&B project, run name, and output folder.

Keep all other settings the same across runs. Before generating configs, list every field you changed relative to the reference run.

**The W&B reference run is the authority** for the model, initialization, dataset, tokenizer, validation data, optimizer, and precision. If the smoke config on the target server, the launch prompt, or any local default gives a different value, use the W&B value and report the difference. Take only the environment, job file, Slurm settings, shared storage path, and W&B credentials from the smoke record. The W&B config is a resolved record (it contains object representations such as `<function ...>` and sharding reprs); map each field to the repository's YAML format instead of copying it verbatim.

Use `llama3_1b-5stages-valaligned-wsd5_75_20_lr-constant-bs` as the experiment group name in config filenames and output paths.

## Dataset

**Use exactly the same training and validation data as the W&B reference run. This is the most important requirement of this specification.** Check every dataset field below against the W&B reference run config, not against the smoke config, a local file, an older config, or memory. This section overrides the data instructions of the launch prompt, including the smoke data and the public-dataset restore. Do not substitute another dataset, a re-tokenized copy, a different split, or a different training packing mode. Only the validation sequence length changes (below). The reference run uses:

| Item | Reference value (`config` field) |
|---|---|
| Training dataset | `dataloader.dataset = fineweb_edu_long`, `split = train`, `packing = stream` |
| Training data path | `dataloader.dataset_path = /checkpoint/optim/zhiqibu/data/seqlen_assets/data/fineweb_edu_240909_30b_v1` |
| Training manifest SHA-256 | `dataloader.expected_manifest_sha256 = 2ca374652bc4c4f0c8649d58672f30d17cac83a32d915d3d99c4c1e3a0a47da3` |
| Tokenizer | `hf_assets_path = /checkpoint/optim/zhiqibu/data/seqlen_assets/hf/llama3-tokenizer` |
| Validation data | `validator.artifact_path = /checkpoint/optim/zhiqibu/data/seqlen_assets/data/fineweb_edu_240909_val_v1/validation_packed_16384` |
| Validation manifest SHA-256 | `validator.expected_manifest_sha256 = 8652035304766d8edc00943b899d04a90ac0fb85d2322f807d4d24141950e831` |
| Validation settings | `validator.token_budget = 15320140`, `validator.token_intervals = 50`, `validator.eval_local_batch_size = 1`. The reference value `validator.eval_seq_len = 16384` is replaced by the training-aligned length (section "Evaluation and output") |

`validator.dataloader.dataset` shows `c4_validation`, but the validation data come from `validator.artifact_path`. Verify which one the code actually reads.

The validation data stay the same FineWeb-Edu validation set (`fineweb_edu_240909_val_v1`); only the length of the evaluated sequences changes. The reference artifact is packed at 16384. Find out how the code builds validation sequences of another length from this set (for example by cutting the same token stream into sequences of the training length, or from another packing in `fineweb_edu_240909_val_v1`). Use the same validation tokens at every length as far as the code allows. If another artifact is needed, report its path and computed manifest SHA-256 and get the user's confirmation; do not use a validation set from another source.

**Confirm the dataset with the user before the micro-batch tests and before any formal run.** Show the user the resolved training path, validation path, both manifest hashes (expected from W&B and computed on the server), the tokenizer path, and the first few token IDs of one training batch. Do not start any run until the user explicitly confirms. If a path is missing on this server, a hash differs, or any field is unclear or cannot be matched to the W&B config, stop and ask the user. Do not use a fallback dataset, and do not restore or rebuild the public dataset (`Erstin/fineweb-edu-long-30b`) unless the user asks for it; a rebuilt copy must still match both manifest hashes above.

## Model initialization and model checkpoints

**Match the model initialization and model-checkpoint settings of the W&B reference run exactly.** All reference runs in the project use identical values:

| Item | Reference value (`config` field) |
|---|---|
| Model | `model_spec.name = llama3`, `model_spec.flavor = 1B` (all `model_spec.model` fields unchanged) |
| Initialization | `checkpoint.initial_load_path = null`: train from random initialization, no pretrained or seed checkpoint (`create_seed_checkpoint = false`) |
| Seed and determinism | `debug.seed = 42`, `debug.deterministic = true` |
| Checkpoint saving | `checkpoint.enable = true`, `folder = checkpoint`, `export_dtype = float32`, `last_save_model_only = false`, `last_save_in_hf = false`; schedule: 10 token-based checkpoints (below) |
| Checkpoint loading | `load_step = -1`, `exclude_from_loading = []`, `initial_load_model_only = false` |

**Save 10 complete checkpoints at 10% increments of the token budget** (every 2,013,265,920 nominal tokens), including the final state, and keep all ten. Save each one after the first complete optimizer update whose cumulative tokens reach the target. Each checkpoint contains the model, optimizer state, data position, random-number state, completed update count, and token count. The table gives the completed update count at each checkpoint; it is the same at every learning rate:

| Schedule ID | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `staged_5` | 4043 | 7767 | 10728 | 13161 | 14989 | 16269 | 17549 | 18240 | 18720 | 19200 |
| `fixed_4096` | 1920 | 3840 | 5760 | 7680 | 9600 | 11520 | 13440 | 15360 | 17280 | 19200 |

Do not change the initialization or checkpoint settings on your own. If a field is unclear in the code or the saved checkpoints differ from the table, ask the user. Record the actual update and token count of every checkpoint in the run report.

## Default hardware

Use the same allocation for smoke and each formal run. Change it only at the user's request.

```yaml
num_nodes: 1
gpus_per_node: 8
num_gpus: 8
gpu_type: H200
```

Get all other training settings from the reference config.

## Submission settings

| Field | Value | Use |
|---|---|---|
| `max_concurrent_runs` | `null` | Optional limit across this batch. Omitted or null means no user limit. |
| `wandb_entity` | **dynamic_seq_length** | Confirmed W&B entity for the formal runs. |
| `wandb_project` | **Llama3-1B-20B_Fineweb-edu_val-sl-aligned** | Formal W&B project for the validation-aligned runs (not the reference project). |
| `wandb_run_name` | **{sl}-wsd_lr-cons_bs-lr_{x}** | Replace `{sl}` with the schedule ID and `{x}` with the LR as written in `lr_grid`. |

To limit the batch, replace `null` with a positive integer. For example, `2` allows two runs to train at once.
Pending jobs do not count toward this limit. Slurm site limits still apply.

## Run settings

Every run uses a global batch size of **256 sequences per optimizer update** and exactly **20,132,659,200 nominal tokens**: count `256 * sequence_length` per optimizer update, which is W&B `n_tokens_seen` (the reference run ends at `n_tokens_seen = 20,132,659,200`). Record the valid-token count (`n_valid_tokens_seen`) separately; do not add updates to make it reach the budget. Use nominal tokens for the checkpoint and validation targets. Each update uses `256 * sequence_length` tokens, so the update count `T` differs between settings.

| Schedule ID | Optimizer updates (`T`) | Warmup updates | Stable updates | Decay updates | First decay update | Stages | Mean sequence length |
|---|---:|---:|---:|---:|---:|---:|---:|
| `staged_5` | 19200 | 960 | 14400 | 3840 | 15360 | 5 | 4096 |
| `fixed_4096` | 19200 | 960 | 14400 | 3840 | 15360 | 1 | 4096 |

`T` is the total number of optimizer updates. It does not count gradient accumulation steps.

## Learning rate

Use a Warmup-Stable-Decay schedule: 5% linear warmup to the **peak learning rate**, 75% at the peak, and 20% linear decay to zero. Compute the phase lengths from each run's own `T` (table above). At the zero-based index in the column "First decay update", the run still uses the peak LR; the next update uses a lower LR.

```yaml
lr_grid: [1e-3, 2e-3]
```

Run both sequence-length settings at every learning rate in `lr_grid`, with the same LR schedule. Use **seed 42** for every run. This gives **4 runs** in this specification. Set `optimizer.learning_rate` to the grid value. Do not add or remove learning rates. Continue the LR schedule across every sequence-length change.

Training is deterministic (`debug.deterministic = true`) and only the validation changes, so each new run should repeat the training of the finished dense run with the same name in the reference project, where one exists:

| Schedule ID | LR | W&B run name | Finished dense run with the same training (validated at 16384) |
|---|---|---|---|
| `staged_5` | 1e-3 | `staged_5-wsd_lr-cons_bs-lr_1e-3` | [`79b286be`](https://wandb.ai/dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/runs/79b286be4f2547a8ac1b9b74a73d6d94) (finished) |
| `staged_5` | 2e-3 | `staged_5-wsd_lr-cons_bs-lr_2e-3` | [`858175ff`](https://wandb.ai/dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/runs/858175ff26e34e89b7ff5282a05b08ba) (finished) |
| `fixed_4096` | 1e-3 | `fixed_4096-wsd_lr-cons_bs-lr_1e-3` | none |
| `fixed_4096` | 2e-3 | `fixed_4096-wsd_lr-cons_bs-lr_2e-3` | [`e7a8b6a2`](https://wandb.ai/dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/runs/e7a8b6a27818461ead9943bb953af118) (finished) |

## Sequence length

Use a zero-based update index `u`. Select the row where `start_update <= u < end_update`. The start is included and the end is excluded. Keep the same length for all gradient accumulation steps in one update. Change length only between optimizer updates. Increase `u` after the optimizer update. Stop at `u == T`.

- `staged_5`: the 5-stage schedule of the dense reference run `staged_5-wsd_lr-cons_bs-lr_2e-3` (1056 -> 2112 -> 3232 -> 6144 -> 16384); the dense `staged_5` runs use it at every LR. Copy it exactly.
- `fixed_4096`: one length for the whole run.

| Schedule ID | Stage | `start_update` | `end_update` | Updates | Sequence length |
|---|---:|---:|---:|---:|---:|
| `staged_5` | 1 | 0 | 638 | 638 | 1056 |
| `staged_5` | 2 | 638 | 9290 | 8652 | 2112 |
| `staged_5` | 3 | 9290 | 14316 | 5026 | 3232 |
| `staged_5` | 4 | 14316 | 17887 | 3571 | 6144 |
| `staged_5` | 5 | 17887 | 19200 | 1313 | 16384 |
| `fixed_4096` | 1 | 0 | 19200 | 19200 | 4096 |

Use the exact integer boundaries on restart. Do not round lengths or rescale boundaries.

## Micro-batch size and throughput

**Turn activation checkpointing off** in every run. Set `activation_checkpoint.mode = "none"`. The 1B model fits on H200 without it, and recomputation costs throughput.

Keep `D * m * A = 256`, where `D = 8` data-parallel replicas (`parallelism.data_parallel_replicate_degree = 8` in the reference run), `m` is sequences per micro-batch on each GPU, and `A` is gradient accumulation steps per optimizer update.

**Use the micro-batch sizes of the dense reference runs**, not earlier `kickoff-smoke` or smoke-server measurements. Every length here was used by a finished dense run: the `staged_5` lengths by the reference runs (`microbatch_scheduler`), and 4096 by the `fixed_4096` runs (`batch_size = 8`).

| Sequence length | Used by | `m` (sequences per GPU) | `A` (accumulation) | Tokens per GPU micro-batch |
|---|---|---:|---:|---:|
| 1056 | `staged_5` | 32 | 1 | 33,792 |
| 2112 | `staged_5` | 16 | 2 | 33,792 |
| 3232 | `staged_5` | 16 | 2 | 51,712 |
| 4096 | `fixed_4096` | 8 | 4 | 32,768 |
| 6144 | `staged_5` | 8 | 4 | 49,152 |
| 16384 | `staged_5` | 2 | 16 | 32,768 |

1. Write `m` and `A` for each length into the generated config (`microbatch_scheduler` with anchors at the stage boundaries, as in the reference run). Use the same value for the same length in every run.
2. Test each length once with complete optimizer updates, including optimizer-state allocation and **one validation pass at the same length**, because validation now runs at the training length. Measure peak memory and input tokens per second after startup. If a value runs out of memory, report it before changing it.
3. Normalize gradients for the full global batch of 256 sequences. Do not apply the accumulation divisor twice. Do not change the schedule or the global batch size to fit memory.

Keep every other runtime setting exactly as in the reference run, including `dataloader.num_workers`, `pin_memory`, `prefetch_factor`, `compile.enable`, `optimizer.implementation`, `metrics.log_freq`, and the parallelism settings. Record the tokens per second, the validation time at each length, and the estimated wall-clock time of each run, and report them to the user before starting formal runs.

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

- The LR schedule matches the section "Learning rate" for the run's own `T` and grid value.
- The validation length at every one of the 51 validations matches the plan in section "Evaluation and output".

Check the update and token counts during training and after restart. After launch, compare the training loss of the first validation interval with the finished dense run of the same name (table in section "Learning rate"), where one exists, and report any difference.

## Evaluation and output

Use the same validation data, token budget, and metrics for all runs (section "Dataset"). Use `validator.token_intervals = 50` and remove `validator.freq`: evaluate 51 times, once before training, at 49 intermediate token targets, and once after the final update. Evaluate after the first complete update that reaches each target and record the actual count. Do not change `T` to align evaluation points.

**The validation sequence length follows the training sequence-length schedule.** This differs from the dense reference runs, which validated at the fixed length 16384.

- A validation after `c` completed updates uses the sequence length of the last completed update, update index `c - 1`. The validation before training (`c = 0`) uses the length of update 0.
- `fixed_4096` always validates at 4096. `staged_5` validates at the length of the stage that contains its last update. A validation at a stage boundary uses the length of the stage that just ended.
- Keep `validator.token_budget = 15320140` tokens per validation at every length. The dense runs evaluate `ceil(15320140 / 16384) = 936` sequences at 16384, the last one partial. Record the length, the number of sequences, and the evaluated tokens of every validation (W&B `validation_metrics/num_sequences` and `validation_metrics/evaluated_tokens`).
- Before any micro-batch test or formal run, show the user the validation length plan for each run, the config fields and code path that set it, and how the validation tokens are cut at each length, as the launch prompt requires. If the code cannot follow the training length, report the gap and obtain permission before changing shared code.

Planned validation lengths of `staged_5` (index 0 is the validation before training, index 50 the one after the final update):

| Validation indices | Completed updates `c` | Validation length | Sequences per validation (`ceil(15320140 / L)`) |
|---|---|---:|---:|
| 0 | 0 | 1056 | 14508 |
| 1-12 | 1064-9256 | 2112 | 7254 |
| 13-22 | 9755-14135 | 3232 | 4741 |
| 23-36 | 14477-17805 | 6144 | 2494 |
| 37-50 | 17952-19200 | 16384 | 936 |

Validation losses at different lengths are not directly comparable with each other. Report every value with its length.

### Alignment check with `fixed_16384`

Do not train `fixed_16384` again. Under the aligned rule its validation length is always 16384, so its validation curve is the one of the finished dense runs:

| LR | Finished dense `fixed_16384` run | State |
|---|---|---|
| 1e-3 | `fixed_16384-wsd_lr-cons_bs-lr_1e-3` [`a7603b0d`](https://wandb.ai/dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/runs/a7603b0d3c7841e2bff368db25ffacd4) | finished |
| 2e-3 | `fixed_16384-wsd_lr-cons_bs-lr_2e-3` [`c934d813`](https://wandb.ai/dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/runs/c934d81382db45ffb6521152ac1b9378) | finished |

Use these runs to check the aligned validation code before any formal run:

1. Generate the `fixed_16384` config of this specification at LR `1e-3` (`T = 4800`, one stage at 16384, `m = 2`, `A = 16`) with the new validation code.
2. Train it for the first 96 updates, up to and including the first intermediate validation target (402,653,184 tokens). Do not log this check to the formal project `Llama3-1B-20B_Fineweb-edu_val-sl-aligned`, and do not keep it as a formal run.
3. Compare with `fixed_16384-wsd_lr-cons_bs-lr_1e-3` ([`a7603b0d`](https://wandb.ai/dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/runs/a7603b0d3c7841e2bff368db25ffacd4)): the validation before training (`validation_metrics/index = 0`), the first intermediate validation (`index = 1`, `completed_updates = 96`), the validation length, `num_sequences`, `evaluated_tokens`, and the training loss of each of the 96 updates. The reference run reports `validation_metrics/loss` = 12.300577 at index 0 and 6.229566 at index 1, with `num_sequences = 936` and `evaluated_tokens = 15320140` (the last sequence is partial).
4. Show the comparison to the user. The new code must give the same validation length, sequence count, and evaluated tokens, and the same losses up to numerical noise. If anything differs, stop and report it before the formal runs.

Produce two schedule configurations for each learning rate under `llama3_1b-5stages-valaligned-wsd5_75_20_lr-constant-bs`, with the selected micro-batch and accumulation values, and a launcher for the 4 runs. Record each run's input settings, batch configuration, update count, token count, validation results (with length), and training time.

Prepare the configurations and hardware tests first. Start full training only when the user requests it.
