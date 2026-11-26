# qwen3_a1b-5stages-constant_lr-constant-bs

Use the [experiment launch prompt](finewebedu_experiment_launch_prompt.md) with this specification after the target server passes smoke.

This specification trains a **Qwen3-style MoE model with about 1B active parameters** (`QWEN3-A1B`) in the same setting as the dense 20-stage experiments: same data, token budget, global batch, and checkpoint rules. **One difference from the dense runs: the validation sequence length follows the training sequence-length schedule** (section "Evaluation and output"); the dense runs validated at a fixed length of 16384. It compares three sequence-length settings, `staged_5`, `fixed_16384`, and `fixed_4096`, under the `constant_lr-constant-bs` learning-rate schedule, at each learning rate in the grid below.

## Model config (supplied at launch)

The collaborator provides the MoE model config. The user gives its path in the launch request.

```yaml
model_config: <path given in the launch request>
```

**The supplied model config is the authority for the model architecture**: model name and flavor, hidden sizes, layer count, attention, number of experts, experts per token (top-k), shared experts, router settings, load-balancing loss and its coefficient, capacity factor or token dropping, RoPE settings, and initialization settings. Do not take the model from the W&B reference run below; that run is a dense Llama 3 1B model.

- If the launch request gives no model config path, ask for it in the one message of missing inputs and do not generate configs until you have it.
- Record the path, the SHA-256 of the file, the total and active parameter counts, and the parallelism settings it requires (for example expert parallelism). Report them to the user with the dataset confirmation.
- The data are pre-tokenized with the Llama 3 tokenizer (section "Dataset"). Keep that tokenizer and data. If the model config sets a different tokenizer, or a `vocab_size` smaller than the Llama 3 tokenizer vocabulary, stop and ask the user. Do not re-tokenize the data.
- The model must support training and evaluation at length 16384. If the config sets a smaller maximum length or RoPE limit, stop and ask the user.
- If the repository cannot build the model from this config, report the gap. Obtain permission before changing shared code.

## Reference config

Use this finished dense W&B run for everything except the model:

```yaml
reference_run:
  entity: dynamic_seq_length
  project: Llama3-1B-20B_Fineweb-edu-240909-val240909
  run_name: staged_5-cons_lr-cons_bs-lr_5e-4
  run_id: 9f5a4eedb9e047d2be6ef569b28750f2
  url: https://wandb.ai/dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/runs/9f5a4eedb9e047d2be6ef569b28750f2
```

Read it with `wandb.Api().run("dynamic_seq_length/Llama3-1B-20B_Fineweb-edu-240909-val240909/9f5a4eedb9e047d2be6ef569b28750f2").config`. It uses the same LR schedule as this experiment and the same 5-stage sequence-length schedule as `staged_5` here. Take the dataset, tokenizer, validation data and settings (except the validation sequence length), optimizer (AdamW, betas, weight decay, gradient clipping), precision, seed, data order, checkpoint settings, and runtime settings from it. Change only these settings:

- the model (from the supplied MoE config) and the parallelism it requires;
- update count `steps`, sequence-length schedule, and LR warmup/decay lengths for each `T`;
- learning rate (grid below);
- micro-batch schedule, and the memory settings of section "Micro-batch size, memory, and throughput";
- removal of `validator.freq` (validation by `validator.token_intervals = 50`, as in the launch prompt);
- validation sequence length: follows the training schedule instead of the fixed `validator.eval_seq_len = 16384`;
- W&B project, run name, and output folder.

Before generating configs, list every field you changed relative to the reference run.

**The W&B reference run is the authority** for the dataset, tokenizer, validation data, optimizer, and precision. If the smoke config on the target server, the launch prompt, the model config, or any local default gives a different value for these, use the W&B value and report the difference. Take only the environment, job file, Slurm settings, shared storage path, and W&B credentials from the smoke record. The W&B config is a resolved record (it contains object representations such as `<function ...>` and sharding reprs); map each field to the repository's YAML format instead of copying it verbatim.

Use `qwen3_a1b-5stages-constant_lr-constant-bs` as the experiment group name in config filenames and output paths.

## Dataset

**Use exactly the same training and validation data as the W&B reference run. This is the most important requirement of this specification.** Check every dataset field below against the W&B reference run config, not against the smoke config, the model config, a local file, an older config, or memory. This section overrides the data instructions of the launch prompt, including the smoke data and the public-dataset restore. Do not substitute another dataset, a re-tokenized copy, a different split, or a different training packing mode. Only the validation sequence length changes (below). The reference run uses:

| Item | Reference value (`config` field) |
|---|---|
| Training dataset | `dataloader.dataset = fineweb_edu_long`, `split = train`, `packing = stream` |
| Training data path | `dataloader.dataset_path = /checkpoint/optim/zhiqibu/data/seqlen_assets/data/fineweb_edu_240909_30b_v1` |
| Training manifest SHA-256 | `dataloader.expected_manifest_sha256 = 2ca374652bc4c4f0c8649d58672f30d17cac83a32d915d3d99c4c1e3a0a47da3` |
| Tokenizer | `hf_assets_path = /checkpoint/optim/zhiqibu/data/seqlen_assets/hf/llama3-tokenizer` |
| Validation data | `validator.artifact_path = /checkpoint/optim/zhiqibu/data/seqlen_assets/data/fineweb_edu_240909_val_v1/validation_packed_16384` |
| Validation manifest SHA-256 | `validator.expected_manifest_sha256 = 8652035304766d8edc00943b899d04a90ac0fb85d2322f807d4d24141950e831` |
| Validation settings | `validator.token_budget = 15320140`, `validator.token_intervals = 50`, `validator.eval_local_batch_size = 1`. The dense value `validator.eval_seq_len = 16384` is replaced by the training-aligned length (section "Evaluation and output") |

`validator.dataloader.dataset` shows `c4_validation`, but the validation data come from `validator.artifact_path`. Verify which one the code actually reads.

The validation data stay the same FineWeb-Edu validation set (`fineweb_edu_240909_val_v1`); only the length of the evaluated sequences changes. The reference artifact is packed at 16384. Find out how the code builds validation sequences of another length from this set (for example by cutting the same token stream into sequences of the training length, or from another packing in `fineweb_edu_240909_val_v1`). Use the same validation tokens at every length as far as the code allows. If another artifact is needed, report its path and computed manifest SHA-256 and get the user's confirmation; do not use a validation set from another source.

**Confirm the dataset and the model config with the user before the micro-batch tests and before any formal run.** Show the user the resolved training path, validation path, both manifest hashes (expected from W&B and computed on the server), the tokenizer path, the first few token IDs of one training batch, and the model-config record from section "Model config". Do not start any run until the user explicitly confirms. If a path is missing on this server, a hash differs, or any field is unclear or cannot be matched to the W&B config, stop and ask the user. Do not use a fallback dataset, and do not restore or rebuild the public dataset (`Erstin/fineweb-edu-long-30b`) unless the user asks for it; a rebuilt copy must still match both manifest hashes above.

## Model initialization and model checkpoints

| Item | Value |
|---|---|
| Initialization | Random initialization with the initializer of the supplied model config. `checkpoint.initial_load_path = null`, no pretrained or seed checkpoint (`create_seed_checkpoint = false`) |
| Seed and determinism | `debug.seed = 42`, `debug.deterministic = true` (from the reference run) |
| Checkpoint saving | `checkpoint.enable = true`, `folder = checkpoint`, `export_dtype = float32`, `last_save_model_only = false`, `last_save_in_hf = false`; schedule: 10 token-based checkpoints (below) |
| Checkpoint loading | `load_step = -1`, `exclude_from_loading = []`, `initial_load_model_only = false` |

All runs start from the same initial weights (same model config and seed 42). If an MoE kernel (for example grouped GEMM or token scatter) does not support `debug.deterministic = true`, stop and ask the user; do not turn determinism off on your own.

**Save 10 complete checkpoints at 10% increments of the token budget** (every 2,013,265,920 nominal tokens), including the final state, and keep all ten. Save each one after the first complete optimizer update whose cumulative tokens reach the target. Each checkpoint contains the model (including router and expert weights), optimizer state, data position, random-number state, completed update count, and token count. The table gives the completed update count at each checkpoint; it is the same at every learning rate:

| Schedule ID | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `staged_5` | 3322 | 6643 | 9489 | 12104 | 14216 | 16063 | 17326 | 18240 | 18720 | 19200 |
| `fixed_16384` | 480 | 960 | 1440 | 1920 | 2400 | 2880 | 3360 | 3840 | 4320 | 4800 |
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

## Submission settings

| Field | Value | Use |
|---|---|---|
| `max_concurrent_runs` | `null` | Optional limit across this batch. Omitted or null means no user limit. |
| `wandb_entity` | **dynamic_seq_length** | Confirmed W&B entity for the formal runs. |
| `wandb_project` | **QWEN3-A1B-20B_Fineweb-edu_val-sl-aligned** | Formal W&B project for the MoE runs (not the dense reference project). |
| `wandb_run_name` | **{sl}-cons_lr-cons_bs-lr_{x}** | Replace `{sl}` with the schedule ID and `{x}` with the LR as written in `lr_grid`. |

To limit the batch, replace `null` with a positive integer. For example, `2` allows two runs to train at once.
Pending jobs do not count toward this limit. Slurm site limits still apply.

## Run settings

Every run uses a global batch size of **256 sequences per optimizer update** and exactly **20,132,659,200 nominal tokens**: count `256 * sequence_length` per optimizer update, which is W&B `n_tokens_seen` (the dense reference run ends at `n_tokens_seen = 20,132,659,200`). Record the valid-token count (`n_valid_tokens_seen`) separately; do not add updates to make it reach the budget. Use nominal tokens for the checkpoint and validation targets. Each update uses `256 * sequence_length` tokens, so the update count `T` differs between settings. The token budget does not depend on the parameter count; the collaborator keeps the active parameters at about 1B.

| Schedule ID | Optimizer updates (`T`) | Stages | Mean sequence length |
|---|---:|---:|---:|
| `staged_5` | 19200 | 5 | 4096 |
| `fixed_16384` | 4800 | 1 | 16384 |
| `fixed_4096` | 19200 | 1 | 4096 |

`T` is the total number of optimizer updates. It does not count gradient accumulation steps.

## Learning rate

Keep the learning rate constant within each run. Do not use warmup or decay.

```yaml
lr_grid: [2.5e-4, 5e-4, 1e-3]
```

Run all three sequence-length settings at every learning rate in `lr_grid`, with the same LR schedule. Use **seed 42** for every run. This gives **9 runs** in this specification. Set `optimizer.learning_rate` to the grid value. Do not add or remove learning rates, except under the single-LR option of the launch prompt (no sweep: constant `5e-4`, or peak `2e-3` for both decay schedules; then this specification has three runs). Continue the LR schedule across every sequence-length change.

| Schedule ID | LR | W&B run name |
|---|---|---|
| `staged_5` | 2.5e-4 | `staged_5-cons_lr-cons_bs-lr_2.5e-4` |
| `staged_5` | 5e-4 | `staged_5-cons_lr-cons_bs-lr_5e-4` |
| `staged_5` | 1e-3 | `staged_5-cons_lr-cons_bs-lr_1e-3` |
| `fixed_16384` | 2.5e-4 | `fixed_16384-cons_lr-cons_bs-lr_2.5e-4` |
| `fixed_16384` | 5e-4 | `fixed_16384-cons_lr-cons_bs-lr_5e-4` |
| `fixed_16384` | 1e-3 | `fixed_16384-cons_lr-cons_bs-lr_1e-3` |
| `fixed_4096` | 2.5e-4 | `fixed_4096-cons_lr-cons_bs-lr_2.5e-4` |
| `fixed_4096` | 5e-4 | `fixed_4096-cons_lr-cons_bs-lr_5e-4` |
| `fixed_4096` | 1e-3 | `fixed_4096-cons_lr-cons_bs-lr_1e-3` |

## Sequence length

Use a zero-based update index `u`. Select the row where `start_update <= u < end_update`. The start is included and the end is excluded. Keep the same length for all gradient accumulation steps in one update. Change length only between optimizer updates. Increase `u` after the optimizer update. Stop at `u == T`.

- `staged_5`: the 5-stage schedule of the dense reference run `staged_5-cons_lr-cons_bs-lr_5e-4` (2368 -> 3008 -> 4256 -> 6656 -> 16384). Copy it exactly.
- `fixed_16384`, `fixed_4096`: one length for the whole run.

| Schedule ID | Stage | `start_update` | `end_update` | Updates | Sequence length |
|---|---:|---:|---:|---:|---:|
| `staged_5` | 1 | 0 | 7734 | 7734 | 2368 |
| `staged_5` | 2 | 7734 | 13003 | 5269 | 3008 |
| `staged_5` | 3 | 13003 | 16289 | 3286 | 4256 |
| `staged_5` | 4 | 16289 | 18057 | 1768 | 6656 |
| `staged_5` | 5 | 18057 | 19200 | 1143 | 16384 |
| `fixed_16384` | 1 | 0 | 4800 | 4800 | 16384 |
| `fixed_4096` | 1 | 0 | 19200 | 19200 | 4096 |

Use the exact integer boundaries on restart. Do not round lengths or rescale boundaries.

## Micro-batch size, memory, and throughput

Use pure data parallelism over the 8 GPUs, with no tensor, pipeline, or context parallelism, so there are `D = 8` data-parallel ranks. Keep `D * m * A = 256`, where `m` is sequences per micro-batch on each GPU and `A` is gradient accumulation steps per optimizer update. Expert parallelism inside the 8 data-parallel ranks, if the model config requires it, does not change `D`.

The dense micro-batch values below are only starting candidates. The MoE model has different memory and throughput, so **measure every length on this hardware**. Do not reuse dense or earlier `kickoff-smoke` measurements.

| Sequence length | Used by | Starting `m` (sequences per GPU) | `A` (accumulation) | Tokens per GPU micro-batch | Source of the starting value |
|---|---|---:|---:|---:|---:|
| 2368 | `staged_5` | 16 | 2 | 37,888 | dense reference run |
| 3008 | `staged_5` | 16 | 2 | 48,128 | dense reference run |
| 4096 | `fixed_4096` | 8 | 4 | 32,768 | derived (8 x 4096 = 32,768 <= 53,248 tokens) |
| 4256 | `staged_5` | 8 | 4 | 34,048 | dense reference run |
| 6656 | `staged_5` | 8 | 4 | 53,248 | dense reference run |
| 16384 | `staged_5`, `fixed_16384` | 2 | 16 | 32,768 | dense reference run |

Find the memory setting first, then the micro-batch:

1. **Default setting.** Start from the reference runtime settings with activation checkpointing off (`activation_checkpoint.mode = "none"`) and the parallelism that the model config requires. If the code needs FSDP for the expert parallelism of the model config, use FSDP from the start.
2. **Micro-batch search.** Test each length with complete optimizer updates, including optimizer-state allocation. Measure peak memory and input tokens per second after startup. Include one validation pass at the same length, because validation now runs at the training length. If the starting `m` runs out of memory, halve it. If it leaves clear memory headroom, also test the next larger `m` (at most 32, because `D * m <= 256`). Select the fastest stable value and keep headroom for normal variation.
3. **Memory fallback.** If the model states do not fit, or a length does not fit even at `m = 1`, the user allows two memory-saving settings: FSDP (`parallelism.data_parallel_shard_degree = 8`, `data_parallel_replicate_degree = 1`) and activation checkpointing. Test, in this order, FSDP without activation checkpointing, FSDP with selective activation checkpointing, and FSDP with full activation checkpointing, and also activation checkpointing without FSDP if the model states fit without it. Choose the fastest setting in which every length of the run fits, then repeat step 2 for every length under that setting.
4. **One setting per run.** The parallelism and activation-checkpointing settings cannot change inside a run, so choose them for the whole run from the lengths the run uses. Use the same settings for a schedule ID at every learning rate, and the same `m` for the same length and settings in every run.
5. Write the selected memory settings, `m`, and `A` into each generated config (`microbatch_scheduler` with anchors at the stage boundaries, as in the reference run).
6. Normalize gradients for the full global batch of 256 sequences. Do not apply the accumulation divisor twice. Include the load-balancing loss as the model config defines it, and scale it across accumulation steps in the same way as the cross-entropy. Do not change the schedule or the global batch size to fit memory; if nothing fits, report it.

**Performance comes only from the memory setting and a carefully selected micro-batch for each length.** Keep every other runtime setting as in the reference run, including `dataloader.num_workers`, `pin_memory`, `prefetch_factor`, `compile.enable`, `optimizer.implementation`, and `metrics.log_freq`. Record the tested values, peak memory, tokens per second, and the estimated wall-clock time of each run, and report them to the user before starting formal runs.

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
- The validation length at every one of the 51 validations matches the rule in section "Evaluation and output".

Check the update and token counts during training and after restart.

## Evaluation and output

Use the same validation data, token budget, and metrics for all runs (section "Dataset"). Use `validator.token_intervals = 50` and remove `validator.freq`: evaluate 51 times, once before training, at 49 intermediate token targets, and once after the final update. Evaluate after the first complete update that reaches each target and record the actual count. Do not change `T` to align evaluation points.

**The validation sequence length follows the training sequence-length schedule.** This differs from the dense reference runs, which validated at the fixed length 16384.

- A validation after `c` completed updates uses the sequence length of the last completed update, update index `c - 1`. The validation before training (`c = 0`) uses the length of update 0.
- So `fixed_4096` always validates at 4096 and `fixed_16384` at 16384. `staged_5` validates at the length of the stage that contains its last update, starting at 2368 and ending at 16384. A validation at a stage boundary uses the length of the stage that just ended.
- Keep `validator.token_budget = 15320140` tokens per validation at every length. Record the length, the number of sequences, and the evaluated tokens of every validation.
- Before any micro-batch test or formal run, show the user the validation length plan for each run (initial, each stage, final), the config fields and code path that set it, and how the validation tokens are cut at each length, as the launch prompt requires. If the code cannot follow the training length, report the gap and obtain permission before changing shared code.

Validation losses at different lengths are not directly comparable with each other, or with the dense runs evaluated at 16384. Report every value with its length.

Report the validation loss as the language-model cross-entropy only, without the load-balancing loss. Log the training cross-entropy and the load-balancing loss separately. Also log the expert load statistics and the dropped-token fraction that the model code provides.

Produce three schedule configurations for each learning rate under `qwen3_a1b-5stages-constant_lr-constant-bs`, with the selected memory settings, micro-batch, and accumulation values, and a launcher for the 9 runs. Record each run's input settings, model-config hash, batch configuration, update count, token count, validation results, and training time.

Prepare the configurations and hardware tests first. Start full training only when the user requests it.
