---
name: launch-sequence-experiment
description: Generate local training configurations from a sequence schedule specification. Launch the agreed runs on a server that passed the repository smoke test.
---

# Launch a sequence length experiment

Give the agent one experiment file. For example, send this message from the repository root.

```text
Use @docs/sl_schedule_docs/Nov_26_sl_schedule_dense_val_aligned/5stages-dense-constant_lr-constant-bs.md and its linked launch prompt.
Prepare the configurations. Launch the runs that we agreed to submit.
```

Use the named file as the experiment specification. For a new experiment, replace that one path.
Keep shared experiment changes in Markdown under `docs/sl_schedule_docs/Nov_26_sl_schedule_dense_val_aligned/`, next to this prompt.
Generate machine-specific YAML and job files under the ignored directory `local/generated_configs/<experiment>/`.
If the existing code cannot express a requirement, identify the gap and its effect.
Obtain explicit permission before changing shared code.

## 1. Read the specification and the smoke record

Use the checkout and environment from the successful smoke on this server. Take only the software environment, job file, submission command, Slurm settings, shared storage path, and W&B credentials from the smoke; the smoke data and tokenizer play no role here.

**The W&B reference run of the specification is the authority for the model, initialization, training data, tokenizer, validation data, optimizer, and precision.** The specification changes only the validation sequence length (section 2) and the settings it lists.
Read its config with the W&B API. Use its dataset paths and require that both manifest SHA-256 values match it.
If the smoke config uses other data or another tokenizer, ignore them and report the difference.
If the reference data are missing on this server or a hash differs, stop and ask the user.
Do not restore or rebuild the public dataset unless the user asks for it.
Confirm the resolved data with the user as the specification requires before measuring micro-batches or submitting runs.

Every specification for this prompt names a W&B reference run. Never take the training data, validation data, or tokenizer from the smoke record or a smoke config. If a specification names no W&B reference run, stop and ask the user for the data source.

Use one node with eight NVIDIA H200 GPUs for each run by default.
Use a different layout only when the user requests it. Check the smoke record for the requested layout.
For the default, verify one hostname and eight ranks in one training group.
If the matching smoke record is missing, follow the [target-server smoke guide](../../codebase_kick_off/getting_started.md) before submission.

Recover the working job file, submission command, Slurm settings, and shared storage path from that smoke.
Read the W&B reference run and the experiment overrides.
A W&B config is a resolved record; map each field to the repository's YAML format instead of copying object representations.
Read the learning rate (one value or a list of candidates), formal W&B destination, and run naming pattern from the specification.

**Learning rate.** Each specification gives an `lr_grid`. Run every value in it, with `{x}` in the run name set to that LR. State the number of runs in the confirmation message.
Ask for all missing inputs in one message. Keep pending values open during preparation.

## 2. Translate the schedule into launch settings

Create a local schedule template for each schedule in the specification.
Derive optimizer steps, cumulative nominal tokens, and validation targets.
Check every stage boundary and the exact total against the specification.

Keep the reference model, initialization, data order, loss, and optimizer unless the specification explicitly changes them.

Apply these overrides under `fixed` in each formal YAML. Remove the smoke setting `validator.freq` from that YAML.

```yaml
fixed:
  smoke_test: false
  validator.token_intervals: 50
```

Run 51 validations. Evaluate once before training, at 49 intermediate targets, and once after the final update.
Use cumulative nominal training tokens for the targets. Evaluate after the first complete update that reaches each target.
**Do not save checkpoints in this batch. This overrides the specifications' sections "Model initialization and model checkpoints" (checkpoint saving only) and "Stage changes and restart".** These runs repeat the training of finished dense runs and change only the validation, so set `checkpoint.enable = false` in every formal YAML and ignore the specifications' checkpoint table and checkpoint contents. Keep every other field from the reference run. Check in the dry run and in the first minutes of each run that no checkpoint folder is written.

**Validation sequence length: it follows the training schedule. Confirm it with the user.** This is the difference from the earlier dense experiments, which validated at the fixed length `validator.eval_seq_len = 16384` with the `validation_packed_16384` artifact. Every specification for this prompt evaluates each validation at the training length of the last completed update; the validation before training uses the length of update 0.
Find out from the code and the generated config which length each validation actually uses, and how `validator.token_budget` tokens of the same validation set are cut into sequences at each length. Check fixed-length runs, staged runs, and a validation exactly at a stage boundary.
Report the result in the confirmation message before any micro-batch test or formal run: the length used at the initial validation, at each stage, and at the final validation, together with the config fields and code path that set it. If the code does not support the rule of the specification, report the gap and obtain permission before changing shared code. Use the same validation rule for every run in the batch.

Use the pinned TorchTitan LR scheduler. For the three LR-schedule specifications below, set `lr_scheduler.decay_type` to `linear`.
Set `optimizer.learning_rate` to the specified LR for that run. Prefix each field below with `lr_scheduler.` in `fixed`.

Match a specification to a row by its LR-schedule name, ignoring a prefix such as `5stages-dense-` or `llama3_1b-5stages-valaligned-`.

| Specification | `warmup_steps` | `decay_ratio` | `min_lr_factor` |
|---|---:|---:|---:|
| `constant_lr-constant-bs` | 0 | Omit | 1.0 |
| `warmup5_linear95_lr-constant-bs` | 5% of optimizer updates | 0.95 | 0.0 |
| `wsd5_75_20_lr-constant-bs` | 5% of optimizer updates | 0.20 | 0.0 |

Compute 5% warmup from each run's own update count. For example, write 960 for 19,200 updates, 240 for 4,800, 1,920 for 38,400, and 7,358 for 147,160.
Use the warmup and decay counts in the specification's run table when it gives them.
Use TorchTitan's native behavior at each phase boundary. Both decay schedules reach zero after the final optimizer update.

For each decay schedule, check the first update, peak LR, first decrease, and final update.
For constant LR, check that every update uses the candidate LR.
Record the LR after the final update separately.
If the specification and scheduler differ, report the first difference before changing either one.

## 3. Measure microbatches and check each config

Use the micro-batch values that the specification takes from its reference runs, and follow its tests and activation-checkpointing setting. Include one validation pass at each length in the tests, because validation runs at the training length.
Check peak memory and tokens per second at the requested GPU count.
Check that data-parallel ranks × microbatch × accumulation equals the specified global batch at every stage.

Generate one runnable YAML for each approved schedule, LR, and seed combination.
Copy the top-level `num_gpus: 8` from the smoke config for the default allocation.
Copy the specified project name to `wandb.project`. Set `wandb.entity` to the team confirmed for the formal runs.
Set the top-level YAML field `name` from the specification's `wandb_run_name` template.
Substitute only its placeholders. Preserve the spelling, case, and punctuation in the resulting name.

Formal training requires online W&B. Load the approved credentials as in smoke. [W&B credential setup](../../codebase_kick_off/getting_started.md#verify-wb-access)
Set `ENTITY` and `PROJECT` to the formal values from the generated YAML.
Run this check before submission and once inside each allocation before starting workers.

```bash
python scripts/check_wandb_access.py "$ENTITY" "$PROJECT"
```

Require `W&B login OK` for that exact destination. If the check fails, stop and follow the linked credential setup.
Keep the specified project and run names.

From the repository root, run this command for each generated YAML.
Set `CONFIG` to the YAML path. The default allocation uses eight GPUs.

```bash
python scripts/train.py "$CONFIG" --nproc 8 --dry-run
```

Compare the printed token plan with the specification.
Before the first formal run, complete the specification's alignment check with `fixed_16384` and show the comparison to the user.
For token-based validation, check the initial point, the final point, and all intermediate targets.
Estimate run time and storage from the measured settings and the number of runs.

## 4. Submit the agreed runs

Run the checked command inside the smoke job allocation layout. Remove `--dry-run` to start training.
Start one driver on the host with access to all eight GPUs.
The dry run above checks the plan. Use the matching smoke job file for distributed execution.
Check the site's current submission limits. Use shared storage for logs.

Read the optional `max_concurrent_runs` from the specification or the user's launch request.
Omit it or set it to `null` for no user limit. A positive integer limits runs with active GPU allocations.
Count all schedules, LRs, seeds, and retries in this batch together. Exclude pending jobs from this count.

Apply the limit in Slurm. For example, ten runs with a limit of two fit one array with `--array=0-9%2`.
If the batch needs separate submissions, enforce one limit across them. Account for active jobs before submitting retries.
Site limits on running and submitted jobs apply in both cases. [Slurm job arrays](https://slurm.schedmd.com/job_array.html)

Submit the requested combinations within the agreed resource budget.
If the request covers preparation only, report the dry run and the exact submission command.

## 5. Verify launch and completion

Record the code revision, config path, job ID, run directory, and W&B link for each submitted run.
Check the first training loss and the first scheduled validation against the actual training token count.
For a token-based baseline, check validation at zero tokens before the first update.
Check that the sequence length of each validation follows the specification's validation-length rule, at the first validation and at the first validation after each stage change.

If a run stops, check its job state and log, and report it to the user. There is no checkpoint, so a run cannot resume. Restart a stopped run from update 0 only when the user asks, with the same config, node layout, replica count, and environment, and record the lost tokens.

At completion, check the final token count, the 51 validation records, and scheduler exit status.
Report which runs finished and which still need attention. See the [formal run guide](../../formal_training.md) for launch details.
