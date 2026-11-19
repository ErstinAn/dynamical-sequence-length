---
name: launch-sequence-experiment
description: Generate local training configurations from a sequence schedule specification. Launch the agreed runs on a server that passed the repository smoke test.
---

# Launch a sequence length experiment

Give the agent one experiment file. For example, send this message from the repository root.

```text
Use @docs/sl_schedule_docs/constant_lr-constant-bs.md and its linked launch prompt.
Prepare the configurations. Launch the runs that we agreed to submit.
```

Use the named file as the experiment specification. For a new experiment, replace that one path.
Keep shared experiment changes in Markdown under `docs/sl_schedule_docs/`.
Generate machine-specific YAML and job files under the ignored directory `local/generated_configs/<experiment>/`.
If the existing code cannot express a requirement, identify the gap and its effect.
Obtain explicit permission before changing shared code.

## 1. Read the specification and the smoke record

Use the checkout, environment, data, and tokenizer from the successful smoke on this server.
The verified data and tokenizer need no Hugging Face login. If data is missing, restore the pinned public dataset.
Use the anonymous download for `Erstin/fineweb-edu-long-30b`. [Data recovery procedure](../codebase_kick_off/getting_started.md#download-and-rebuild-the-dataset)

Use one node with eight NVIDIA H200 GPUs for each run by default.
Use a different layout only when the user requests it. Check the smoke record for the requested layout.
For the default, verify one hostname and eight ranks in one training group.
If the matching smoke record is missing, follow the [target-server smoke guide](../codebase_kick_off/getting_started.md) before submission.

Recover the working job file, submission command, Slurm settings, and shared storage path from that smoke.
Read the reference config and the experiment overrides.
Read the LR candidates, formal W&B destination, and run naming pattern from the specification.
Ask for all missing inputs in one message. Keep pending values open during preparation.

## 2. Translate the schedule into launch settings

Create a local schedule template for each schedule in the specification.
Derive optimizer steps, cumulative nominal tokens, validation targets, and checkpoint targets.
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
Keep all ten complete checkpoints at 10% increments of the token budget, including the final state.

Use the pinned TorchTitan LR scheduler. For the three Phase 1 experiments, set `lr_scheduler.decay_type` to `linear`.
Set `optimizer.learning_rate` to the LR candidate for that run. Prefix each field below with `lr_scheduler.` in `fixed`.

| Specification | `warmup_steps` | `decay_ratio` | `min_lr_factor` |
|---|---:|---:|---:|
| `constant_lr-constant-bs` | 0 | Omit | 1.0 |
| `warmup5_linear95_lr-constant-bs` | 5% of optimizer updates | 0.95 | 0.0 |
| `wsd5_75_20_lr-constant-bs` | 5% of optimizer updates | 0.20 | 0.0 |

For 5% warmup, write 960 for 19,200 optimizer updates and 240 for 4,800 optimizer updates.
Use TorchTitan's native behavior at each phase boundary. Both decay schedules reach zero after the final optimizer update.

For each decay schedule, check the first update, peak LR, first decrease, and final update.
For constant LR, check that every update uses the candidate LR.
Record the LR after the final update separately.
If the specification and scheduler differ, report the first difference before changing either one.

## 3. Measure microbatches and check each config

Reuse matching measurements from the target server. Measure missing settings with complete optimizer updates.
Check peak memory and tokens per second at the requested GPU count.
Check that replicas × microbatch × accumulation equals the specified global batch at every stage.

Generate one runnable YAML for each approved schedule, LR, and seed combination.
Copy the top-level `num_gpus: 8` from the smoke config for the default allocation.
Copy the specified project name to `wandb.project`. Set `wandb.entity` to the team confirmed for the formal runs.
Set the top-level YAML field `name` from the specification's `wandb_run_name` template.
Substitute only its placeholders. Preserve the spelling, case, and punctuation in the resulting name.

Formal training requires online W&B. Load the approved credentials as in smoke. [W&B credential setup](../codebase_kick_off/getting_started.md#verify-wb-access)
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
For token-based validation, check the initial point, the final point, and all intermediate targets.
Estimate run time and storage from the measured settings and the number of runs.

## 4. Submit the agreed runs

Run the checked command inside the smoke job allocation layout. Remove `--dry-run` to start training.
Start one driver on the host with access to all eight GPUs.
The dry run above checks the plan. Use the matching smoke job file for distributed execution.
Check the site's current submission limits. Use shared storage for logs and checkpoints.

Read the optional `max_concurrent_runs` from the specification or the user's launch request.
Omit it or set it to `null` for no user limit. A positive integer limits runs with active GPU allocations.
Count all schedules, LRs, seeds, and retries in this batch together. Exclude pending jobs from this count.

Apply the limit in Slurm. For example, ten runs with a limit of two fit one array with `--array=0-9%2`.
If the batch needs separate submissions, enforce one limit across them. Account for active jobs before submitting retries.
Site limits on running and submitted jobs apply in both cases. [Slurm job arrays](https://slurm.schedmd.com/job_array.html)

Submit the requested combinations within the agreed resource budget.
If the request covers preparation only, report the dry run and the exact submission command.

## 5. Verify launch and recovery

Record the code revision, config path, job ID, run directory, and W&B link for each submitted run.
Check the first training loss and the first scheduled validation against the actual training token count.
For a token-based baseline, check validation at zero tokens before the first update.

If a run stops, check its job state and latest complete checkpoint.
For recovery, reuse the node layout, replica count, and environment from the original run.
Use the recovery command verified during smoke with the original run directory.

At completion, check the final token count, validation records, checkpoint count, and scheduler exit status.
Report which runs finished and which still need attention. See the [formal run guide](../formal_training.md) for launch and recovery details.
