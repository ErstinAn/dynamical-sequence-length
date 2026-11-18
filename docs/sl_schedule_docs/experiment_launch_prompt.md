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
Recover the working job example and shared storage path from that smoke.
Read the reference config and the experiment overrides.
Collect the GPU count, LR candidates, formal W&B destination, and run naming pattern.
Ask for all missing inputs in one message. Keep pending values open during preparation.

## 2. Translate the schedule into launch settings

Create a local schedule template for each schedule in the specification.
Derive optimizer steps, cumulative nominal tokens, validation targets, and checkpoint targets.
Check every stage boundary and the exact total against the specification.

Set `smoke_test: false` in the formal settings.
Use `validator.token_intervals` for equally spaced token targets. Use `validator.freq` for a step interval.
Keep the reference model, initialization, data order, loss, and optimizer unless the specification explicitly changes them.

## 3. Measure microbatches and check each config

Reuse matching measurements from the target server. Measure missing settings with complete optimizer updates.
Check peak memory and tokens per second at the requested GPU count.
Check that replicas × microbatch × accumulation equals the specified global batch at every stage.

Generate one runnable YAML for each approved schedule, LR, and seed combination.
Set `wandb.entity` and `wandb.project` from the formal destination. Set `name` to the unique display name for that combination.
Check access to the formal project with the account from smoke. Keep W&B online.

From the repository root, run this command for each generated YAML.
Set `CONFIG` to the YAML path. Set `GPU_COUNT` to the agreed GPU count per run.

```bash
python scripts/train.py "$CONFIG" --nproc "$GPU_COUNT" --dry-run
```

Compare the printed token plan with the specification.
For token-based validation, check the initial point, the final point, and all intermediate targets.
Estimate run time and storage from the measured settings and the number of runs.

## 4. Submit the agreed runs

Use the job example from this server to prepare the local submission file.
Inside the GPU allocation, run the checked command without `--dry-run`.
Use shared run storage for logs and checkpoints.
Submit only the requested combinations within the agreed resource budget.
If the request covers preparation only, report the dry run and the exact submission command.

## 5. Verify launch and recovery

Record the code revision, config path, job ID, run directory, and W&B link for each submitted run.
Check the first training loss and the first scheduled validation against the actual training token count.
For a token-based baseline, check validation at zero tokens before the first update.

If a run stops, check its job state and latest complete checkpoint.
For recovery, allocate the same replica count. Use the same environment.
Set `RUN_DIRECTORY` to the original run directory.

```bash
python scripts/train.py --resume-dir "$RUN_DIRECTORY"
```

At completion, check the final token count, validation records, checkpoint count, and scheduler exit status.
Report which runs finished and which still need attention. See the [formal run guide](../formal_training.md) for launch and recovery details.
