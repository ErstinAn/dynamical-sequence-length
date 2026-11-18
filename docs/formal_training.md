# Prepare and resume a formal run

Use this procedure after the [target-server smoke test](codebase_kick_off/getting_started.md) passes on the requested layout.
Smoke and each formal run use two nodes with eight H200 GPUs each by default (16 GPUs total).
Work from the repository root with the same Python environment.
Keep the tracked code and configuration files unchanged.

Use the [experiment launch prompt](sl_schedule_docs/experiment_launch_prompt.md) with the agreed experiment document.
Do not submit a formal run from the smoke settings.

Store generated configurations in `local/generated_configs/`.
Keep only `configs/smoke.yaml` under `configs/`.
Use `torchtitan_config: llama3_1b` and `smoke_test: false` for formal runs.
Require online W&B.

Specify these settings explicitly:

| Fields | Required decision |
|---|---|
| `token_budget`, `training.steps` | Exact budget and the first complete final step that reaches it |
| `training.seq_len` | Context capacity for validation and every scheduled training length |
| `training.global_batch_size`, `training.local_batch_size` | Initial global batch and measured microbatch |
| `seq_len_scheduler.*` | Sequence lengths, boundaries, and schedule kind |
| `batch_size_scheduler.*` | Global batch schedule |
| `microbatch_scheduler.*` | Measured microbatches for the target GPU and DP degree |
| `optimizer.learning_rate`, `lr_scheduler.*` | Agreed learning rate and schedule |
| `validator.freq` | Steps between complete fixed validations |

The data stream is finite.
The training manifest contains 1,348,067 documents and 30,000,004,070 raw text tokens.
The raw count excludes beginning-of-document and end-of-document tokens.
Nominal token budget is not identical to an epoch count or the original document token total.
Packing boundaries, special tokens, and final windows affect that relationship.
Preserve the document members and reading order across schedule comparisons.

Do not silently repeat the dataset after exhaustion.

For a constant global batch and length, calculate steps as `ceil(budget / (global_batch * sequence_length))`.
For changing schedules, sum each step's product until the first complete step reaches the budget.
The approximately 20B budget consumes only part of this finite dataset.
Its exact document position depends on the schedules and DP partition.

Run this command from the repository root:

```bash
python scripts/train.py local/generated_configs/experiment.yaml \
  --nproc <GPU_COUNT> --dry-run
```

Require ten distinct save steps at nominal thresholds 10%, 20%, through 100%.
The planner uses the first complete step that reaches each threshold.
It reports any excess above the threshold.
It rejects schedules that cross multiple thresholds in one step.
Keep all ten complete states.
Do not implement this rule as retention of the ten most recent checkpoints.

Check runtime, storage, and sweep scope against the agreed budget. Reuse the verified smoke allocation by default.
Estimate training time separately for each sequence length and batch setting.
Multiply measured seconds per step by the planned steps for that setting.
Add all phase times, complete validation events, checkpoint saves, and process initialization.
Treat estimates from two steps per length as preliminary.

After agreement, use the distributed launch command verified during smoke.
The dry run checks configuration settings only. The current launcher starts its workers on a single host.
Append `--index <INDEX>` to select a point from an agreed YAML sweep.

The run directory is `<DUMP_ROOT>/<NAME>/<EXPERIMENT_ID>`.
It contains the configuration, command, source hash, token plan, and W&B segment records.
Checkpoint directories are `checkpoint/step-<STEP>/` within that run directory.
Each checkpoint contains model, optimizer, learning rate scheduler, counters, and data reader state.

To resume an interrupted run, use the same code, environment, node layout, and DP degree.
Use the recovery command verified during smoke. Pass the original run directory to `--resume-dir`.

Do not supply replacement schedules or microbatches.
The launcher checks recorded source and configuration hashes.
To resume a run created before this cleanup, use the original checkout for that run.
It resumes the latest checkpoint with DCP metadata.
It rejects a run that already contains its final checkpoint.

Each recovery creates a new W&B run segment in the same experiment group.
The new segment uses the restored token count for its first updates.
This preserves earlier history when the checkpoint precedes the last uploaded step.
Use `segment-*.json` to identify W&B run IDs and recovery steps.
