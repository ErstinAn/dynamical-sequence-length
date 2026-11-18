# Token counting and loss logging audit

Audit date: 2026-11-18.
This audit records existing behavior and the required changes.
It does not change training source code, loss calculations, schedules, or checkpoint formats.
It does not submit a training job.
The [run guide](../codebase_kick_off/getting_started.md) defines the required interface.

## Required behavior

Retain `loss_metrics/global_avg_loss`, `loss_metrics/global_max_loss`, and `validation_metrics/loss`.
Use the existing `n_tokens_seen` counter as the W&B horizontal axis for all three metrics.
Train loss uses the actual training sequence length at that step.
Validation uses the fixed 936 windows of length 16,384.
Exclude long and short loss series unless the experiment document explicitly requests them.

The user clarified that only the horizontal axis must change.
This clarification replaces the earlier proposal to rename the losses and remove global maximum loss.
The existing metric names and calculations remain required.

Let `B_t` denote the global batch size at optimizer step `t`.
Let `L_t` denote the actual training sequence length at that step.
After completed step `s`, the nominal total is `T_s = sum(B_t * L_t, t=1..s)`.
Global batch already includes all data parallel ranks.
Do not multiply `T_s` by the GPU count again.
Do not approximate `T_s` as `s * B_s * L_s` when schedules change.

Masked labels reduce the valid target count but do not reduce nominal tokens.
`n_tokens_seen` counts nominal tokens, while `n_valid_tokens_seen` counts valid prediction targets.
Validation targets do not increase `T_s`.
Gradient accumulation partitions one global batch and does not create extra nominal tokens.
The checkpoint thresholds use this same `T_s`.
Ten distinct checkpoints remain required at 10% increments of the budget, including the final state.

## Code findings

The inspected TorchTitan commit is `737594746fda65a6d94dc9482ef07863a80c8588`.
Its editable installation is outside this repository.
The [structured record](20261118_token_metrics_audit.json) contains source hashes, commands, and environment details.

| Path and function | Observed behavior | Assessment |
|---|---|---|
| `dynamic_seq_bs/trainer.py:accumulation_for_step` | Requires global batch to equal DP degree times microbatch times accumulation | Matches required arithmetic for supported replicated DP |
| `dynamic_seq_bs/trainer.py:_set_batch_step` | Selects sequence length, global batch, and microbatch before each step | Uses actual scheduled values |
| Upstream `Trainer.post_dataloading_process` | Adds `labels.numel()` to the local cumulative counter for each microbatch | Counts nominal positions, including ignored labels |
| Upstream `Trainer.train_step` | Sums local cumulative counters across the loss mesh when logging | Produces global `n_tokens_seen` without another GPU multiplier |
| `dynamic_seq/trainer.py:forward_backward_step` | Adds global valid targets once per optimizer step | Does not count each accumulation pass twice |
| `dynamic_seq/trainer.py:state_dict` | Saves local nominal count, global valid count, and the schedule signature | Same-topology recovery preserves counts |
| `dynamic_seq_bs/trainer.py:_schedule_signature` | Includes DP degree and all three schedules | Rejects unsupported topology or schedule changes on recovery |
| `data/fixed_validation.py:_validate` | Divides global summed loss by global valid targets | Weights valid targets across unequal final validation batches |
| `dynamic_seq/metrics.py:validation_scope` | Separates validation throughput counters from training counters | Validation does not increase the training budget |
| `scripts/prepare_longseq_joint_configs.py` | Sums scheduled global batch times sequence length | Matches required arithmetic, but retains a historical budget and paths |

Paths in this table are relative to `torchtitan_dynamic_seq_len/`, except the script and upstream entries.
The dynamic trainer supports one GPU or replicated DP.
These findings do not establish correctness for other forms of parallelism.

The upstream variable `global_avg_loss` already contains the required training loss value.
It divides summed loss contributions by all valid targets in the current optimizer step.
It does not average losses across the logging interval.
The `global_max_loss` value is the maximum of the per-rank mean losses.
Preserve both metric names and calculations.
An unweighted average of rank losses or microbatch losses can change the value when valid target counts differ.

The nominal checkpoint counter is local to a rank.
The valid target checkpoint counter is global.
These counters are not interchangeable.
The current schedule checks require equal nominal work per rank and retain the DP degree during recovery.
Only under these conditions can local nominal count times DP degree recover the global nominal count.

## Observed gaps

1. W&B receives the optimizer step without a custom training-token axis for these losses.
2. Validation records do not explicitly contain the cumulative training token count.
3. `n_tokens_seen` reaches the metric processor only on training logging steps.
4. The native-vocabulary 1B entry does not yet combine the required joint schedules and token axis.
5. Ten checkpoint thresholds based on nominal tokens remain unimplemented.

The CPU probe demonstrated the validation gap at a step without a training log.
The cumulative training count was 16, but the validation record contained no training-token axis.
Using the last logged training count would place this validation at 8 instead of 16.

The raw probe output predates the user's clarification about loss names.
Its reference to old loss keys no longer describes a requirement gap.
The missing explicit token axis remains a gap.
The raw execution records remain unchanged.

Historical `loss_grad_seq` deliberately reports long and short losses.
Its `n_tokens_seen` counts gradient crop positions, while `parent_tokens_seen` counts parent positions.
Do not select this trainer for the ordinary sequence length experiment.
Retain its behavior and tests until the user chooses its archival scope.

## Executed checks

All commands ran from the repository root.
They reused `/home/kangan/.conda/envs/torchtitan-macro/bin/python`.
The CPU model fixtures execute production training, normalization, metric, and validation methods.
They replace GPU initialization and the full 1B model.
These checks do not establish a successful installation from zero or a successful 1B GPU run.

| Check | Result |
|---|---|
| Targeted CPU suite | 33 tests, 32 passed, 1 optional Gloo test skipped |
| Optional validation Gloo test, run separately | Passed with two CPU ranks |
| Existing dynamic batch distributed script | Passed with two CPU ranks, including distributed checkpoint recovery |
| Additional CPU probe | Passed checks of changing-length counts, masked targets, current-step loss, validation isolation, and failure rollback |
| Current fixed labels | Shape `(936, 16384)`, 15,335,424 valid targets, zero ignored labels, checksum matched |
| W&B online history | Not executed |
| Full 1B model and production data training | Not executed |

The first distributed attempt failed because the sandbox blocked local sockets.
The repeated test passed with local socket access.
These checks used no cluster job or GPU allocation.
The separate validation Gloo test also used local socket access.

The independent CPU probe used nominal increments `8, 8, 16, 32, 32, 32`.
Its cumulative counts were `8, 16, 32, 64, 96, 128`.
One ignored label per step produced 122 valid targets after six steps.
The probe also checked an injected failure before the optimizer update.
That failure left the cumulative counters unchanged.

The distributed script checked global normalization with unequal valid target counts across ranks.
It restored model, optimizer, scheduler, trainer counters, and data reader state through distributed checkpoint files.
Resumed updates and counters matched uninterrupted updates.
Coordinated data exhaustion did not advance the counters.
These checks used two CPU ranks and a scalar model.

The fixed-label check reused the current local dataset arrays.
It checked their count and checksum without executing model validation over those arrays.
The fixed validation manifest SHA256 is `eb2faf62a36e324d07e08da73170e1268182cdfea1f38352362b86232f47155b`.

Run the targeted suite with the recorded environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:cluster/vt_arc OMP_NUM_THREADS=1 \
/home/kangan/.conda/envs/torchtitan-macro/bin/python -m unittest \
  tests.test_dynamic_seq.TrainerTest \
  tests.test_dynamic_seq_bs.TrainerTest \
  tests.test_dynamic_seq_bs.SchedulerTest \
  tests.test_fixed_validation tests.test_fixed_document_order -v
```

Run the distributed training and recovery check where local sockets are available:

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 \
/home/kangan/.conda/envs/torchtitan-macro/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m tests.dynamic_seq_bs_distributed
```

Run the separate distributed validation check:

```bash
timeout 45s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 \
FIXED_VALIDATION_GLOO_TEST=1 \
/home/kangan/.conda/envs/torchtitan-macro/bin/python -m unittest \
  tests.test_fixed_validation.FixedBudgetValidationTest.test_real_two_rank_reduction_and_partial_batches -v
```

These environment paths describe the audit machine only.
Use the independently installed Python environment for the final delivery checks.
The [structured record](20261118_token_metrics_audit.json) links each raw log and the probe source.

## Implementation and acceptance plan

1. Set the W&B token axis for the existing average, maximum, and validation loss metrics.
2. Publish the current global nominal count after every completed optimizer step, including steps without a training log.
3. Include that count explicitly in each validation record.
4. Restore the count before logging after checkpoint recovery.
5. Use the same count for checkpoint thresholds and budget completion checks.
6. Connect the native-vocabulary 1B model, agreed data reader, joint schedules, and main logger in one configuration entry.
7. Add regression tests for preserved loss keys, axis definitions, masked targets, skipped logging steps, and recovery.
8. Check real W&B history during the agreed GPU smoke test.

Preserve the loss value, attention mask, gradients, data membership, and document order during these changes.
Keep historical monitoring behavior separate from the main experiment entry.
Do not add guessed schedules or hyperparameters to the smoke placeholder.

Require the existing global average loss, global maximum loss, and validation loss series in the main W&B run.
Require all three default axes to use the same cumulative nominal count.
Compare logged counts with an independent sum over the agreed schedules.
Check token continuity and the next checkpoint threshold after recovery.
Report threshold excess from complete optimizer steps explicitly.
Do not mark this logging requirement complete until these acceptance checks pass.
