# Document packing and optional masking

Use [the main guide](../getting_started.md) for the current 1B experiment and 16,384-token fixed validation.

Status: historical implementation plan.
Read the [implementation report](implementation-report.md) for completed changes and check results.
Date: 2026-11-13.

## Purpose

Use documents and document tails that cannot fill one parent window.
Provide a document masking option with a default value of `true`.
Keep the long loss and the crop loss in `LossGradSeqTrainer`.

The first implementation will concatenate documents into a token stream.
The reader will divide this stream into fixed-length windows.
This method can divide a document between windows.
Best-fit packing is outside this first implementation.

## Terms

| Term | Meaning |
| --- | --- |
| Parent | A packed sequence for the long loss measurement. |
| Crop | A contiguous part of a parent for gradient computation. |
| Document masking | An attention restriction that prevents access to other documents. |
| Causal attention | Attention that permits access only to the current and previous positions. |
| Boundary | The connection between two source documents. |
| Valid label | A prediction target that contributes to loss. |
| Padding | Extra positions that fill the final incomplete window. |
| Protocol | The data rules and model settings that define an experiment. |

## Configuration contract

This plan proposes the following public configuration names.
The implementation report must record the final names.

| Key | New experiment value | Effect |
| --- | --- | --- |
| `dataloader.packing` | `stream` | Fill windows across source documents. |
| `document_masking` | `true` | Restrict attention to each document. |
| `loss_seq_len_scheduler.length` | `22256` | Set the constant parent length. |
| `validator.eval_seq_len` | `22256` | Set the constant validation length. |
| Gradient schedule range | `2048` to `8192` | Retain the six schedule trends. |

Use one masking setting for long forward, crop backward, and validation.
Reject configurations that give these phases different masking settings.
Preserve the existing reader for historical experiment configurations.
Select the new reader explicitly in new configurations.
Record the resolved packing mode and masking value in each experiment manifest.

## 1. Construct the token stream

Reuse the current raw token cache and length indexes.
Keep the existing train and validation split membership.
Keep a deterministic document order for each rank.
Retain the current assignment of documents to data parallel ranks.
Do not mix documents between the train and validation splits.

1. Add the native BOS and EOS tokens to each source document.
2. Append documents to the stream for their assigned rank.
3. Read enough tokens to construct the input and shifted labels.
4. Retain unconsumed tokens for the next window.
5. Preserve the successor token across adjacent windows.
6. Record every source fragment that contributes to a window.

A short document must remain in the stream.
A document tail must remain in the stream.
Storage shard boundaries must not discard tokens.
Packing must not require another tokenization pass or another dataset upload.

Retain the label that predicts EOS.
Ignore the artificial target that predicts the next document's BOS from the previous document's EOS.
Apply this label rule in both masking modes.
Use `IGNORE_INDEX` for ignored labels.
Keep each document's first text token as a valid target after BOS.

For infinite training, carry the final fragment into the next epoch.
Record epoch transitions and repeated data in the reader state.
For finite validation, pad the final incomplete window.
Ignore padding labels and prevent padding from affecting real tokens.
Reject an empty rank stream instead of waiting indefinitely.

## 2. Preserve positions and select attention

Use the same position rule in both masking modes.
Reset positions at each document start inside a window.
Rebase the initial document fragment to position zero in each window.
Keep positions within the configured model capacity.

When `document_masking=true`, combine causal attention with the document restriction.
When `document_masking=false`, apply causal attention without the document restriction.
Continue to exclude padding in both modes.
Disabling document masking must not disable causal attention.

The local TorchTitan Decoder already constructs masks from document positions.
Its current implementation couples document boundaries to RoPE positions.
Add a project adapter that selects the attention rule independently of those positions.
Use the same adapter in training and validation.
Do not change global behavior for unrelated TorchTitan models.
Do not create a dense attention tensor with quadratic storage.

The initial implementation supports the current FlexAttention backend.
Reject unsupported backends with a specific configuration error.
Record any required upstream change in the final report.

## 3. Preserve crop behavior

Retain the existing seed, step, rank, and stream rules for crop selection.
Permit a crop to cross document boundaries inside its parent.
Preserve boundaries and ignored labels when the crop function selects tokens.
Rebase only the initial fragment in each crop.
Keep subsequent document position resets.

The existing `crop_batches` function already handles position resets.
Check this behavior with parents that contain several document fragments.
Regroup crops without changing their order or label alignment.
Reject a batch with no valid gradient labels.

The short loss uses the selected crop context.
The long loss uses the full parent context before the optimizer update.
Neither mode supplies attention across separate parent windows.

## 4. Restore the exact data state

Save the document cursor, token offset, epoch, and unconsumed stream state.
Save boundary metadata for any retained tokens.
Define the saved state after each emitted batch.
Restore the same next batch after a checkpoint load.
Preserve this property when microbatch size or crop length changes.

Include these values in the checkpoint protocol identity:

- Dataset manifest hash and split.
- Packing mode and packing version.
- Special token and boundary label rules.
- Position rule and masking value.
- Data parallel rank layout.
- Validation artifact identity.
- Existing schedule and crop settings.

Reject a checkpoint from the old reader when the new reader is active.
Reject a checkpoint with a different masking value.
Start new experiment runs for the new protocol.

## 5. Prepare validation and metrics

Create a new validation artifact from the validation split only.
Keep the old validation artifact available for historical runs.
Record all document fragments for each packed sequence.
Record the packing version and position rule in the artifact manifest.
Check tokenizer hashes, tensor hashes, and source provenance.

Permit `IGNORE_INDEX` only in the label array.
Reject invalid token IDs and invalid positions.
Store the exact valid label count in the artifact manifest.
Do not calculate that count as sequence count multiplied by sequence length.
Use the valid label count as the loss denominator across ranks.
Ensure that each validation sequence contributes once per evaluation.
Keep the sequence count compatible with the supported data parallel layouts.
Account explicitly for any final padding needed by those layouts.

Prepare one artifact for all six runs.
Freeze its membership before the runs start.
Target approximately the previous evaluation budget of 20.9 million valid labels.
Record the final count instead of assuming the previous 936-window proposal remains exact.

Rename the long loss outputs to `loss_metrics/long_avg_loss` and `loss_metrics/long_max_loss`.
Retain `loss_metrics/short_seq_loss`.
Keep the maximum loss definition as the maximum of the rank-local long loss means.
Record physical positions, valid labels, ignored boundary labels, and padding separately.
Record document counts and document fragment lengths for packed windows.

Recalculate the six schedule budgets using the final label rules.
Distinguish the scheduled token budget from the measured valid gradient labels.
Report each run's deviation from the target of approximately 12.08B gradient tokens.
A packed parent length does not establish an equal single-document context length.

## 6. Files and responsibilities

Paths in this table are relative to the repository root.
New module names can change during implementation.
The final report must list the actual files and relevant functions.

| File or area | Planned change |
| --- | --- |
| `torchtitan_dynamic_seq_len/data/long_documents.py` | Reuse artifact checks and expose an explicit reader selection. |
| New module under `torchtitan_dynamic_seq_len/data/` | Construct packed windows and preserve reader state. |
| New attention adapter within this project | Select causal attention with or without document restrictions. |
| `torchtitan_dynamic_seq_len/config_registry.py` | Register the packed experiment configuration. |
| `torchtitan_dynamic_seq_len/loss_grad_seq/trainer.py` | Check options and extend the checkpoint identity. |
| `torchtitan_dynamic_seq_len/data/seq_grad_windows.py` | Preserve crop boundaries and check existing position behavior. |
| `torchtitan_dynamic_seq_len/data/fixed_validation.py` | Support ignored labels and exact valid label budgets. |
| `torchtitan_dynamic_seq_len/loss_grad_seq/validation.py` | Preserve the fixed long validation length and masking setting. |
| `scripts/prepare_long_validation.py` | Create packed validation arrays and fragment provenance. |
| `torchtitan_dynamic_seq_len/loss_grad_seq/metrics.py` | Rename long loss metrics and expose packing measurements. |
| `scripts/prepare_longseq_loss_grad_configs.py` | Generate distinct packed configurations for the six schedules. |
| `tests/` | Check data coverage, attention isolation, recovery, and distributed losses. |

## 7. Acceptance checks

### CPU checks

1. Check exact input and label values against a small independent reference stream.
2. Check documents shorter than, equal to, and longer than the parent length.
3. Check boundaries inside windows and at window edges.
4. Check document tails across storage shards and epoch transitions.
5. Check that each source prediction target appears once during a finite pass.
6. Count excluded boundary targets separately from retained document targets.
7. Check that data parallel ranks have disjoint source documents.
8. Compare uninterrupted batches with batches after checkpoint recovery.
9. Check crops that start inside a document or cross several boundaries.
10. Check validation provenance, padding, and the exact valid label budget.
11. Check rejection of incompatible checkpoints and unsupported configurations.

### Attention and GPU checks

1. Compare packed masked loss with separate-document loss on a deterministic small model.
2. Compare gradients from those two computations within a stated numerical tolerance.
3. Change document A while holding document B constant in the same parent.
4. Check that masking preserves B's valid predictions and corresponding gradient contribution.
5. Check that the unmasked path permits attention from B to A.
6. Compare the unmasked path with a causal reference using identical tokens and positions.
7. Check that neither mode permits future-token attention.
8. Exercise long loss, crop backward, and validation with both masking values.
9. Exercise parent length 22,256 and crop lengths 2,048 and 8,192.
10. Exercise schedule extrema, partial microbatches, and a checkpoint restart.
11. Check token-weighted loss reduction with unequal valid label counts across ranks.
12. Record peak GPU memory, throughput, and compilation behavior.

Use conservative microbatches for the first GPU tests.
Do not assume that document masking reduces measured peak memory.
Consult current ARC documentation before selecting job resources.
Check live resource availability before submitting test jobs.
Use account `jason_optai-paid` and an applicable short QoS for those jobs.
Record job IDs and outcomes as evidence.

## 8. Completion report

Create `implementation-report.md` in this directory after implementation and checks.
The report must distinguish completed work, failed checks, and remaining limitations.
Do not use planned behavior as evidence of completed work.

Include these sections in the report:

1. State the final behavior and supported configurations.
2. Map each change to its file, function, and training phase.
3. Explain token flow from the raw cache through packing, attention, crops, and loss.
4. State the boundary label, position, padding, and epoch rules.
5. List configuration keys, defaults, and example overrides.
6. Record validation identities and exact token counts.
7. Record checkpoint compatibility and recovery results.
8. List test commands, results, numerical tolerances, and GPU measurements.
9. Identify run configurations and any submitted jobs.
10. State remaining limitations and the procedure to select the historical reader.

Add a directory README with links to the plan and completed report.
Update affected experiment documentation with links to the new report.
Keep historical results separate from new measurements.

## Writing checks

Follow the [project writing guide](../writing.md).
Use short sentences, active voice, and defined technical terms.
Run the ASD-STE100 skill linter on the plan and final report.
Review technical meaning after each language correction.
The linter checks writing patterns, not the complete ASD approved dictionary.
Do not claim certified compliance without the dictionary review.
