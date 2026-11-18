# Document packing and masking implementation

This historical report uses an earlier packed validation artifact.
Use [the main guide](../getting_started.md) for the current 1B experiment and single-document validation.

Date: 2026-11-13.
Status: implementation complete. CPU and GPU checks passed.

## Result

The new reader fills each parent from a continuous stream of documents.
It retains documents shorter than the parent length.
It also retains document tails across windows, storage shards, and training epochs.
The raw dataset and tokenizer remain unchanged.

The `document_masking` option defaults to `true`.
One setting controls long forward, crop backward, and validation.
Both settings retain the same tokens, labels, document order, and RoPE positions.
Only the attention restriction changes.

## Data flow

1. Read raw tokens from the existing document cache.
2. Insert native BOS and EOS tokens around each document.
3. Combine documents into the stream for their assigned data parallel rank.
4. Read one parent and its successor token.
5. Create labels with the next-token shift.
6. Ignore the artificial EOS-to-BOS prediction between documents.
7. Measure the long loss without gradients.
8. Select a reproducible contiguous crop from each parent.
9. Apply the crop gradients through the existing accumulation procedure.

Each document with N raw tokens supplies N+1 valid prediction targets during a finite pass.
The EOS target contributes to loss.
The target after each document's BOS includes its first text token.
An ignored boundary target does not remove a document text token.

The reader divides documents when they cross parent boundaries.
Attention does not extend across separate parent windows.
The reader does not implement best-fit packing.

## Changes by phase

| Phase | File and entry point | Change |
| --- | --- | --- |
| Reader selection | [long_documents.py](../../torchtitan_dynamic_seq_len/data/long_documents.py), `LongDocumentDataLoader` | Select the historical reader or the stream reader. |
| Stream assembly | [packed_documents.py](../../torchtitan_dynamic_seq_len/data/packed_documents.py), `PackedDocumentDataset` | Preserve source tokens, fragments, lookahead, and reader state. |
| Dynamic batches | `PackedDocumentBatchDataset` in the same file | Read the current microbatch size before each batch. |
| Attention | [attention.py](../../torchtitan_dynamic_seq_len/loss_grad_seq/attention.py), `select_attention` | Select document restrictions independently of RoPE positions. |
| Training | [trainer.py](../../torchtitan_dynamic_seq_len/loss_grad_seq/trainer.py), `LossGradSeqTrainer` | Check options, route attention, extend checkpoint identity, and record packing metrics. |
| Crop selection | [seq_grad_windows.py](../../torchtitan_dynamic_seq_len/data/seq_grad_windows.py), `crop_batches` | Reuse existing boundary and position handling without changing this file. |
| Validation preparation | [prepare_packed_validation.py](../../scripts/prepare_packed_validation.py), `prepare_packed` | Write arrays, exact label counts, and source fragments. |
| Preparation command | [prepare_long_validation.py](../../scripts/prepare_long_validation.py) | Add `--packing stream`. |
| Validation artifact | [fixed_validation.py](../../torchtitan_dynamic_seq_len/data/fixed_validation.py), `FrozenValidationArtifact` | Accept schema 2 and check ignored labels, vocabulary bounds, positions, and exact counts. |
| Validation forward | [validation.py](../../torchtitan_dynamic_seq_len/loss_grad_seq/validation.py), `LossLengthValidator` | Apply the shared attention setting and check the source split. |
| Validation prefixes | `ValidationPrefixView` in the same file | Count valid labels in the selected prefix. |
| Metrics | [metrics.py](../../torchtitan_dynamic_seq_len/loss_grad_seq/metrics.py), `LongLossLogger` | Rename native long loss keys at the logger boundary. |
| Configuration | [config_registry.py](../../torchtitan_dynamic_seq_len/config_registry.py) | Add `llama3_100m_packed_documents_loss_grad_seq`. |
| Schedule generation | [prepare_packed_configs.py](../../scripts/prepare_packed_configs.py) | Create separate packed experiment and test configurations. |
| Shared schedule helper | [prepare_longseq_loss_grad_configs.py](../../scripts/prepare_longseq_loss_grad_configs.py), `make_schedule` | Accept minimum and maximum lengths while preserving historical defaults. |

No upstream TorchTitan source file changed for this implementation.
The existing FlexAttention backend supplies the document mask.

## Attention and position rules

Positions restart at zero at each document boundary inside a parent.
Positions also restart at the first fragment of each parent or crop.
Subsequent document boundaries remain intact after crop selection.

With masking enabled, tokens attend only to current and earlier positions within the same document fragment.
With masking disabled, tokens can attend to earlier documents in the same parent or crop.
The unmasked path uses synthetic continuous positions only to construct the attention mask.
The model still receives the original document positions for RoPE.
Both paths preserve causal attention.

Finite streams pad the final incomplete window with EOS-valued input positions.
The loss excludes labels at those positions.
Causal attention prevents real tokens from attending to the later padding positions.
Infinite training carries the final fragment into the next epoch without padding.

## Configuration and experiment identity

The new YAML files contain these settings:

```yaml
fixed:
  dataloader.packing: stream
  validator.dataloader.packing: stream
  document_masking: true
  training.seq_len: 22256
  loss_seq_len_scheduler.length: 22256
  validator.eval_seq_len: 22256
  validator.token_budget: 20830867
```

Set `document_masking: false` to permit cross-document attention.
Do not set a separate validation masking value.
The trainer copies its resolved value into the validator configuration.

The six schedules retain 3,840 optimizer steps.
Cyclic schedules retain 12 periods of 320 steps.
Crop lengths range from 2,048 to 8,192.
All lengths are multiples of eight.

| Schedule | Global batch range | Scheduled gradient positions |
| --- | ---: | ---: |
| Batch increases, crop decreases | 384 to 1,536 | 12,079,608,768 |
| Batch decreases, crop increases | 384 to 1,536 | 12,079,608,768 |
| Batch and crop cycle inversely | 384 to 1,536 | 12,078,316,032 |
| Linear batch, cyclic crop, cosine LR | 304 to 920 | 12,079,601,344 |
| Linear batch, cyclic crop, constant LR | 304 to 920 | 12,079,601,344 |
| Four-stage cycle | 296 to 1,200 | 12,075,466,752 |

These totals count scheduled positions before boundary exclusions.
The nominal error from the target is at most 0.0342 percent.
Actual valid gradient labels depend on the selected crops.
Use `n_valid_tokens_seen` to compare the measured gradient budgets.
Do not treat the nominal totals as exact valid label counts.

The new W&B group is `packed-masked-loss22256-grad2048-8192-v1`.
The initial configurations used `longseq-finwebedu-llama-100m-seqbs-joint-scheduler`.
The later [formal submission](formal-submission.md) uses `longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler`.
The test configurations use `dynamic-seq-bs-debug`.
The six full training runs are not submitted by this implementation check.

## Checkpoint recovery

The reader saves its document cursor, token offset, and epoch.
It also saves label counters and the packing protocol.
Lookahead does not advance the saved cursor.
The next read reconstructs the successor token from that cursor.

The trainer protocol signature now uses version 2.
It includes masking, packing, dataset identity, validation identity, and the existing schedule settings.
A different signature causes checkpoint rejection.
Historical version 1 checkpoints cannot resume through this trainer version.
Use the historical code snapshot to resume those checkpoints.

The default value `dataloader.packing: none` still selects the historical reader.
That option preserves the old window policy for new runs.
It does not bypass checkpoint compatibility checks.

## Validation evidence

The main artifact is [longseq_validation_llama3_packed_22256](../../assets/data/longseq_validation_llama3_packed_22256/manifest.json).
It contains the first 936 windows from the deterministic validation stream.
It is a fixed evaluation subset, not a random sample of all validation documents.

| Quantity | Main artifact | Test artifact |
| --- | ---: | ---: |
| Sequence length | 22,256 | 22,256 |
| Sequences | 936 | 8 |
| Physical positions | 20,831,616 | 178,048 |
| Valid labels | 20,830,867 | 178,040 |
| Ignored boundary labels | 749 | 8 |
| Padding positions | 0 | 0 |

The artifact manifest records tokenizer hashes and array hashes.
The main manifest SHA256 is `5a9b5164709bf177bc5b09fb3a0561313371efab73c959c639f0526c26f72fd4`.
The source manifest SHA256 is `4504314f51733ce085991de14b11ef595424d1a006b1ebb2475e4fc3d29579db`.
Its provenance file records every input fragment and successor fragment.
The artifact does not depend on the masking value.
Both modes can evaluate the same tokens.

The validation loss denominator is the global valid label count.
The trainer measures long loss on train data and validation loss on held-out data.
These losses use separate metrics and the same processing rules.

## Metric names

| Metric | Definition |
| --- | --- |
| `loss_metrics/long_avg_loss` | Long loss averaged over valid labels across ranks. |
| `loss_metrics/long_max_loss` | Maximum rank-local mean long loss. |
| `loss_metrics/short_seq_loss` | Crop loss used for gradient computation. |
| `validation_metrics/loss` | Mean loss over valid held-out labels. |
| `packing/document_masking` | One for enabled masking, zero for disabled masking. |
| `packing/parent_fragments` | Number of document fragments in the parent batch across ranks. |
| `packing/mean_fragment_positions` | Parent positions divided by the fragment count. |
| `packing/ignored_parent_labels` | Parent positions that do not contribute to long loss. |

The logger replaces the old `global_avg_loss` and `global_max_loss` keys for this trainer.
Historical W&B records retain their original keys.
Console formatting remains native TorchTitan formatting.

## Checks

The final CPU regression run executed 61 tests.
It passed 60 tests and skipped one distributed test.
Eight packing tests include all six configurations at DP sizes 1, 2, 4, and 8.
The packing checks also cover recovery after a batch shape change and rejection after a masking change.
The two-rank GPU check also passed distributed normalization with unequal valid label counts.

Commands:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python -m unittest tests.test_document_packing
OMP_NUM_THREADS=1 PYTHONPATH=. python -m unittest tests.test_loss_grad_seq tests.test_loss_grad_validation tests.test_fixed_validation tests.test_seq_grad tests.test_longseq_loss_grad_schedules
OMP_NUM_THREADS=1 PYTHONPATH=. python -m unittest tests.test_long_documents
python -m scripts.prepare_packed_configs
```

Use `/home/kangan/.conda/envs/torchtitan-macro/bin/python` for these commands on this server.
The regression launcher imports SQLite before Arrow to select the environment's C++ runtime.
The first launcher failed because multiprocessing cannot restore a main module from standard input.
A file-based launcher removed that invocation error.
The subsequent regression run passed.

Evidence: [CPU regression log](../../outputs/document_packing/cpu-tests-final.log),
[packing test log](../../outputs/document_packing/packing-tests.log),
and [schedule totals](../../outputs/document_packing/schedule_plan.json).

## GPU checks

Functional job `1482275` passed on two B200 GPUs at `owl-sr003`.
It used account `jason_optai-paid` and QoS `owl_b200_normal_short`.
It requested 35 minutes of walltime and ran for 4 minutes 38 seconds.
It started at 00:38:48 EST and ended at 00:43:26 EST on 2026-11-13.
Its queue wait was 9 minutes 21 seconds.
I cancelled the superseded A100 job `8665263` before execution.

The small Llama model passed these attention checks:

- Packed masked outputs matched separate-document outputs. Maximum absolute error was `2.862e-6`.
- Packed masked gradients matched separate-document gradients. Maximum absolute error was `7.451e-8`.
- Changes to document A did not change document B's masked outputs or gradient contribution.
- The unmasked path matched the independent SDPA causal reference.
- Changes to document A changed document B's unmasked outputs. Maximum absolute change was `0.18535`.
- Future token changes did not affect earlier outputs in either mode.

Output tolerance was `rtol=2e-4, atol=2e-5` for the masked reference.
Gradient tolerance was `rtol=3e-4, atol=2e-5`.
The unmasked reference used `rtol=3e-4, atol=2e-5`.
See the [attention results](../../outputs/document_packing/gpu-1482275/attention.json).

Both trainer modes passed four updates with a 22,256-token parent.
The crop lengths were 2,048, 8,192, 4,096, and 8,192.
Global batch 10 exercised partial microbatches on two ranks.
Both modes evaluated exactly 178,040 valid labels at steps 1 and 4.
Neither mode reported an allocation retry or an out-of-memory error.

| Test | Peak active GiB | Peak reserved GiB | Step 4 gradient tokens/s per GPU |
| --- | ---: | ---: | ---: |
| Masking enabled | 6.392 | 8.145 | 113,425 |
| Masking disabled | 6.393 | 8.994 | 107,965 |

These values come from rank zero's metrics.
The throughput includes long forward work in the elapsed time.
The first updates include compilation cost and have much lower throughput.
These short checks do not measure steady-state full-run throughput.

The masked checkpoint restart loaded step 2 and completed steps 3 and 4.
Long loss, short loss, and gradient norm matched the uninterrupted values exactly in this check.
The assertions permit `rtol=2e-5, atol=2e-5`.
The resumed validation loss also matched the uninterrupted validation loss.

W&B test runs:

- [Masking enabled](https://wandb.ai/dynamic_seq_length/dynamic-seq-bs-debug/runs/3645e21af68447348e88daa429c5dc8c).
- [Checkpoint restart](https://wandb.ai/dynamic_seq_length/dynamic-seq-bs-debug/runs/f49c3fe1c06e4373a3afeecb8c50a5f2).
- [Masking disabled](https://wandb.ai/dynamic_seq_length/dynamic-seq-bs-debug/runs/bcdfc4e8c92c41e4a9ff9e9b4b07f0cc).

The [test script](../../outputs/document_packing/test.sbatch) records the GPU commands.
Submission used `-M owl --partition=b200_normal_q --qos=owl_b200_normal_short` to override its A100 defaults.
The [functional job log](../../outputs/document_packing/test-1482275.log) records compilation and distributed checks.

Capacity job `1482313` passed with the same account and QoS.
It requested two B200 GPUs and 15 minutes of walltime.
It ran for 1 minute 37 seconds, after a 59-second wait that included the predecessor dependency.
It checked steps 1 and 161 of the inverse schedule with global batches 384 and 1,536.
The corresponding crop lengths were 8,192 and 2,048.
The corresponding gradient microbatches were 4 and 16.
Both steps used long microbatch 2.
Peak active memory was 7.014 GiB. Peak reserved memory was 8.771 GiB.
The job recorded zero allocation retries and zero out-of-memory errors.
Rank zero recorded 35,119 and 52,432 gradient tokens per second per GPU.
These measurements include first-use compilation and do not establish full-run throughput.

See the [capacity result](../../outputs/document_packing/gpu-1482313/capacity/result.json),
[capacity script](../../outputs/document_packing/capacity.sbatch),
and [capacity W&B run](https://wandb.ai/dynamic_seq_length/dynamic-seq-bs-debug/runs/b2e297610417430f9848a643391241df).

See the [job record](../../outputs/document_packing/jobs.json).
Resource selection follows [ARC Slurm options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html)
and the [Owl resource documentation](https://www.docs.arc.vt.edu/resources/compute/01owl.html).

## Limits and remaining work

- The implementation supports FlexAttention with single-GPU or replicated data parallel training.
- The reader preserves the existing deterministic document order and rank assignment.
- The validation builder requires a positive sequence count that the source can supply.
- Validation sequence counts must divide by the data parallel size.
- Best-fit packing and additional single-document validation subsets remain outside this change.
- The short GPU tests do not establish convergence or full-run duration.
- Actual full-run gradient budgets require runtime valid label counters.

The writing linter found zero issues in the plan, README, and report.
All local Markdown links resolve.
This report follows the project STE writing rules and uses the terms defined in the plan.
The writing linter does not check the complete ASD approved dictionary.
