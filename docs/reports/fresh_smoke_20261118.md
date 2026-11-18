# Independent H200 smoke verification

## Result

The smoke test passed in a new Python and Torch environment.
Both training paths completed all eight optimizer steps.
The final checkpoints matched exactly: 489 tensors and 1,379 non-tensor values.
The comparison found no differences.

The independent online check also passed.
All three W&B segments finished successfully.
Training losses and validation losses matched exactly across the uninterrupted and recovered paths.
Read [the online result](fresh_smoke_wandb_20261118.json) and [the timing record](fresh_smoke_timing_20261118.json).

This test ran on the existing ARC server infrastructure.
It did not run on another physical server.
No formal 20B-token experiment or learning-rate sweep ran.

## Hardware and versions

| Item | Actual value |
| --- | --- |
| Slurm job | `8796064`, `COMPLETED`, exit `0:0` |
| Node | `tc-xe006` |
| GPUs | Two NVIDIA H200 GPUs |
| Reported memory | 143,771 MiB per GPU |
| Driver | 595.71.05 |
| Python | 3.12.14 |
| Torch | 2.15.0.dev20260830+cu130 |
| TorchTitan commit | `737594746fda65a6d94dc9482ef07863a80c8588` |
| FlashAttention 3 | 3.0.0, official wheel with a pinned SHA256 |
| W&B | 0.29.0 |

The test used local source snapshot `fe7e5f69c6c38a2b727a8e509c47205c4c81e75a`.
Its runtime program digest was:

```text
583f82e438472b24d53e8aa4f93fe6e5fd3d0131ba79b9724a72d5361d3fd47e
```

The delivery uses a new, sanitized Git history.
The runtime program files have the same digest.
The external delivery receipt records the final bundle hash and commit.
Do not distribute the earlier private test bundles.

The [environment record](fresh_environment_20261118.md) contains installation commands, package versions, source checks, and CPU results.
The full CPU suite ran 211 tests: 206 passed and five skipped.
It reported no failures or errors.
Separate distributed CPU checks passed.

## Data and model

The dataset was `long-sequence-length-dataset/fineweb-edu-long-30b` at revision `b6ea2fde77bab36dff963aba5e6f482477959933`.
The tokenizer revision was `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`.
The test rebuilt all 236 token-cache shards from the downloaded text.
It did not reuse the original training token cache.

All shard hashes and document-order permutations passed their checks.
An independent check reconstructed all 936 validation windows from the fresh cache.
It checked labels, positions, document boundaries, and the absence of window overlap.
Read [the data record](fresh_data_20261118.md) for complete provenance and checks.

The model was TorchTitan's native, randomly initialized Llama3 `1B` configuration.
It had 1,235,814,400 parameters and a vocabulary of 128,256 tokens.
The model shared its input and output weights.
The test kept the native Hopper FA3 attention path.

The smoke used AdamW with a constant learning rate of `0.0001`.
Model parameters used FP32 with BF16 mixed-precision computation.
Gradient reduction used FP32.
The smoke configuration disabled compilation.
These smoke settings do not select the future formal schedule or learning-rate sweep.

## Executed training paths

The memory probe completed 21 measurements with real data and the native model.
It ran forward and backward operations at all four sequence lengths.
The probe ran no optimizer updates and saved no training checkpoints.
The full distributed training then checked the selected microbatches.

| Optimizer steps | Sequence length | Microbatch per GPU | Accumulation groups | Global batch |
| --- | ---: | ---: | ---: | ---: |
| 1–2 | 2,048 | 32 | 8 | 512 |
| 3–4 | 4,096 | 16 | 16 | 512 |
| 5–6 | 8,192 | 8 | 32 | 512 |
| 7–8 | 16,384 | 4 | 64 | 512 |

The uninterrupted path ran steps 1 through 8 in one process group.
The recovered path stopped normally after checkpoint 4.
A new process group loaded checkpoint 4 and completed steps 5 through 8.
Each path saved complete checkpoints at steps 2, 4, 6, and 8.

Each path processed 31,457,280 nominal training tokens.
Each path counted 31,455,900 effective prediction targets.
The 1,380 ignored prediction positions did not reduce the nominal token counter.
Validation did not increase either training counter.

The formal checkpoint policy remains ten milestones at 10% intervals of the nominal token budget.
The eight-step smoke used four phase checkpoints.
The separate distributed CPU test checked all ten formal milestones and recovery.

The recorded GPU command ran from the private test checkout:

```bash
export TEST_ROOT=/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118
export DYNAMIC_SEQ_LEN_ASSETS="$TEST_ROOT/assets"
export PYTHONNOUSERSITE=1
cd "$TEST_ROOT/checkout-v2"
"$TEST_ROOT/environment/conda/bin/python" scripts/gpu_smoke.py \
  --probe --profile "$TEST_ROOT/probe/microbatch_probe.json" \
  --assets "$TEST_ROOT/assets" --output "$TEST_ROOT/acceptance" --nproc 2
```

The output directory did not exist before that command.
Use a new output directory when repeating the test.
Use the sanitized delivery checkout for future runs.

## W&B and validation

The online checker read these runs:

- [Uninterrupted steps 1–8](https://wandb.ai/rotation_optimizer/debug/runs/baa83f28bed34c2fbd4ca05168a42570).
- [Recovery path, steps 1–4](https://wandb.ai/rotation_optimizer/debug/runs/14773bfb9c9f483ea1fd194ca305e0d7).
- [Resumed steps 5–8](https://wandb.ai/rotation_optimizer/debug/runs/a8e9673e9b20417ebb944074e0720cc5).

The uploaded metric definitions used `n_tokens_seen` as the horizontal axis for all three loss metrics:

- `loss_metrics/global_avg_loss`.
- `loss_metrics/global_max_loss`.
- `validation_metrics/loss`.

Native TorchTitan validation runs at step 1 and at multiples of the configured validation frequency.
Therefore, this smoke ran validation at steps 1 and 8, although `validator.freq` was 8.
The test preserved that behavior.
The original expectation of validation only at step 8 was incomplete.

Every validation event used the same 936 windows of length 16,384.
Every event evaluated 15,335,424 prediction targets.
The validation token axes were 1,048,576 at step 1 and 31,457,280 at step 8.

| Step | Global average training loss | Global maximum training loss | Validation loss |
| --- | ---: | ---: | ---: |
| 1 | 12.283363343 | 12.287185669 | 13.463331996 |
| 8 | 10.598470688 | 10.605973244 | 10.225661943 |

Both training paths produced identical values at every corresponding step.
These short runs check execution and recovery.
They do not establish convergence or compare formal schedules.
The fixed validation set primarily measures long-document performance.

To check this historical run, use its original W&B project. Run this command from the repository root.

```bash
"$TEST_ROOT/environment/conda/bin/python" docs/reports/check_smoke_wandb.py \
  --entity rotation_optimizer --project debug \
  --acceptance "$TEST_ROOT/acceptance/acceptance.json" \
  --output /tmp/smoke-wandb-check.json
```

Require `"status": "passed"`.
The original acceptance-driver JSON still states that independent online verification was pending.
The separate online result records the completed check.

## Measured time and storage

The job requested four hours.
It ran from `11:57:27` to `12:26:02` UTC on 2026-11-18.
Its execution time was 28 minutes and 35 seconds.
This time included the probe, both training paths, checkpoint operations, and the final comparison.

The job became eligible at `11:40:02` UTC.
It waited 17 minutes and 25 seconds after eligibility.
The total interval from submission to start was one hour, 43 minutes, and 28 seconds.
That interval included the data dependency and the temporary dependency-check hold.
The separate diagnostic GPU job never started and used no GPU execution time.

| Operation | Uninterrupted path | Recovered path |
| --- | ---: | ---: |
| Training and garbage collection | 381.041 s | 355.500 s |
| Two validation intervals | 119.647 s | 119.333 s |
| Four checkpoint saves | 78.186 s | 77.097 s |

The validation intervals include small loop and profiler overhead.
The timing parser used rank-0 structured event timestamps.
The recovered path used the same data files after the first run.
The test did not clear filesystem caches between paths.

The uninterrupted path averaged 82,556 nominal training tokens per second, excluding validation and checkpoint saves.
A simple 20B-token extrapolation gives 67.3 training hours on the same two GPUs with the same schedule weighting.
This estimate excludes startup, validation, checkpoints, queue time, and long-run performance changes.
Do not use it as a forecast for another GPU count or another schedule.

Each full checkpoint used approximately 15.884 GB of apparent file space.
The largest checkpoint contained 15,883,544,496 bytes.
Ten such checkpoints require approximately 158.835 GB before temporary writes and other run files.
The eight smoke checkpoints together used approximately 127.068 GB.
Use the measured size instead of the earlier estimate based only on unique parameter counts.

The [timing JSON](fresh_smoke_timing_20261118.json) records each step, checkpoint size, and timing method.
Run `docs/reports/summarize_smoke_runtime.py` to reproduce that record.

## Reuse and limits

The selected Torch environment used fresh Python packages in a separate prefix.
Its initial Python creation used ARC's Conda manager.
A separate test downloaded Miniforge and created Python with that new manager and an empty package cache.
Both Python creation methods passed their checks.
The test did not install packages from the original training environment.

Repeated package-install attempts reused the new test download cache.
The initial Hugging Face metadata download did not isolate every Xet cache setting.
Possible chunk-cache reuse for that preliminary download is not excluded.
The complete token cache was rebuilt and checked independently.
Existing account credentials supplied Hugging Face and W&B access.

The test does not check H100 microbatch sizes, another server, or another GPU count.
Measure the target machine before formal training.
The formal sequence schedule, learning rate, token budget, and sweep scope still require the experiment document.
