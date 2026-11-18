# Handoff acceptance: 2026-11-18

This record is optional evidence for the owner and reviewers.
Its local paths describe the original verification server.
The collaborator must follow [the main run guide](../getting_started.md).

Status: smoke acceptance passed.
The fresh environment, complete token reconstruction, and real 1B training passed.
The two-H200 test passed uninterrupted training and checkpoint recovery.
Independent online checks passed for all required W&B losses and token axes.
The final source packaging receipt records delivery checks separately.
This test used the same cluster service, not another physical server service.

## Checks after removing the historical configurations

The cleanup used a separate checkout and the existing handoff Python environment.
The unit suite ran 67 tests with distributed validation enabled.
All 67 passed.
A separate two-rank CPU run passed nominal token counting, ten complete saves, exact recovery, and the validation token axis.
The two-GPU smoke dry run returned 31,457,280 tokens and save steps 2, 4, 6, and 8.
The ARC submission script passed `bash -n`.
See section 6 of the [run guide](../getting_started.md#6-run-automatic-checks) for the CPU commands.

The remaining sections record the earlier hardware run.
The cleanup did not repeat GPU training.
It removed files from the launcher's source hash.
Use the checkout recorded by the original run when resuming its checkpoints.

## Tokenizer delivery check

The dataset downloader ran with `--metadata-only` in a new directory with an empty download cache.
It used the existing Python environment and private dataset credentials.
It downloaded the pinned dataset metadata and installed the tokenizer from the checkout.
All three tokenizer file hashes matched the dataset manifest.
The tokenizer reported 128,256 entries and passed an English and Chinese encoding round trip.
This check covered tokenizer delivery. The GPU results below come from the earlier smoke run.

## Authorized smoke

Use the native TorchTitan Llama3 1B model with random initialization.
The model has 1,235,814,400 parameters and vocabulary size 128,256.
Use training lengths 2048, 4096, 8192, and 16384 for two steps each.
Keep global batch size 512 throughout all eight steps.
The nominal training total is 31,457,280 tokens per run.

Use the fixed 936 validation windows of length 16,384.
Each complete validation contains 15,335,424 valid targets.
Use online W&B in `rotation_optimizer/debug`.
Preserve the original global average, global maximum, and validation loss metrics.
Use cumulative nominal tokens for all three horizontal axes.

Compare an uninterrupted run with a run interrupted after step 4.
Require exact final model, optimizer, scheduler, counter, and data reader states.
Smoke checkpoints occur at steps 2, 4, 6, and 8.
Formal experiments retain ten distinct checkpoints at 10% token thresholds.
Formal sequence schedules and learning rate sweeps remain outside this execution phase.

## Current evidence

| Check | Result | Evidence |
|---|---|---|
| New Python and Torch environment | Passed, without original installed packages | [Environment record](fresh_environment_20261118.md) |
| Installation without an existing Python manager | Standalone Miniforge and new Python creation passed | [Bootstrap guide](../environment_bootstrap.md) |
| Exact dependency versions | Pinned Torch/TorchTitan preserved, missing FA3 added from an official wheel | [FA3 record](fresh_environment_20261118_fa3.json) |
| Source acquisition | New local Git snapshot, bundle, and independent clone passed | [Source and CPU record](fresh_environment_20261118.md#independent-source-and-cpu-checks) |
| Full CPU suite | 211 tests: 206 passed, 5 skipped, no failures or errors | [Raw CPU log](fresh_environment_20261118_cpu_passed.txt) |
| Distributed reductions | Two explicit CPU checks passed | [Raw Gloo log](fresh_environment_20261118_gloo.txt) |
| Two-rank token and checkpoint test | Ten complete saves and exact recovery passed | [Raw token log](fresh_environment_20261118_token_progress_dp.txt) |
| Dataset download | All 236 text shards matched their hashes | [Data record](fresh_data_20261118.md) |
| Complete token reconstruction | All 236 shards matched original token, index, and manifest hashes | [Data record](fresh_data_20261118.md) |
| New-cache reconstruction of validation windows | All 936 windows, labels, positions, and disjoint spans passed | [Window result](fresh_data_20261118_windows.json) |
| GPU memory probe and eight-step training | Passed on two H200 GPUs | [GPU report](fresh_smoke_20261118.md) |
| GPU checkpoint recovery and exact comparison | All 489 tensors and 1,379 other state values matched exactly | [Online acceptance result](fresh_smoke_wandb_20261118.json) |
| Uploaded W&B metrics and axes | All three runs finished, original loss names and token axes passed | [Online acceptance result](fresh_smoke_wandb_20261118.json) |
| Main documentation links and text checks | Passed | [Run guide](../getting_started.md) |
| Historical file preservation | All 76 configurations and 11 archived files retained their recorded bytes | [Archive inventory](https://github.com/ErstinAn/dynamical-sequence-length/blob/427d321c40e8b82444d2f5c07082e6155c0f914e/legacy/report_checksums.json) |

The five skipped tests concern opt-in checks and unused historical artifacts.
The explicit distributed checks ran separately.
The first independent CPU attempt found a tokenizer path fixed to the original repository.
The updated preparation tool uses the external assets path and verified downloaded provenance.
The updated source then passed the full CPU suite.

## Source versions and private local evidence

The original repository base is `91ef5297deae1d0cf96d9d22869151e68bd31142`.
The independent GPU source snapshot is `fe7e5f69c6c38a2b727a8e509c47205c4c81e75a`.
Its runtime source SHA256 is `583f82e438472b24d53e8aa4f93fe6e5fd3d0131ba79b9724a72d5361d3fd47e`.
The working repository's runtime source had the same hash after documentation changes.
The delivery bundle uses this same tested runtime source.
Its separate receipt records the final commit, bundle SHA256, and fresh-clone checks.

The initial local test bundles contain an old private conversation archive.
Those bundles permit owner access only and must not serve as delivery bundles.
The conversation bytes remain in ignored local storage.
The local backup bundle uses a new Git history that excludes the private archive.
The agents did not upload an initial bundle.

The local backup files are `source-final.bundle` and `source_snapshot-final.json` under the handoff directory.
Their receipt records the completed packaging checks before GitHub publication.
The GitHub delivery uses the default `main` branch.
It preserves the existing remote history and the tested runtime source.
The GitHub documentation update identifies the document audiences and direct clone commands.
Formal training remains unexecuted.

## GPU results

The allocation used two NVIDIA H200 GPUs on `tc-xe006`, with driver `595.71.05`.
Slurm job `8796064` started at `2026-11-18T11:57:27Z`.
The complete GPU driver finished at `2026-11-18T12:26:02.540463Z`.
Slurm reported `COMPLETED` with exit code `0:0` after 28 minutes and 35 seconds.
The [ARC record](arc_smoke_plan_20261118.md) separates queue waits, requested limits, and actual execution.

| Sequence length | Steps | Global batch | Microbatch per GPU | Accumulation count |
|---|---|---:|---:|---:|
| 2048 | 1–2 | 512 | 32 | 8 |
| 4096 | 3–4 | 512 | 16 | 16 |
| 8192 | 5–6 | 512 | 8 | 32 |
| 16384 | 7–8 | 512 | 4 | 64 |

Both cases completed all eight steps and retained checkpoints at steps 2, 4, 6, and 8.
Recovery loaded step 4 and continued through the sequence length change at step 5.
The final comparison checked every stored model, optimizer, scheduler, counter, and reader value.
It reported `exact_match: true` with no differences.
Uploaded training and validation losses also matched exactly between both cases.

| Final metric | Value |
|---|---:|
| Nominal training tokens | 31,457,280 |
| Valid training targets | 31,455,900 |
| Global average training loss | 10.598470687866211 |
| Global maximum training loss | 10.605973243713379 |
| Fixed validation loss | 10.225661943101475 |
| Targets per complete validation | 15,335,424 |

The 1,380 masked training positions still count toward the nominal token budget.
All three loss axes use `n_tokens_seen`.
Validation runs at native steps 1 and 8, without advancing that counter.
Their horizontal-axis values are 1,048,576 and 31,457,280.

W&B stores three finished segments in the authorized debug project:

- [Uninterrupted steps 1–8](https://wandb.ai/rotation_optimizer/debug/runs/baa83f28bed34c2fbd4ca05168a42570).
- [Recovery case, steps 1–4](https://wandb.ai/rotation_optimizer/debug/runs/14773bfb9c9f483ea1fd194ca305e0d7).
- [Recovery case, steps 5–8](https://wandb.ai/rotation_optimizer/debug/runs/a8e9673e9b20417ebb944074e0720cc5).

Each full checkpoint occupies approximately 15.884 GB.
The paired smoke retains eight complete states.
Ten states with the same model and optimizer need approximately 158.84 GB before temporary files.
The uninterrupted GPU peak was 73.07 GiB.
The resumed process peak was 75.64 GiB.
See the [timing record](fresh_smoke_timing_20261118.json) for measured step, validation, and save times.

## Environment and data identities

| Item | Value |
|---|---|
| Python | 3.12.14 |
| Torch | `2.15.0.dev20260830+cu130` |
| TorchTitan | `737594746fda65a6d94dc9482ef07863a80c8588` |
| FlashAttention 3 | `3.0.0`, official SHA256-pinned PyTorch wheel |
| W&B | `0.29.0` |
| Dataset revision | `b6ea2fde77bab36dff963aba5e6f482477959933` |
| Training manifest SHA256 | `4bf6946f7b7f7cbad0799746519baaa9ef2518f974d2239a8df2f184db004806` |
| Fixed validation manifest SHA256 | `eb2faf62a36e324d07e08da73170e1268182cdfea1f38352362b86232f47155b` |
| Tokenizer revision | `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920` |

The selected runtime is under `environment/conda/` in the handoff directory.
Data and tokenizer files are under its standalone `assets/` directory.
The handoff directory is `/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/`.

## Reuse and limits

The test uses the same physical server service and existing operating system.
It reuses the user's Hugging Face and W&B authentication.
It does not reuse the original installed Torch packages or original training token cache.
The first Python creation used the ARC Miniforge manager with a new package cache.
A separate check created Python with the newly installed standalone manager.

The preliminary metadata download used the original environment as a download client.
Possible Xet chunk reuse during that preliminary download cannot be excluded.
The full text download used the new runtime and an isolated new Hub/Xet cache.
It reused metadata and tokenizer files downloaded earlier in this same session.
The token cache rebuild encodes the downloaded text and checks the original token and index hashes.

The current project quota remains unverified.
Filesystem free space does not establish that quota.
Official ARC references and live job observations remain separate in the ARC record.
Scheduler estimates do not establish actual start times.

## Work outside this acceptance

Formal training and learning rate sweeps did not run.
The user must supply the exact experiment document and agree to the resulting resource request.
The collaborator must supply the target server's Slurm example and persistent paths.
H100, other GPU models, different DP degrees, and multi-node execution remain untested.
The target server must run its own memory probe and smoke before formal training.

Use [getting_started.md](../getting_started.md) for all execution commands.
Use [collaborator_prompt.md](../collaborator_prompt.md) for the next agent's startup instructions.
