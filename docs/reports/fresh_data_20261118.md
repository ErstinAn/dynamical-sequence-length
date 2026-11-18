# Standalone dataset preparation record

Standalone dataset preparation passed.
The new environment reconstructed all 236 cache shards from downloaded text.
All source, token, index, order, validation, and tokenizer hashes passed.
An independent check reconstructed every fixed validation window from the new cache.
See [the structured record](fresh_data_20261118.json) for exact counts and hashes.

The dataset repository is `long-sequence-length-dataset/fineweb-edu-long-30b`.
The authenticated Hub API returned commit `b6ea2fde77bab36dff963aba5e6f482477959933`.
The check ran at `2026-11-18T07:54:43Z`.
All download commands use this fixed commit.

The standalone asset root is `/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/assets/`.
The dataset is under `data/fineweb_edu_long_30b_shuffled_v1/`.
The tokenizer is under `hf/llama3-tokenizer/`.
These paths do not use the original dataset's directory links.
The original data package remains unchanged.

## Metadata and tokenizer checks

| Check | Result |
|---|---|
| Dataset manifest | Matched `7e284ce4fc90a93e5bdcee160a60051dc29ed17775b8ecb261feafd93f047d80` |
| Training manifest | Matched `4bf6946f7b7f7cbad0799746519baaa9ef2518f974d2239a8df2f184db004806` |
| Fixed validation manifest | Matched `eb2faf62a36e324d07e08da73170e1268182cdfea1f38352362b86232f47155b` |
| Training document order | Complete permutation of 1,348,067 documents |
| Validation document order | Complete permutation of 2,725 documents |
| Fixed validation labels | 936 by 16,384 positions, with 15,335,424 valid targets and zero ignored labels |
| Tokenizer | All three file hashes matched the pinned provenance |

The tokenizer repository is `meta-llama/Meta-Llama-3-8B`.
Its fixed commit is `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`.
The download did not include model weights.

The manifest contains 236 text shards and 52,027,268,242 text bytes.
The new token binaries contain 120,249,458,012 bytes.
These sizes exclude indexes, metadata, download caches, and training checkpoints.
The complete dataset directory contains 172,672,635,493 logical bytes, including indexes and fixed validation arrays.
The tokenizer directory contains 9,139,915 logical bytes, including local download metadata.
The data and token directories are real directories.
All their files are regular files.
This data test did not establish the project quota.

## Execution scope and cache reuse

The preliminary metadata download used the existing `torchtitan-macro` environment as a bootstrap client.
It used new Hub and local download directories.
It did not explicitly isolate the Xet chunk cache.
Possible reuse of existing Xet chunks cannot be excluded for that preliminary download.
Fresh Conda Python then ran the complete download, reconstruction, and independent checks.
The data CPU jobs used the current repository checkout and recorded their script hashes.
The test reused the host operating system, Slurm installation, and authentication files.

The new `scripts/download_dataset.py` isolates both its Hub cache and Xet cache.
It downloads one fixed revision and checks all text, order, validation, and tokenizer hashes.
It rejects links at the dataset, data, token, and tokenizer destinations.
It writes `download_receipt.json` after a successful download.
It does not rebuild token caches or claim that the dataset is ready for training.

Five tests passed in the new Conda environment in 1.135 seconds.
They checked metadata hashes, linked destinations, parallel encoding, and interrupted cache reconstruction.
The encoding tests used an independent token and index oracle.
The tests also checked the complete manifest bytes and document order.
They rejected changed source text and changed document order.
They regenerated corrupt tokens and missing indexes before reuse.
These tests used small synthetic files and did not replace the real-data checks above.

The first two full-download attempts did not execute.
Automatic permission review timed out before dispatch on the first attempt and its permitted retry.
The reviewer did not state a safety rejection.
The later CPU allocation executed the full download successfully.

## Full download command

Run from the repository root with the selected isolated Python environment:

```bash
python scripts/download_dataset.py \
  --dataset-dir /projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/assets/data/fineweb_edu_long_30b_shuffled_v1 \
  --tokenizer-dir /projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/assets/hf/llama3-tokenizer \
  --cache-dir /projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/download-cache \
  --revision b6ea2fde77bab36dff963aba5e6f482477959933 \
  --max-workers 4
```

After download, run the repository's cache builder on a CPU compute allocation.
Use the same dataset and tokenizer directories.
Measure the first shard before estimating the complete reconstruction time.
Do not substitute a subset or existing token cache for the full reconstruction without recording that limitation.

## CPU preparation command

The tested wrapper is `scripts/data_handoff_cpu.sh`.
The test used a Slurm allocation with 32 CPUs and 128 GiB of memory.
It uses the new Python environment under `environment/conda/` in the handoff directory.
It repeats the five tests before the download.
It measures one complete shard before the full reconstruction.
It uses four shard processes, with eight tokenizer threads per process.

The wrapper preserves the first benchmark in `first_shard_benchmark.json`.
The benchmark reports an ideal scaling estimate, not a guaranteed duration.

The cache builder preserves the original token bytes, index bytes, and manifest bytes.
The builder writes each token file only after the token hash passes.
The builder writes each index file only after the index hash passes.
The `--resume` option first checks the source text, token file, and index file.
An unchecked file cannot use the resume shortcut.

The `--limit-shards` option is for a benchmark only.
It does not publish a complete cache.

The wrapper writes `standalone_ready.json` only after all shards and fixed validation checks pass.
The test produced this marker before the GPU job became eligible.

Slurm started CPU job `8795433` on node `tc093` at `2026-11-18T09:43:16Z`.
The queue wait was seven seconds.
The preparation script started two seconds later.

The job used 32 CPUs, 128 GiB of memory, and a two-hour limit.
The job used `normal_q`, `tc_normal_base`, and the paid account `jason_optai-paid`.
The same five tests passed on the compute node in 8.542 seconds.
The complete text download started after those tests.

The log is `local/slurm_logs/data-8795433.log`.
The persistent log is `logs/data_handoff_8795433.log` under the handoff directory.
The job completed successfully at `2026-11-18T11:38:10Z`.
Its total execution time was 1 hour, 54 minutes, and 54 seconds.
That duration includes the download, tests, benchmark, reconstruction, and final checks.

The full download used a new Hub cache and a new Xet directory.
It reused only this session's preliminary metadata and tokenizer files.
It used the account's existing Hugging Face credentials.
The Xet transport log reported HTTP 416 range errors during the download.
The downloader recovered without intervention.
All final text hashes matched their reference values.

The first complete shard contains 128,002,324 tokens.
It took 67.095 seconds with eight tokenizer threads.
Its measured rate was 1,907,772 tokens per second.
Its token file and document index matched their reference hashes.
The full reconstruction started at approximately `2026-11-18T10:09:51Z`.
The initial ideal estimate was 65.38 minutes for four workers.
The actual full reconstruction took 5,295.83 seconds, or 88.26 minutes.
The full run reused the checked first shard from this same fresh reconstruction.
It encoded the other 235 shards.
No step reused the original machine's token cache.

The measured parallel rate was slower than the ideal estimate.
At `2026-11-18T10:31:15Z`, 48 shards passed after 1,248 seconds of reconstruction.
The scheduler refused an increase to the active job's time limit.
Continuation job `8796361` started on `tc093` at `2026-11-18T11:38:11Z`.
It completed successfully at `2026-11-18T11:40:01Z`.
Its execution time was 110 seconds.
Its dependency hold was 1 hour, 7 minutes, and 12 seconds.
It had no scheduler wait after the dependency cleared.

The continuation used 32 CPUs, 128 GiB of memory, and a one-hour limit.
It ran `scripts/resume_data_handoff_cpu.sh`.
It checked every existing source, token file, and index before reuse.
The checks took 107.20 seconds.
All 236 shards passed, so the continuation encoded zero shards.

It did not repeat the download or benchmark.
It wrote a new readiness marker after the complete dataset passed.
GPU job `8796064` became eligible after the continuation succeeded.

## Independent validation reconstruction

The independent check passed at `2026-11-18T11:41:09Z` in 5.75 seconds.
It used the new Conda Python and the new dataset directories.
See [the exact check results](fresh_data_20261118_windows.json).

All 236 document indexes and both complete document order permutations passed their checks.
Every index offset followed the preceding document without a gap.
The training cache contains 30,000,004,070 raw tokens.
The validation cache contains 62,360,433 raw tokens.
These raw corpus totals are separate from nominal training progress.

The check reconstructed all 936 windows from the new validation token file.
Each window matched its fixed input array, shifted labels, and position array.
All windows stayed within one document.
The check found 798 distinct documents, including 100 documents with multiple windows.
No windows overlapped, including their final lookahead tokens.
The total remained 15,335,424 valid targets.

Run the same read-only check from the repository root:

```bash
python docs/reports/fresh_data_20261118.py \
  --dataset-dir /projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/assets/data/fineweb_edu_long_30b_shuffled_v1 \
  --tokenizer-dir /projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/assets/hf/llama3-tokenizer
```

Require `passed: true` and `reconstructed_validation_windows: 936`.
The check requires a complete `rebuild_progress.json` with the fixed manifest hash.
It also checks `standalone_ready.json` when that optional Slurm marker exists.
The generic download and reconstruction commands do not need this Slurm marker.

A controlled repeat made only the optional marker appear absent to the process.
All checks passed again in 1.54 seconds with the same data.
The marker remained on disk.
See [the controlled repeat record](fresh_data_20261118_optional_marker.json).

This record establishes data readiness only.
The separate GPU acceptance record establishes training, validation, checkpoint, and resume outcomes.
