# ARC smoke submission record

Date: 2026-11-18. All observation times below use UTC unless a row says otherwise.
This record covers cluster preparation and the completed GPU smoke.
For a new installation, use the [main guide](../codebase_kick_off/getting_started.md).

## Scope

Use the native TorchTitan Llama3 1B model with random initialization.
Use sequence lengths 2048, 4096, 8192, and 16384. Train two optimizer steps at each length.
Keep the global batch size at 512 sequences. The complete smoke consumes 31,457,280 nominal training tokens.
Use all 936 fixed validation windows at length 16384. Validation contains 15,335,424 prediction targets.
This run sent training metrics to the W&B project `rotation_optimizer/debug`.

The smoke saves complete checkpoints at steps 2, 4, 6, and 8.
The formal experiment saves ten checkpoints at successive 10% token milestones.
Eight optimizer steps cannot produce ten distinct checkpoints.

The recovery check stops after step 4. It resumes through the length change to 8192.
It compares the final checkpoint with an uninterrupted eight-step run.

## Source example

The user supplied this local example:

```text
/home/kangan/titanprecond-lab/rotation_optimizers/torchtitan_rotation_optimizers/slurm_run_config/llama3_430m_muon_adam.yaml
```

The example requests four H200 GPUs, 32 CPUs, and seven hours on one node.
It uses `jason_optai-paid`, `h200_normal_q`, and `tc_h200_normal_short`.
Its model, tokenizer, optimizer, and experiment settings do not define this smoke.

## Official ARC references

ARC requires an account and an explicit GPU request for a GPU job.
See [Slurm options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html).

The H200 nodes have eight GPUs per node. Each H200 has 141 GB of device memory.
See the [GPU resource guide](https://www.docs.arc.vt.edu/resources/gpu.html).

The H200 base QoS has a usage factor of 1. The short QoS has a usage factor of 2.
The short QoS gives higher scheduling priority. It does not guarantee an immediate start.
See the [TinkerCliffs configuration](https://www.docs.arc.vt.edu/resources/compute/00tinkercliffs.html).

Use project storage for files that must remain after the job.
ARC deletes local `TMPDIR` files when the job ends.
See the [storage guide](https://www.docs.arc.vt.edu/resources/storage.html).

The submission file retains the default Slurm modules. It does not run `module purge`.
See the [module guide](https://www.docs.arc.vt.edu/usage/modules.html).

## Live observations

| UTC time | Check | Result |
| --- | --- | --- |
| 07:51:34 | `sinfo` and `scontrol` | `h200_normal_q` was UP. Six nodes exposed 48 H200 GPUs. |
| 07:54:57 | User association | `kangan` could use `jason_optai-paid` with H200 base and short QoS. |
| 07:54:57 | QoS limits | Base allowed seven days. Short allowed one day. Usage factors were 1 and 2. |
| 07:54:57 | `tc-xe006` detail | `scontrol` reported MIXED+PLANNED. Slurm allocated all eight GPUs. |
| 07:55:28 | Partition inventory | `sinfo` showed five ALLOCATED nodes and one MIXED+PLANNED node. No H100 partition appeared. |
| 07:55:28 | CPU inventory | `normal_q` was UP. Mixed nodes exposed 2623 idle CPUs without the PLANNED state. |

The visible H200 job list did not expose job end times.
The observations above do not guarantee later availability.

The project filesystem reported 3.6 PiB of free space.
This value describes the whole filesystem. It does not establish the project quota.
The quota command did not execute because automatic approval review timed out twice.
The project quota remains unverified.

During this audit, Slurm accepted both resource requests in test-only mode:

| Request | Scheduler forecast, America/New_York | Node |
| --- | --- | --- |
| Four H200s, base QoS, 32 CPUs, 256 GiB, four hours | 2026-11-19 13:16:12 | `tc-xe005` |
| CPU base QoS, 32 CPUs, 128 GiB, two hours | 2026-11-18 03:44:38 | `tc024` |
| Four H200s, short QoS, 32 CPUs, 256 GiB, two hours | 2026-11-19 13:16:12 | `tc-xe005` |
| One H200, base QoS, 8 CPUs, 64 GiB, four hours | 2026-11-19 09:24:35 | `tc-xe006` |
| Two H200s, short QoS, 16 CPUs, 128 GiB, four hours | 2026-11-18 04:47:56 | `tc-xe006` |

The two-GPU query ran at 09:29:59 UTC. Slurm reported that this request could preempt job 8359537.
The scheduler controls preemption. No manual cancellation occurred.
Use the two-GPU request if a fresh scheduler query still supports it.

These are scheduler estimates. Test-only mode did not submit jobs.
The four-hour GPU limit is a conservative request. It is not a measured runtime.

## Submitted CPU job

Job `8795433` runs the standalone data preparation on `tc093`.
It uses `normal_q`, `tc_normal_base`, and `jason_optai-paid`.
It requests 32 CPUs, 128 GiB of memory, and a two-hour time limit.

| Event | UTC time |
| --- | --- |
| Submission | 2026-11-18 09:43:09 |
| Actual start | 2026-11-18 09:43:16 |
| RUNNING observation | 2026-11-18 09:43:50 |
| Time-limit endpoint | 2026-11-18 11:43:16 |

The actual queue wait was seven seconds.
The time-limit endpoint does not predict completion.
The job passed five data preparation tests before the download.

The Slurm log is `local/slurm_logs/data-8795433.log` in the working repository.
The persistent pipeline log is `logs/data_handoff_8795433.log` below the handoff directory.
The submission uses `--wrap='bash /absolute/path/scripts/data_handoff_cpu.sh'`.
This preserves the original script path for its repository lookup.

The first submission review timed out before dispatch. One retry returned the job number.
The data agent will report the one-shard benchmark and complete-cache checks.

The complete text download passed all 236 file hashes at 10:08:43 UTC.
One shard produced 128,002,324 tokens in 67.095 seconds on `tc093`.
This measured rate was 1.908 million tokens per second with eight tokenizer threads.
The first ideal estimate assumed four workers could maintain that rate.
Later measurements showed slower parallel throughput.

Slurm denied a request to increase job `8795433` to two hours and thirty minutes.
The denial was `Access/permission denied for job 8795433`.
The running job retained its two-hour limit.

Continuation job `8796361` waits for `8795433` with `afterany`.
It requests the same CPU and memory resources for one hour.
It does not repeat the download or the benchmark.
It checks each existing source, token file, and index before reuse.
It publishes the readiness receipt only after all checks pass.

## GPU and bootstrap submissions

The independent source passed 211 tests with no failures or errors.
Five tests reported expected skips. Distributed checks also passed.
An independent W&B read check passed for `rotation_optimizer/debug`.

GPU job `8796064` uses two H200 GPUs, 16 CPUs, and 128 GiB of host memory.
It uses `tc_h200_normal_short` with a four-hour limit.
Its dependency is `afterok:8796361`.
The driver measures microbatches before the uninterrupted and resumed smoke runs.

The frozen source is `checkout-v2` below the handoff directory.
Its snapshot commit is `fe7e5f69c6c38a2b727a8e509c47205c4c81e75a`.
The source bundle SHA256 is `b290a34f273ba6a91f43912fdc8aff88ab3a9bcfd4e31aaf2cf2923f4652fa7b`.
The independent environment is `environment/conda` below the same directory.

| Job | Purpose | Submit, UTC | Actual start, UTC | Requested limit | State at 10:32:07 UTC |
| --- | --- | --- | --- | --- | --- |
| 8795433 | Download and first cache rebuild | 09:43:09 | 09:43:16 | 02:00:00 | RUNNING on `tc093` |
| 8796064 | Probe and GPU acceptance | 10:13:59 | Not started | 04:00:00 | PENDING |
| 8796354 | Independent Python bootstrap | 10:29:46 | 10:29:47 | 00:10:00 | RUNNING |
| 8796361 | Checked cache continuation | 10:30:59 | Not started | 01:00:00 | PENDING |

The bootstrap job requests one CPU and 4 GiB of memory.
It uses a newly installed Miniforge manager, a separate package cache, and a separate Python prefix.
It does not change the active training environment.
The job completed on `tc130` at 10:33:17 UTC with exit code `0:0`.
Its actual queue wait was one second. Its execution time was three minutes and thirty seconds.
The independent check reported Python 3.12.14, the expected prefix, and the `ssl` and `sqlite3` imports.

The GPU log is `logs/gpu-8796064.log` below the handoff directory.
The profile is `probe/microbatch_probe.json`.
The acceptance output directory is `acceptance`.

The current repository helper supports the generic dataset preparation commands.
Its gate checks `rebuild_progress.json`, the fixed manifest identities, and validation arrays.
The `standalone_ready.json` receipt is optional for that helper.
Isolated gate checks passed without changing the original datasets.
They covered an absent optional receipt, an incomplete rebuild, and a conflicting receipt.

The queued job captured the earlier helper from frozen `checkout-v2`.
That helper requires `standalone_ready.json`. The checked continuation creates this receipt.
The frozen checkout remained unchanged after submission.

## Hopper attention dependency

The fresh environment initially lacked `flash_attn_interface`.
A direct import raised `ModuleNotFoundError` before any GPU allocation started.
The pinned TorchTitan code selects FlashAttention 3 on Hopper GPUs.
The pinned PyTorch FA3 registration code imports `flash_attn_interface`.
The CPU import suite did not exercise this GPU-selected construction path.

The root agent reviewed this failure. GPU job `8796064` entered a user hold.
The hold preserves the queued job and its data dependency.
The environment agent owns the pinned FA3 installation and checks.
The model and attention backend remain unchanged.

At 11:15:14 UTC, the live ARC module check found `CUDA/13.0.2`.
Its toolkit directory is `/apps/common/software/CUDA/13.0.2`.
Its `nvcc` reports CUDA release 13.0 and compiler version 13.0.88.
The default `gcc` and `g++` report version 11.5.0.
These observations describe this ARC host. Another server must check its own compiler and toolkit.

The environment update installed the official PyTorch `flash-attn-3` 3.0.0 wheel.
Its SHA256 is `44a019ca428d0866fc73ca49b8af02eb4b6a37ae8a1a6b1a07917c383a187fe6`.
The update also installed `ninja` 1.13.0.
The independent check passed module import, FA3 activation, forward/backward operator registration, and `pip check`.
Torch and TorchTitan retained their original pinned versions. No compiler job was necessary.
The GPU hold ended at 11:21:45 UTC. The data dependency remained active.
The later GPU probe passed real forward and backward checks.

## Complete data preparation

The first data job completed successfully at 11:38:10 UTC on `tc093`.
Its execution time was 1:54:54. Its requested limit was 2:00:00.
The full reconstruction itself took 5295.83 seconds, after download and the first-shard benchmark.

The continuation became eligible and started at 11:38:11 UTC on `tc093`.
It completed at 11:40:01 UTC with exit code `0:0`.
Its dependency wait was 1:07:12. Its scheduler wait after eligibility was zero seconds.
Its execution time was 1:50, against a requested one-hour limit.
It checked and reused all 236 shards. It encoded zero shards.

Independent data checks passed at 11:41:09 UTC.
The checks covered every index, both order permutations, and all fixed validation windows.
The checks reconstructed validation windows from the new token cache.
They checked labels, positions, and nonoverlap, including each lookahead token.

The current helper's read-only gate also passed on the complete standalone dataset.
The result contains 936 validation windows and 15,335,424 prediction targets.

GPU job `8796064` became eligible at 11:40:02 UTC.
The scheduler initially forecast 12:48:33 UTC while the job waited for resources.
New 60-minute and 90-minute test-only requests gave a later forecast of 13:28:33 UTC.
The existing four-hour GPU request remained unchanged.

## Actual GPU execution

GPU job `8796064` started at 11:57:27 UTC on `tc-xe006`.
The actual scheduler wait after eligibility was 17:25.
The total interval from submission to start was 1:43:28, including dependency waits.
The requested execution limit remains four hours.

The allocation contains two NVIDIA H200 GPUs.
`nvidia-smi` reports 143771 MiB per GPU and driver version 595.71.05.
The runtime reports Torch 2.15.0.dev20260830+cu130 and CUDA 13.0.

The GPU probe passed 21 candidate measurements.
Real forward and backward passes succeeded at all four sequence lengths.
The probe used the native model with 1,235,814,400 parameters.
It performed zero optimizer updates.

| Sequence length | Selected microbatch per GPU | Accumulation groups with two GPUs | Peak reserved memory in the probe |
| --- | --- | --- | --- |
| 2048 | 32 | 8 | 73,219,964,928 bytes |
| 4096 | 16 | 16 | 73,219,964,928 bytes |
| 8192 | 8 | 32 | 73,219,964,928 bytes |
| 16384 | 4 | 64 | 73,219,964,928 bytes |

The measured device capacity was 150,111,977,472 bytes.
Each selected candidate used 9,886,515,200 bytes for persistent optimizer state.
The selection also reserved 7,784,628,224 bytes for estimated optimizer temporary memory.
Larger tested candidates fit in memory but failed the required reserve.
The complete acceptance run also passed real distributed optimizer steps.

The uninterrupted smoke started with online W&B enabled.
Its run is [smoke-uninterrupted](https://wandb.ai/rotation_optimizer/debug/runs/baa83f28bed34c2fbd4ca05168a42570).
Complete checkpoint metadata exists for steps 2, 4, 6, and 8 in both cases.
Training progress appears in the run's `structured_logs` directory and online W&B.
The console log can buffer training messages until the process ends.

A bounded diagnostic job, `8797605`, requested one H200 for 30 minutes.
It used the sanitized `checkout-v3` and a maximum microbatch of one.
The main job started before this diagnostic could run.
The root agent authorized cancellation of the redundant pending diagnostic.
The scheduler cancelled job `8797605` at 12:03:38 UTC. It had zero execution time and no assigned node.
Its submission time was 11:57:21 UTC.
The 47 package Python files and probe entry match between `checkout-v2` and `checkout-v3`.

## Final acceptance result

GPU job `8796064` completed with exit code `0:0` at 12:26:02 UTC.
Its actual execution time was 28:35, against a requested four-hour limit.
The final accounting query completed at 12:53:56 UTC after an approval-review timeout.
The timeout affected the status query. It did not affect the GPU job.

| Job | Final state | Exit code | Execution | Requested limit | Node |
| --- | --- | --- | --- | --- | --- |
| 8795433 | COMPLETED | 0:0 | 1:54:54 | 2:00:00 | `tc093` |
| 8796361 | COMPLETED | 0:0 | 0:01:50 | 1:00:00 | `tc093` |
| 8796354 | COMPLETED | 0:0 | 0:03:30 | 0:10:00 | `tc130` |
| 8796064 | COMPLETED | 0:0 | 0:28:35 | 4:00:00 | `tc-xe006` |
| 8797605 | CANCELLED before start | 0:0 | 0:00:00 | 0:30:00 | None |

The uninterrupted case completed eight optimizer steps.
The recovery case stopped after step 4, then resumed through the 4096-to-8192 length change.
Both cases saved complete checkpoints at steps 2, 4, 6, and 8.
Both cases used global batch size 512 throughout training.

The final checkpoint comparison checked 489 tensors and 1379 other values.
Every value matched exactly. The comparison reported no differences.
The resumed run reached a peak reserved memory of 75.64 GiB.
Its final validation completed at 12:24:37 UTC.

Independent online checks passed for all three W&B segments:

- [Uninterrupted steps 1–8](https://wandb.ai/rotation_optimizer/debug/runs/baa83f28bed34c2fbd4ca05168a42570)
- [Recovery case, steps 1–4](https://wandb.ai/rotation_optimizer/debug/runs/14773bfb9c9f483ea1fd194ca305e0d7)
- [Recovery case, resumed steps 5–8](https://wandb.ai/rotation_optimizer/debug/runs/a8e9673e9b20417ebb944074e0720cc5)

Both cases reached 31,457,280 nominal training tokens and 31,455,900 effective prediction targets.
The difference is 1380 masked positions. Nominal tokens remain the loss-plot x-axis.
The original `global_avg_loss` and `global_max_loss` metrics remain unchanged.
Validation loss also uses cumulative nominal training tokens on its x-axis.

The native validator ran at steps 1 and 8 in each case.
Every validation used 936 windows and 15,335,424 prediction targets.
The final validation loss was 10.225661943101475 in both cases.
Training and validation losses matched exactly across the uninterrupted and recovery cases.

The acceptance files are below `acceptance/` in the handoff directory:

- `acceptance.json` records the driver result.
- `checkpoint_comparison.json` records the exact state comparison.
- The phase logs record the interruption and recovery commands.

The driver receipt predates the independent online check.
The [independent W&B receipt](fresh_smoke_wandb_20261118.json) resolves its pending online-check field.
See the [independent timing receipt](fresh_smoke_timing_20261118.json) for phase timings.

The GPU run used frozen `checkout-v2`.
The sanitized `checkout-v3` has identical package Python files and active smoke commands.
Hash comparisons also passed for `train.py`, `gpu_smoke.py`, `probe_microbatch.py`, `compare_checkpoints.py`, and `configs/smoke.yaml`.
Use the sanitized handoff source for collaborator installation.

## Files and storage

Use this persistent base directory for the current ARC check:

```text
/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/
```

Keep these items below that directory:

```text
assets/                 Standalone dataset and tokenizer
environment/conda/      Independent Python environment
checkout-v2/            Frozen source for acceptance
logs/                   Slurm output
probe/                  Candidate measurements and logs
acceptance/             Training runs, checkpoints, and comparisons
wandb/                  Local W&B files
```

The data preparation agent estimated 52.03 GB of text and 120.25 GB of token caches.
These values use the pinned dataset manifest. They exclude checkpoints and environment packages.
The dataset revision is `b6ea2fde77bab36dff963aba5e6f482477959933`.

## Memory probe

Run `scripts/probe_microbatch.py` inside a GPU allocation.
The script tests powers of two that divide `512 / data_parallel_degree`.
Each candidate uses a separate process. An out-of-memory result cannot corrupt the next candidate.

The probe uses the real ordered dataset and the production model construction path.
It preserves the attention backend, loss, precision, and activation checkpoint settings.
It allocates AdamW moment tensors without an optimizer update.
It accumulates gradients across two real microbatches.
It records allocated memory, reserved memory, GPU capacity, runtime, and prediction targets.

The probe requires a 20% memory reserve by default.
It also reserves two parameter-group tensors for estimated AdamW temporary memory.
This estimate does not measure an optimizer update or distributed communication.
The complete distributed smoke must check the selected values.

The probe disables W&B and checkpoint writes for this diagnostic only.
The training smoke requires online W&B and complete checkpoints.
Do not use probe token counts as training progress.

## Submission procedure

Run the following commands from the frozen checkout.
Replace the checkout path if the acceptance copy has a different location.

```bash
export HANDOFF_ROOT=/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118
export HANDOFF_CHECKOUT="$HANDOFF_ROOT/checkout-v2"
export HANDOFF_VENV="$HANDOFF_ROOT/environment/conda"
export DYNAMIC_SEQ_LEN_ASSETS="$HANDOFF_ROOT/assets"
mkdir -p "$HANDOFF_ROOT/logs" "$HANDOFF_ROOT/probe"
cd "$HANDOFF_CHECKOUT"
bash -n cluster/vt_arc/smoke_handoff.sbatch
```

Check the scheduler forecast before submission:

```bash
sbatch --test-only --clusters=tinkercliffs \
  --gpus-per-node=2 --cpus-per-task=16 --mem=128G --qos=tc_h200_normal_short \
  --output="$HANDOFF_ROOT/logs/probe-%j.log" \
  cluster/vt_arc/smoke_handoff.sbatch \
  scripts/probe_microbatch.py --dp-degree 2 \
  --output "$HANDOFF_ROOT/probe/microbatch_probe.json"
```

Submit the probe only when the environment, source, and token cache are complete:

```bash
sbatch --parsable --clusters=tinkercliffs \
  --gpus-per-node=2 --cpus-per-task=16 --mem=128G --qos=tc_h200_normal_short \
  --output="$HANDOFF_ROOT/logs/probe-%j.log" \
  cluster/vt_arc/smoke_handoff.sbatch \
  scripts/probe_microbatch.py --dp-degree 2 \
  --output "$HANDOFF_ROOT/probe/microbatch_probe.json"
```

The JSON must contain `status: passed` and a `microbatches` entry for each length.
Read the candidate logs if a measurement fails.

Submit the acceptance run after the probe passes:

```bash
sbatch --parsable --clusters=tinkercliffs \
  --gpus-per-node=2 --cpus-per-task=16 --mem=128G --qos=tc_h200_normal_short \
  --output="$HANDOFF_ROOT/logs/acceptance-%j.log" \
  cluster/vt_arc/smoke_handoff.sbatch \
  scripts/gpu_smoke.py \
  --profile "$HANDOFF_ROOT/probe/microbatch_probe.json" \
  --assets "$DYNAMIC_SEQ_LEN_ASSETS" \
  --output "$HANDOFF_ROOT/acceptance" --nproc 2
```

Use a new acceptance directory for each independent check.
Do not edit the frozen source while either job is pending or running.

To run both phases in one allocation, add `--probe` to `scripts/gpu_smoke.py`.
The driver measures microbatches before it starts either training run.
An `--dependency=afterok:DATA_JOB_ID` submission can wait for successful data preparation.
Replace `DATA_JOB_ID` with the current data job number.
The current helper requires a complete rebuild and the fixed validation checks.
The captured helper for job `8796064` also requires `standalone_ready.json`.

Inspect an actual job with these commands. Replace `JOB_ID` with the returned number.

```bash
squeue --clusters=tinkercliffs --jobs=JOB_ID
squeue --clusters=tinkercliffs --jobs=JOB_ID --start
scontrol --clusters=tinkercliffs show job JOB_ID
sacct --clusters=tinkercliffs --jobs=JOB_ID \
  --format=JobID,State,ExitCode,Submit,Start,End,Elapsed,Timelimit,AllocTRES
```

Record the submit time, actual start, elapsed runtime, and requested time separately.
Do not report a scheduler forecast as an actual start.
The `--start` query lists pending jobs. Use the query without `--start` for running jobs.

## Current checks

- `bash -n cluster/vt_arc/smoke_handoff.sbatch` passed.
- Python bytecode compilation for `scripts/probe_microbatch.py` passed.
- `python3 scripts/probe_microbatch.py --help` passed without training dependencies.
- Slurm test-only accepted the GPU and CPU requests.
- The independent environment parsed the production smoke configuration.
- The AdamW state check allocated zero moments without changing a test parameter.
- The STE linter reported zero hard violations before this result update.
- The GPU probe passed on the native model and real dataset.
- The independent environment passed import checks and `pip check`.
- CPU jobs `8795433` and `8796361` completed with exit code `0:0`.
- All standalone data checks passed.
- GPU job `8796064` completed the uninterrupted and recovery acceptance checks.
- All 489 checkpoint tensors and 1379 other values matched exactly.
- Independent online W&B checks passed for all three segments.

Another server must provide its own submission example before using its scheduler.
Replace ARC account, partition, QoS, GPU requests, and storage paths from that example.
Retain the measured microbatch procedure and the fixed global batch size.
