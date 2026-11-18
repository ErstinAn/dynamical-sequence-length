# Formal packed training submission

This historical submission record does not define the current 1B experiment.
Its status observations apply to the date below.
Use [the main guide](../codebase_kick_off/getting_started.md) for the current procedure.

Date: 2026-11-13.
Status: six jobs submitted. All six jobs are pending in the latest snapshot.

## Experiment

The project is [longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler).
The entity is `dynamic_seq_length`.
The group is `packed-masked-loss22256-grad2048-8192-v1`.
Each job uses account `jason_optai-paid` and short QoS.
Each job requests 24 hours of walltime, 16 CPU cores, and 64 GiB of host memory.
Requested walltime is a limit. It is not an execution-time estimate.

## Alignment

I compared all six configurations with the six completed runs in the reference W&B project.
Optimizer settings, learning-rate schedules, seed, precision, clipping, and loss chunks match their respective reference runs.
The runs retain 3,840 steps, validation every 20 steps, and checkpoint saves every 100 steps.
Each run retains four replicated data parallel ranks.
The [alignment record](../../outputs/document_packing/formal_submission/alignment.json) identifies each reference run.

The approved packing changes remain active:

- Training and validation use stream packing and document masking.
- Long loss and validation use 22,256 positions.
- Gradient crops range from 2,048 to 8,192 positions.
- The six schedules preserve their approved trends and approximate 12.08B nominal gradient-position budgets.
- Microbatch limits and the frozen validation artifact match the tested packed configurations.

This submission changes the W&B project and resource placement.
It does not change the approved training settings.
The configuration generator now preserves the new project and resource choices.

## Jobs

The following times use America/New_York, EST on this date.
Start times are scheduler estimates, not actual starts.

| Schedule and W&B run | Slurm task | GPUs | Observed state | Estimated start |
| --- | --- | --- | --- | --- |
| [bs_increasing_seq_decreasing](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler/runs/650b1e8f4a114b40bf7f0724cd336f64) | 1482326_0 | 4 × B200 | PENDING | 2026-11-13 07:19:56 |
| [bs_decreasing_seq_increasing](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler/runs/a7cfb13fc9db44b5b59602b336ee70d2) | 1482327_0 | 4 × B200 | PENDING | 2026-11-13 14:50:22 |
| [bs_cyclic_seq_inverse](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler/runs/bf5c883ab1ce4d3e8352fdd9f94769cb) | 8665782_0 | 4 × A100 | PENDING | 2026-11-13 03:49:11 |
| [four_stage_cycle](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler/runs/9b27715db3444ccaa8dad6af07396fd1) | 8665783_0 | 4 × A100 | PENDING | 2026-11-13 04:40:49 |
| [linear_bs_cyclic_seq_cosine_lr](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler/runs/8ffaa9e67fe740849b743c7a9711085c) | 8665785_0 | 4 × A100 | PENDING | 2026-11-13 04:07:58 |
| [linear_bs_cyclic_seq_constant_lr](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler/runs/3d9574a4c3af446fb0c23d19be25f097) | 8665786_0 | 4 × A100 | PENDING | 2026-11-13 07:41:24 |

The machine-readable [job record](../../outputs/document_packing/formal_submission/jobs.json) includes the snapshot timestamp.
It also records requested walltime, observed pending time, execution elapsed time, and each configuration snapshot.
W&B run pages will contain training metrics after their jobs start.

The cluster provides A100 and B200 partitions. It does not provide an H100 partition in the current inventory.
Two B200 jobs use the eight-GPU user limit for short QoS.
Four A100 jobs can run independently of those jobs.
All runs retain four ranks to preserve the reference data parallel layout.
No duplicate training job uses the same run ID or checkpoint directory.

Resource selection follows the [ARC Slurm options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html)
and [ARC GPU guide](https://www.docs.arc.vt.edu/resources/gpu.html).
The recorded limits and start estimates come from live Slurm queries.

## Checks and recovery

All eight packing configuration tests passed after the project and resource changes.
All six saved configuration snapshots passed launcher preflight.
All six Slurm submissions returned success.
The scheduler reports the requested paid account and short QoS for every job.

The submission directory contains configuration snapshots, manifests, logs, and a [program source archive](../../outputs/document_packing/formal_submission/submitted-source.tar.gz).
The [source digest](../../outputs/document_packing/formal_submission/source-sha256.txt) identifies the submitted program files.
The launcher rejects a changed program digest.
The source archive provides a copy for recovery.

The trainer saves resumable checkpoints every 100 steps and keeps the latest two checkpoints.
If a job reaches its time limit, use its original configuration snapshot and manifest for strict recovery.
The submission tool supports `--resume-manifest` for this operation.
These submissions do not include an automatic timeout-resubmission service.

Training completion, full-run throughput, and convergence remain unverified while the jobs are pending.
