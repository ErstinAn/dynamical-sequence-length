# Paired single-document sequence-gradient monitor

This document describes historical monitor experiments and their execution records.
Use [the main guide](../../docs/getting_started.md) for the current 1B schedule experiment.
Historical commands below require their original assets and server settings.

The historical paired-monitor configuration is
`legacy/configs/seq_grad/variance_monitor_shuffled_v1_train16384_val22256.yaml`.
Each event samples 256 H=16384 parent sequences from individual raw documents
strictly longer than H. Sampling is uniform over valid H+1-token spans: documents
are weighted by `raw_length-H`. Inputs and next-token labels stay inside one
document; no BOS/EOS insertion, packing or boundary-label masking is used by
monitor batches. Their attention is causal only.

For each s, sample a uniform offset from 0 through H-s independently for each
parent, then slice aligned inputs/labels from the SAME parent batch. g_H uses
all H targets; g_s uses all targets in the short windows. Both are global token
means, and the recorded value is the raw L2 norm of their gradient difference.
The H reference is computed once per event and is not restricted to the sampled
short-window targets. Candidate/reference draws are correlated by design.

Training uses packed length 16384.
The archived configuration now uses 936 single-document validation windows of length 16384.
Its `val22256` filename remains for path compatibility.
The earlier execution records below retain their original validation results.
The dataset is `assets/data/fineweb_edu_long_30b_shuffled_v1`.
Training uses its fixed document permutation with seed 42 before the DP split.
Validation uses document seed 43 and fixed-window sample seed 44.
The trainer checks the intrinsic position limit and expands the RoPE cache.
Training global/local batches remain 32/1 on four GPUs with eight
accumulations, constant LR 5e-4, seed 1 and 3819 updates. The document-length
restriction applies ONLY to monitor. Its gradient population is consequently
the eligible long-document subset. It is not held-out evaluation, a noise-free
population bound, or a guarantee that each optimizer update decreases H loss.

The parent cache is CPU-only and event-local, and is released on success and
failure. Crops are invariant to candidate call order, DP rank ownership and
probe microbatch size for the indexed corpus. Checkpoints/W&B metadata include
both sampling modes; completed independent runs retain their original meaning.

Pass the current smoke/full configs explicitly to the launcher.
The pipeline checks real-corpus window containment and exact target
counts, four-rank gradient normalization, FSDP2 state isolation, pure causal
attention against SDPA, and preservation of the training document mask. Only
successful smoke acceptance starts the independent full trajectory. It audits
all W&B events and the final checkpoint after training.

```bash
export SEQ_PIPELINE_ROOT="$PWD/outputs/variance_monitor/seq_grad_shuffled_$(date +%Y%m%d_%H%M%S)"
/home/kangan/.conda/envs/torchtitan-macro/bin/python scripts/prepare_seq_grad_launch.py "$SEQ_PIPELINE_ROOT" \
  --smoke-config legacy/configs/seq_grad/variance_monitor_shuffled_v1_train16384_val22256_smoke.yaml \
  --full-config legacy/configs/seq_grad/variance_monitor_shuffled_v1_train16384_val22256.yaml
sbatch -M owl --partition=b200_normal_q --qos=owl_b200_normal_short \
  --account=jason_optai-paid --gres=gpu:b200:4 --time=04:00:00 \
  --output="$SEQ_PIPELINE_ROOT/slurm-%j.out" --error="$SEQ_PIPELINE_ROOT/slurm-%j.err" \
  "$SEQ_PIPELINE_ROOT/source/cluster/vt_arc/seq_grad_pipeline.sbatch"
```

Recheck live scheduling before submission. The runtime gate refuses full training
if the measured smoke projection exceeds the remaining allocation. ARC describes
the B200 resources in its
[Owl documentation](https://www.docs.arc.vt.edu/resources/compute/01owl.html).

Job `1512536` started on four B200 GPUs at 00:14:06 EST on 2026-11-17.
Its four GPU checks and four-update smoke run passed.
The [shuffled-data run](https://wandb.ai/dynamic_seq_length/Variance_Monitor/runs/1583ba9510bb4bb19dc113b03bf7ae4f)
started from random weights. At 00:33 EST, the startup audit checked 237 uploaded updates.
All 80 distances through update 200 match the local records.
All four saved data cursors match the shuffled order.
Both completed validations used all 936 windows and 20,830,722 valid targets.
Training continues. Evidence is in
`outputs/variance_monitor/seq_grad_shuffled_v1_20261117/startup_audit.json`.

## Previous run with the unshuffled dataset

Launch `outputs/variance_monitor/seq_grad_paired_doc_20261116/` has 27 passing
CPU tests and a real-corpus audit of 12288 paired windows. H200 preflight job
`8756546` passed causal-logit/gradient comparison with SDPA, preservation of the
training document mask, and exact FSDP2 training-state isolation. The preflight
required two test-fixture initialization fixes; all source revisions and failed
job logs are retained. Production sampler/trainer code did not change during
those fixes. Four-H200 pipeline job `8756347` started at 17:53:45 EST on 2026-11-16.
It uses a four-hour allocation under `jason_optai-paid` / `tc_h200_normal_short`.
All four GPU checks and the four-update smoke run passed.
The [formal run](https://wandb.ai/dynamic_seq_length/Variance_Monitor/runs/5044cadf83e64f71901a62a891015bfe)
started from random weights. At 18:10 EST, its startup audit checked 249 uploaded
training updates, 80 probe distances through update 200, and the step-200 checkpoint.
All checked W&B distances match the local records.
Evidence: `outputs/variance_monitor/seq_grad_paired_doc_20261116/startup_audit.json`.
The run completed at 19:55 EST. The final audit passed all 3819 updates and 77 probe events.
The later data audit found no training shuffle. This run consumed only the initial long-document pool.
Only 38.65% of its packed training windows contain a single document.
These checks establish correct execution, but do not establish representative sampling of the complete corpus.
See `outputs/variance_monitor/seq_grad_paired_doc_20261116/training_distribution_review.md` for the measured distributions.

---

# Historical independent-batch implementations and runs

The following sections document the older protocols and their executed snapshots.
Their sampling rules are not the current paired-window configuration.

# Packed context-gradient distance (v2)

`seq_len_error` is the finite-batch raw distance
`||g_s(theta_t) - g_16384(theta_t)||_2`, measured before clipping or optimizer
transformations. It includes sampling noise; it is not a certified population
bound. No variance correction, squared-distance aliases, relative distances,
cosines, gradient norms or profile plots are added.

The original experiment is `legacy/configs/seq_grad/variance_monitor.yaml`: the existing
Llama 100M native-vocabulary model, Llama3 tokenizer and verified FineWeb-Edu-long
training artifact; AdamW at constant LR 5e-4, training length 1024, global batch
512, local batch 16, 3819 updates. Warmup is zero and the LR multiplier is one.
Parameter-group LR ratios remain unchanged. Training uses production stream
packing and document-blocked FlexAttention. The existing validation population,
length 1024 and 20,971,520-token budget remain unchanged.

Probe global sequence count is 256, independent of the training batch. With four
replicated-DP GPUs and local probe batch one, every length accumulates 64
microbatches per rank. Global sizes 128/256/1024 are supported when divisible by
DP degree times local batch. The reference is a real 16384-token forward/backward;
only its RoPE cache allocation is restored after the upstream trainer shrinks it.
The intrinsic positional limit, theta and scaling are unchanged.

The 16 candidates are:

```text
64 256 384 512 576 640 704 768 832 896 960 1024 1280 1536 2048 4096
```

Each event measures at completed updates 0, 50, 100, …, 3800. Each length has a
fresh random stream derived from `(seed, completed_updates, length, global sample
index)`. Ranks own disjoint sample indices `rank, rank+D, …`; sampled source
records may coincide by chance. A sample chooses a uniform token start from the
concatenated packed training population, with replacement, and runs the existing
packer forward at the requested length. This weights documents by their packed
token mass, matching stream packing's token exposure. No additional minimum
document length is imposed. Short samples are never reference prefixes.

The source indexes are read-only; probe cursors, leftovers, RNG and counters are
separate. BOS/EOS, ignored cross-document labels and rebased fragment positions
come from `PackedDocumentDataset`; masks are constructed by the production
`Trainer.post_dataloading_process`. The reference length describes a packed
rectangle, not an uninterrupted document context. Legacy text-loader fixtures
use separate shuffled text streams with the production text packer.

For every complete batch, count actual non-ignored targets globally. The pinned
TorchTitan `BaseLoss` / `ChunkedLossWrapper` computes local loss sums divided by
this count, and replicated FSDP2 reduces gradients with SUM. There is no further
sequence-length, accumulation or DP scaling. One detached FP32 reference gradient
(FP64 for FP64 parameters) is cached; candidate gradients are processed one at a
time. Tied parameters count once, missing gradients count as zero, and sharded
parameters and nonfinite results are rejected.

Probes restore Python, NumPy, CPU/CUDA RNG and mutable model buffers, including on
failure. They do not step the optimizer/scheduler or consume training/validation
iterators, token budgets or loss averages. The checkpoint stores the last event
and protocol signature. A final update on the interval is measured before saving.

W&B receives exactly `context_bias/step` plus
`context_bias/distance_s{length}` for each candidate, on the existing logging rank.
All 16 series use `context_bias/step` as x-axis. Training and validation metric
names remain unchanged. One open W&B step accommodates initialization, training,
post-update probes and validation. Settings are saved under `seq_grad_probe`.

`seq_grad_events.jsonl` records the same payload, actual global valid-target
counts, maximum event wall time across ranks and maximum allocated GPU GiB.
This cost information stays in local artifacts, outside the 17-key W&B payload.
Each event processes 8,437,760 nominal target positions for N=256.

## Commands

Use the existing environment:

```bash
export DYNAMIC_SEQ_LEN_PYTHON=/home/kangan/.conda/envs/torchtitan-macro/bin/python
OMP_NUM_THREADS=1 "$DYNAMIC_SEQ_LEN_PYTHON" -m unittest tests.test_seq_grad tests.test_document_packing tests.test_llama3_1b -q
OMP_NUM_THREADS=1 "$DYNAMIC_SEQ_LEN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 -m tests.seq_grad_distributed
```

On an allocated four-GPU node, the standalone real-model smoke and full commands
are (use separate run IDs; the launcher supplies them):

```bash
"$DYNAMIC_SEQ_LEN_PYTHON" scripts/train.py legacy/configs/seq_grad/variance_monitor_smoke.yaml --nproc 4
"$DYNAMIC_SEQ_LEN_PYTHON" scripts/train.py legacy/configs/seq_grad/variance_monitor.yaml --nproc 4
```

The automated submission freezes code, creates separate smoke/full manifests,
checks real NCCL normalization, FSDP2 training-state isolation and document masking,
then runs the 4-step smoke with events 0/2/4 and checks the uploaded W&B history.
Only successful acceptance starts the independent 3819-update trajectory. It
writes `acceptance.json` with measured event cost and a conservative runtime
projection; it stops if the projection exceeds the remaining allocation.

```bash
export SEQ_PIPELINE_ROOT="$PWD/outputs/variance_monitor/seq_grad_$(date +%Y%m%d_%H%M%S)"
"$DYNAMIC_SEQ_LEN_PYTHON" scripts/prepare_seq_grad_launch.py "$SEQ_PIPELINE_ROOT"
sbatch -M owl --output="$SEQ_PIPELINE_ROOT/slurm-%j.out" --error="$SEQ_PIPELINE_ROOT/slurm-%j.err" cluster/vt_arc/seq_grad_pipeline.sbatch
```

The script requests four B200 GPUs, `jason_optai-paid`,
`owl_b200_normal_short`, and eight hours. Recheck live availability before a new
submission. ARC documents the partition/QoS limits on its
[Owl page](https://www.docs.arc.vt.edu/resources/compute/01owl.html) and GPU request
syntax on its [Slurm options page](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html).

After the full run finishes, verify every local event against W&B and load the
final checkpoint's update/token counters without loading model tensors:

```bash
"$DYNAMIC_SEQ_LEN_PYTHON" scripts/audit_seq_grad_run.py \
  --plan "$SEQ_PIPELINE_ROOT/plan.json" --stage full \
  --output "$SEQ_PIPELINE_ROOT/full_completion_audit.json"
```

The audit requires all configured training updates, constant LR, scheduled
validation, exactly 17 probe keys and every scheduled event. It also writes
`full_completion_audit.maxima.csv`, with each length's maximum observed raw
distance and corresponding update. These maxima include finite-batch sampling
noise and are not population bounds. The audit does not fit a curve.

Old v1 checkpoints are intentionally incompatible: they used training parents,
crops and a different measurement coordinate. Legacy v1 results remain described
in `docs/seq_grad_distance.md`; they must not be mixed with this sampler's results.

## Long-context training from initialization

`legacy/configs/seq_grad/variance_monitor_train16384.yaml` trains from a fresh seed-1
initialization at the reference length H=16384. Its global batch is 32, local
batch is one, and four replicated-DP GPUs accumulate eight microbatches per
update. This preserves the original 524288 nominal tokens per update, 3819
updates, constant LR 5e-4 and independent N=256 probes. The original 1024 run
and its checkpoints remain separate. A new run manifest refuses to load an
existing checkpoint.

Training and validation use stream packing with document-blocked attention.
Validation also uses length 16384 and local batch one. The frozen artifact has
1280 sequences and 20970765 valid targets: 755 cross-document labels are ignored
among 20971520 physical positions. Loss values do not select the experiment or
its checkpoints; smoke acceptance only requires finite arithmetic. The purpose
of the longer training is to expose model updates to long document contexts.
Finite-batch gradient distances still include sampling noise.

The separate smoke configuration exercises the same training/validation shape
and probes at updates 0/2/4 before the independent full run:

```bash
export SEQ_PIPELINE_ROOT="$PWD/outputs/variance_monitor/seq_grad_train16384_$(date +%Y%m%d_%H%M%S)"
"$DYNAMIC_SEQ_LEN_PYTHON" scripts/prepare_seq_grad_launch.py "$SEQ_PIPELINE_ROOT" \
  --smoke-config legacy/configs/seq_grad/variance_monitor_train16384_smoke.yaml \
  --full-config legacy/configs/seq_grad/variance_monitor_train16384.yaml
sbatch -M tinkercliffs --partition=h200_normal_q --qos=tc_h200_normal_short \
  --account=jason_optai-paid --gres=gpu:h200:4 --time=08:00:00 \
  --output="$SEQ_PIPELINE_ROOT/slurm-%j.out" --error="$SEQ_PIPELINE_ROOT/slurm-%j.err" \
  cluster/vt_arc/seq_grad_pipeline.sbatch
```

H200 availability was better than B200 in the 2026-11-16 afternoon scheduling
checks. Recheck availability on each submission. See ARC's
[TinkerCliffs documentation](https://www.docs.arc.vt.edu/resources/compute/00tinkercliffs.html)
for the H200 partition and short QoS.

Job `8753910` started on `tc-xe006` at 14:48:30 America/New_York on 2026-11-16.
Four-rank gradient normalization, exact training-state isolation and packed
attention checks passed. The 16k smoke completed its four updates and all three
probe events, with an independent checkpoint/W&B audit. The
[full 16k run](https://wandb.ai/dynamic_seq_length/Variance_Monitor/runs/41731bc5c0334df4b5139dc0155c34f5)
then launched from random initialization. Its initial probe matches the smoke
exactly on the same hardware. It completed all 3819 updates and 77 probe events;
the automatic checkpoint/W&B audit passed, and Slurm recorded `COMPLETED`, exit
`0:0`, at 16:57:08 America/New_York. Execution elapsed was 2:08:38 against the
eight-hour request. Evidence is under
`outputs/variance_monitor/seq_grad_train16384_20261116/`.

The production training loader visits documents in fixed file/row order, with
document IDs partitioned across DP ranks; it does not reshuffle each epoch.
Probe sampling is separate: every `(seed, update, length, global sample index)`
selects a random packed-token start with replacement. The short and reference
draws use different seeds and never inherit the training cursor. Independent
draws may overlap; this is not a held-out evaluation protocol.

`scripts/audit_seq_grad_sampling.py` replays the frozen sampler without GPU work.
For this completed run, all 335104 sample seeds were distinct. Across 77 events,
candidate/reference input-position overlap was 51898 / 327061504 = 0.01587%,
including one label-lookahead position per sequence. No short/reference sample
pair had an identical start; 51 of 315392 candidate sequences intersected a
reference window. Probe overlap with the immediately preceding training update
was 0.0004994% of source positions. At event 3800, 6.3403% of probe positions had
appeared in the cumulative training history. These are source-position counts,
not token-ID matches or semantic near-duplicate estimates. See
`sampling_overlap_audit.json` for per-length counts and the replay protocol.

## Verified run: 2026-11-16

The [full Variance_Monitor run](https://wandb.ai/dynamic_seq_length/Variance_Monitor/runs/9073b24d493843bd8e60321e40b2d536)
finished at 03:59:18 America/New_York. Owl job `1502589` used four B200 GPUs,
`jason_optai-paid` and `owl_b200_normal_short`; its allocation ran for 2:06:58
against an eight-hour request. It completed all 3819 updates and 77 probe events
at updates 0, 50, ..., 3800. The final checkpoint is `step-3819`.

The final audit matched every uploaded distance to the local event journal and
verified constant LR 5e-4, 2,002,255,872 nominal training tokens, all scheduled
validation budgets and checkpoint counters. All 1232 recorded distances were
finite. Final training loss was 3.63565; the last scheduled validation, at update
3800, had loss 3.64348. The full run's median steady event cost was 68.65 seconds,
with maximum allocated probe memory 4.735 GiB per GPU. Its first event took
76.96 seconds with compilation caches already warmed by smoke acceptance.

Four-rank NCCL normalization, exact FSDP2 state isolation, document-mask GPU
checks and the real-model smoke at updates 0/2/4 passed before formal training.
The earlier failure came from aliased optimizer tensors in the reusable test
fixture; restoring copied snapshots fixed it without relaxing comparisons.
The acceptance reader also handles W&B's all-null rows for absent metric groups
while rejecting partial events and requiring every expected coordinate.

Results and provenance are in `outputs/variance_monitor/seq_grad_20261116_repair/`:
`full_completion_audit.json`, `full_completion_audit.maxima.csv`,
`final_slurm_outcome.json`, `acceptance.json` and `submission.json`.
The maxima CSV reports empirical finite-batch maxima; it does not certify
population bounds. Source snapshots and the original failed-run evidence are
preserved alongside the corresponding manifests.
