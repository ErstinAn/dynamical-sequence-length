# Pointwise X monitor

Use the [historical revision](https://github.com/ErstinAn/dynamical-sequence-length/tree/427d321c40e8b82444d2f5c07082e6155c0f914e) for the commands on this page.

This document records historical monitor experiments.
Use [the main guide](../../docs/codebase_kick_off/getting_started.md) for the current 1B schedule experiment.
Historical commands require their original assets and server settings.

`GradVarTrainer` supports `grad_var_monitor.mode: x`. The legacy accumulation
monitor remains available for existing configurations and shared tensor/sync
helpers. The X experiment never emits `global_grad_var` or per-parameter metrics.

The fixed training trajectory uses the existing native-vocabulary 103M Llama 3
model, a 1024-token training length, global batch 512, local batch 16, 3819
updates, AdamW LR 0.004, 381-step warmup and cosine decay to 0.02 of peak LR.
The optimizer/schedule settings come from `legacy/configs/seq1024.yaml`; the model and
tokenizer are the existing long-document 100M variant, as required by the token
cache. All parameters start randomly initialized. Probe lengths do not change
the training batch or LR schedule.

`train_loader` advances the ordinary training iterator. `probe_loader` reads the
same verified, read-only cache by index without accessing training iterator state.
The parent pool consists of non-overlapping 16384-target windows entirely inside
one document, after virtual BOS/EOS and next-token label alignment. Tails are
dropped. Independent uniform draws with replacement weight documents by their
number of complete parent windows. Seed and completed-update count define each
event; global draw IDs are partitioned by DP rank. The same 1024 draws are reused
across all candidate lengths, using aligned prefixes with reset positions and
causal attention. No forward pass at 16384 is needed.

Each of R=8 groups averages n=128 sequence gradients. Every microbatch loss sum
is divided by n*s; the pinned FSDP2 uses SUM reduction and no additional DP or
accumulation divisor. The Welford estimator uses one mean vector and a scalar M2:

```
V_group = M2 / (R - 1)
X = n * s * V_group
V_total = V_group / R
X = (R * n * s) * V_total
```

Missing gradients are zeros. Differences use FP64 chunks, means use FP32 (FP64
for FP64 test inputs). Measurements use raw gradients, with no optimizer,
scheduler, clipping or weight-decay calls. Probes restore Python/NumPy/Torch RNG
and model buffers even on failure, and bypass training data/token counters.
Runtime RoPE caches support the largest candidate without changing training
length or RoPE scaling. Unsupported shards, AMP scalers, padding, invalid lengths
and indivisible groups are rejected explicitly.

Events are theta_0 before the first update, theta_50 after update 50, etc. Final
updates at the interval boundary are included. Checkpoints retain the last event
and a protocol/data signature so resume does not repeat a saved event.

Each W&B event contains only `grad_var/step` and `grad_var/x_s{length}` for
64, 512, 576, 640, 704, 768, 832, 896, 960, 1024 and 4096. The latter metrics use
`grad_var/step` as their x-axis. Training, probe and validation writes leave their
shared internal step open until the next step (or close), so differing log
intervals cannot discard probes. Config records the derived total of 1024, all
probe settings, sampler identity `uniform_parent_prefix_v1`, and cache identity.
Local `x_events.jsonl` also records event seconds and peak allocated GiB; these
costs are not additional W&B curves. The peak is the CUDA process high-water mark,
not incremental probe-only memory.

Fit outside training on the nine 512–1024 lengths; hold out 64 and 4096. This is a
pointwise measurement for the controlled sampler, not a uniform parameter bound
or a claim about a production reader's different document weights/correlations.
The delta experiment is not implemented here.

## Data sufficiency (verified token indexes, 2026-11-15)

| Quantity | Actual count |
| --- | ---: |
| Training documents | 1,348,067 |
| Raw training tokens | 30,000,004,070 |
| Available 1024-target training tokens after tails | 29,332,749,312 |
| Eligible 16384-parent documents | 853,433 |
| Complete training parent windows | 1,129,924 |
| Planned training target tokens | 2,002,255,872 |
| Events (including initialization) | 77 |
| Parent draws across events, with replacement | 78,848 |
| Probe target tokens per event | 11,337,728 |
| Probe target tokens across events | 873,005,056 |

The probe uses the training split, not the much smaller validation split. It
requires independent sampler state, not disjoint storage or previously unseen
documents. The validation split has 61,008,896 usable 1024-target tokens. Training
uses 6.83% of the available token budget; data supply is sufficient.

## Commands and acceptance

From the repository root, in a four-GPU allocation:

```bash
export DYNAMIC_SEQ_LEN_PYTHON=/home/kangan/.conda/envs/torchtitan-macro/bin/python
# Short acceptance: same model, batch, data and LR schedule; 4 updates, events 0/2/4.
"$DYNAMIC_SEQ_LEN_PYTHON" scripts/train.py legacy/configs/grad_var/x_monitor_smoke.yaml --nproc 4
# Full trajectory: 3819 updates, interval 50.
"$DYNAMIC_SEQ_LEN_PYTHON" scripts/train.py legacy/configs/grad_var/x_monitor.yaml --nproc 4
```

The submitted pipeline uses frozen source/config snapshots and the existing
manifest launcher. It first runs a four-rank CUDA normalization reference, then
the full-size four-step smoke experiment. It checks all 11 scalars at 0/2/4 both
locally and in W&B before starting the full run. On a CUDA OOM only, it retries
with probe local batch 16 and two accumulations (n/R unchanged). Any other failed
acceptance stops the pipeline. `acceptance.json` contains measured event time and
memory, selected microbatch size and the full-run manifest. The measured results from the first GPU acceptance are recorded below.
The recovery smoke additionally requires a full-budget validation pass.

Validation performed locally: estimator arithmetic/two-group identity, missing
and nonfinite gradients, padding/config guards, real indexed prefix alignment,
independent sampler RNG, accumulation versus direct gradients, event cadence,
resume deduplication, different logging cadence, buffer/dropout/RNG restoration,
exact subsequent optimizer/parameter/scheduler/token trajectory equivalence.
The CPU SUM references passed at DP=1, 2 and 4 (microbatches 1, 2 and 4).
GPU/NCCL and actual FSDP2 acceptance passed in job 1501315. The full run then failed before its first update because the validation artifact was not explicitly configured. Recovery adds an explicit 1024-token long-document artifact, preflight guards, and full-budget validation to smoke acceptance.

```bash
OMP_NUM_THREADS=1 "$DYNAMIC_SEQ_LEN_PYTHON" -m unittest \
  tests.test_grad_var_monitor tests.test_grad_var_trainer \
  tests.test_grad_var_chunked_loss tests.test_x_monitor -q
OMP_NUM_THREADS=1 "$DYNAMIC_SEQ_LEN_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=4 -m tests.x_monitor_distributed
```

Resources: paid account `jason_optai-paid`, Owl `b200_normal_q`,
`owl_b200_normal_short`, four GPUs, 16 CPUs, 128 GiB host memory and 8-hour
requested walltime. This is separate from queue waiting time. Current resource
selection used live `sbatch --test-only` estimates; estimates can change.
See [ARC Owl QoS](https://www.docs.arc.vt.edu/resources/compute/01owl.html#quality-of-service-qos)
and [ARC GPU job options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html).

## Measured acceptance and validation repair (2026-11-15)

Job 1501315 completed the four-rank CUDA SUM reference and three full X events.
Event 0 took 65.664 s including compilation; events 2 and 4 took 14.739 and
14.585 s. Peak CUDA allocated memory was 20.375 GiB (process high-water mark).
The smoke W&B run is `8e086aa4c14e400fbf22a6c7008b1a99`. Formal initialization
failed because an unspecified artifact path selected a legacy validation cache.
Both configs now use `assets/data/longseq_validation_llama3_1024` explicitly:
20,480 sequences, 20,971,520 valid labels, same Llama 3 tokenizer and verified
long-document validation split. Smoke now enables validation with the full budget,
and its W&B validation result must pass before production begins.

## Expanded experiment for fitting (2026-11-15)

`legacy/configs/grad_var/x_monitor_expanded.yaml` uses constant AdamW LR **0.0005** from
the first update, with no warmup or decay, as requested on 2026-11-15. The pinned
TorchTitan scheduler interface remains present but its factor is identically one:
`warmup_steps: 0`, `min_lr_factor: 1.0` (`decay_type: linear` has no effect on LR).
Both full and smoke configs use these settings. The pending cosine experiment
1501581 was cancelled before execution and replaced with a fresh run.
The training seed, model, other optimizer settings, 3819 updates and 50-update
probe cadence are retained. Probe seed 1 supplies draws
independent of the baseline's seed 0. Fit lengths are 256, 384, 512, 576, 640,
704, 768, 832, 896, 960, 1024, 1280, 1536 and 2048. Lengths 64 and 4096 remain
held out from fitting and model selection.

Each length uses 32 independent global gradient groups of 128 sequences, or 4096
draws per event. The W&B X scalar uses all 32 groups with denominator 31. An
additional mean vector and scalar M2 compute four independent eight-group
estimates from the same gradients, without additional forward/backward passes.
These four paired curves are saved in `x_replicates.jsonl` at the same fixed
parameters as the pooled curve. Replicate IDs use the same parent anchors across
lengths; they are independent across blocks, but intentionally paired across
lengths. Four repeats support a stability check; they do not guarantee precise
confidence intervals for heavy-tailed gradient samples.

For raw global-group gradient variance use `x_s / (128 * s)`. For the variance
of a mean of all 4096 sequences use `x_s / (4096 * s)` (the eight-group replicate
uses 1024 instead of 4096). Do not pool different training steps as replicates of
one fixed-parameter curve. Length fitting uses the 14 fit points; held-out points
only measure extrapolation error after the fit is chosen.

The expanded protocol processes 67,895,296 probe target tokens/event and
5,227,937,792 over 77 events, from 315,392 parent draws with replacement. This is
repeated computation on indexed prefixes, not a need for 5.23B distinct probe
tokens. The verified 1,129,924-parent training pool is sufficient.

The supplemental pipeline requests four B200 GPUs, paid account and short QoS,
with a four-hour walltime. Based on the measured baseline, estimated execution is
2.5–3.5 hours, including startup, probes, training, validation and checkpoints;
this is an extrapolation pending the expanded GPU smoke. Smoke acceptance derives
the lengths and cadence from the frozen config and requires all 16 W&B values,
four complete local replicate curves per event, and full-budget validation.
