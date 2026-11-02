# Gradient distance across sequence lengths

This experiment uses one training trajectory and a fixed global batch of 128 sequences.
Every optimizer step reads a new parent batch with shape `[128, 16384]`.
It compares gradients at the same model parameters, before gradient clipping.
Only a random 2048-token crop from each parent sequence contributes to the optimizer update.

## Protocol

The reference gradient uses all 16384 tokens in each parent sequence.
Each candidate uses one continuous random crop from each of the same 128 parent sequences.
Crop starts are uniform over the inclusive interval `[0, 16384 - length]`.
Candidate lengths use independent crop streams.
They share parent sequences, but their windows need not be nested.

```text
Candidate lengths: 512, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 4096
```

The nine lengths from 1024 through 2048 have spacing 128.
Each candidate is sampled once per step.
The 2048-token measurement uses the exact gradient used for that step's optimizer update.
Inputs, next-token labels, and document positions are cropped together.
The initial document fragment starts at position zero, while subsequent document boundaries retain their resets.
Attention follows the configured model backend and its existing document-boundary rules.

Each backward uses its own batch's global valid-token count for normalization.
All accumulation groups and data-parallel ranks contribute before a gradient is measured.
Missing parameter gradients count as zeros, and tied parameters count once.
The trainer saves the reference gradient and computes the other candidates sequentially, clearing gradients between candidates.
It computes the training gradient last, records its distance, then performs the native optimizer step.
The LR scheduler advances once per optimizer step.

Crop seeds depend on the sampling seed, step, rank, and crop stream.
Changing probe order does not change training crops.
The probes restore torch RNG state and model buffers before the training forward.
Checkpoint state includes a protocol signature and the valid training-token counter.
The parent dataloader uses the existing stateful checkpoint support.
Resume requires the same sampling protocol and batch topology.

## Meaning of the measurements

Let `g_ref` be the long-sequence gradient and `g_s` a candidate gradient.
The primary measurement is `distance = ||g_s - g_ref||_2`.
Relative distance is `distance / (||g_ref||_2 + relative_epsilon)`, with default epsilon `1e-12`.
These are single-sample gradient distances, not covariance estimates or unbiased variance estimates.
The reference gradient is itself stochastic.
Changing length changes token count, context length, and sampled content together.

When either gradient has zero norm, cosine is recorded as zero and `cosine_defined` is zero.
The monitor stores one reference gradient, using FP32 for ordinary training and FP64 for FP64 inputs.
It computes differences and scalar reductions in FP64 chunks.
It does not allocate a covariance matrix or retain all candidate gradients.

## Configuration

Use `configs/seq_grad/reference16384.yaml` for the 100M model.
It retains AdamW with constant LR `0.0005` and matrix weight decay `0.1`.
It uses BF16 model parameters with FP32 gradient reductions.
Validation remains fixed at length 1024.

| Setting | Value or meaning |
| --- | --- |
| `training.seq_len` | 16384, both parent length and model capacity |
| `training.global_batch_size` | 128 sequences |
| `training.local_batch_size` | 1 sequence per backward per rank |
| `seq_grad.train_length` | 2048 |
| `seq_grad.lengths` | The 11 candidate lengths |
| `seq_grad.seed` | 12345, independent of model initialization |
| `seq_grad.enable` | True, measure every optimizer step |
| `metrics.log_freq` | Must be 1 |
| `metrics.profile_freq` | 100, controls only W&B profile charts |
| `training.steps` | 7680, an adjustable training budget |
| `compile.enable` | False for the initial configuration |

The training budget is 2,013,265,920 label tokens, excluding probes.
This budget is a default, not a convergence claim.
With two replicas and local batch one, each gradient accumulates 64 local backwards per rank.
The global batch must remain divisible by replica count times local batch.

At each step, training uses 262,144 label tokens.
All forward/backward passes process 4,456,448 label tokens, including the training pass.
This is 17 times the training token count, not a predicted wall-clock ratio.
Each parent read consumes 2,097,152 label tokens from the data stream.
Unused parts of the parents do not become subsequent training windows.

The implementation supports a single model with replicated data parallelism.
Nontrivial parameter sharding, tensor parallelism, and pipeline parallelism are rejected.
The conservative local batch reduces activation memory pressure at length 16384.
Actual memory use and elapsed time still require a GPU run.
The trainer never silently reduces the parent length or global batch after an out-of-memory error.

## W&B records

The full configuration enables offline W&B in project `finweb-llama-100m-seq-grad-distance`.
Set `wandb.mode: online` and the intended `wandb.entity` to upload during training.
Offline runs retain the same scalars and charts locally.
The existing TorchTitan logger selects the logging rank and records the experiment configuration.

One scalar record is written per optimizer step.
Each candidate has stable keys, such as `seq_grad_distance/s01024`.
Names do not contain the current step, which keeps the number of metric series bounded.

| Metric group | Contents |
| --- | --- |
| `seq_grad_distance/sXXXXX` | L2 gradient distance |
| `seq_grad_distance_sq/sXXXXX` | Squared L2 gradient distance |
| `seq_grad_relative_distance/sXXXXX` | Distance divided by reference norm plus epsilon |
| `seq_grad_cosine/sXXXXX` | Gradient cosine similarity |
| `seq_grad_cosine_defined/sXXXXX` | Whether both gradient norms are positive |
| `seq_grad_grad_norm/sXXXXX` | Candidate gradient norm before clipping |
| `seq_grad_loss/sXXXXX` | Candidate global mean loss |
| `seq_grad_valid_tokens/sXXXXX` | Candidate global valid-token count |
| `seq_grad_reference/*` | Reference loss, norm, and valid-token count |
| `seq_grad/model_step` | Parameter state index, equal to optimizer step minus one |
| `seq_grad_cost/*` | Parent tokens, evaluated tokens, and probe elapsed seconds |
| `n_tokens_seen`, `n_valid_tokens_seen` | Training tokens only, excluding reference and probes |

In the W&B workspace, overlay the 11 scalar series within each metric group against training step.
The reference config enables `metrics.wandb_minimal: true`: W&B receives only
`context-truncation bound/seq_<length>`, `context-truncation bound sq/seq_<length>`,
`train/loss`, `val/loss`, and `time/step_seconds`. The two bound series use the
paper's terminology for the L2 gradient distance and its square, respectively.
The underlying calculations are unchanged.

This mode omits profile tables and auxiliary scalars. Local TensorBoard metrics
remain complete. Set the flag to false for the detailed metric names below.

Use relative distance to compare different stages of training.
Use cosine to distinguish direction changes from norm changes.
The 2048 series describes the actual update gradient before clipping.

At step 1 and every 100 steps, three profile charts use sequence length as the horizontal axis.
They show distance, relative distance, and cosine for that parameter state.
Profiles use the [W&B line-series API](https://github.com/wandb/wandb/blob/main/wandb/plot/line_series.py).

Set `seq_grad.interval: 5` to probe every five optimizer steps, plus the first and last
steps. The default interval is one. Every step still loads a new long parent batch
and updates using its independently sampled training window. Distance series are
logged only on probe steps. Ordinary training scalars remain available every step.

Set `metrics.profile_freq: 1` for profiles every step, or zero to disable profiles.
Profiles and scalars share one W&B write, while TensorBoard receives scalar values only.

Native throughput counts training tokens over the whole measured step, including probe time.
FLOP estimates include all evaluated sequence lengths.
Probe elapsed seconds include reference and candidate passes, but exclude the final training pass.
They are measured on the logging rank, with device synchronization at the interval boundaries.

## Launch and checks

Examine the full command without launching training:

```bash
python scripts/train.py configs/seq_grad/reference16384.yaml --nproc 2 --dry-run
```

Within a GPU allocation, run the small integration configuration:

```bash
python scripts/train.py configs/seq_grad/smoke.yaml --nproc 2
```

The full run requires the existing FineWeb validation artifact and tokenizer.
Use a short full-length run to measure memory before starting the full budget:

```bash
python scripts/train.py configs/seq_grad/reference16384.yaml --nproc 2 --training.steps 2
```

Run CPU correctness tests:

```bash
OMP_NUM_THREADS=1 python -m unittest tests.test_seq_grad -v
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m tests.seq_grad_distributed
```

Tests compare distances against direct gradients and check accumulation normalization.
They check that probes preserve parameters, optimizer state, torch RNG state, model buffers, and LR updates.
They also check crop alignment, checkpoint recovery, logging cadence, and errors before updates.
The two-rank CPU test checks SUM reductions, unequal valid-token counts, collective stopping, and DCP recovery.
These CPU checks do not establish GPU memory feasibility at length 16384.
