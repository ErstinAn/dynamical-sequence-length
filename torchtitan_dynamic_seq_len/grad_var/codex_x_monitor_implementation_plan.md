# X monitor: historical experiment plan

This plan predates the current fixed validation artifact.
Its 22,256-token validation settings do not define the current dataset.
Use [the main guide](../../docs/getting_started.md) for the 1B experiment and 16,384-token validation.

Settings for 2026-11-17. Use `legacy/configs/grad_var/x_monitor_train4096_shuffled.yaml`.

## Settings

| Setting | Value |
| --- | --- |
| Model | Llama 3, about 103M parameters, random initialization |
| Data | `assets/data/fineweb_edu_long_30b_shuffled_v1` |
| Training sequence length | 4096 |
| Optimizer | AdamW, constant LR `5e-4`, no warmup or decay |
| Training updates | 3819 |
| Training global batch | 128 sequences, 524288 target tokens |
| GPUs | 4 B200, replicated data parallelism |
| Local batch / accumulation | 32 sequences / 1 backward pass, for training and each probe group |
| Training seed / probe seed | 1 / 1 |
| Probe interval | Update 0, then every 50 completed updates |
| Sequences per probe group, `n` | 128 globally |
| Independent gradient groups, `R` | 32 per length, or 4096 sequence draws |
| Validation | 936 fixed windows, length 22256, 20830722 valid targets, local batch 2, every 200 updates |

Probe lengths: `64, 256, 384, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 2048, 4096`.
Reserve 64 and 4096 for extrapolation checks. Fit the other 14 lengths outside training.

## Sampling

Use `independent_random_window`. For each event and length, draw sequences independently with replacement.
Give every valid document/start pair equal probability. Read `s + 1` consecutive tokens to form `s` inputs and next-token labels.
Draw new samples for each length. Derive each random stream from the seed, update, and length.
Keep each sequence inside one document. Use causal attention and reset positions to zero.
Do not pack training or probe sequences. Use training BOS/EOS rules and a 22256-position RoPE cache without changing scaling.

Training follows `orders/train.npy` (seed 42). It reads consecutive windows inside each document.
Validation blocks attention between documents and ignores boundary labels.
The probe shares the training split but has separate sampler state. It can select data that training already used.
Probe reads do not advance training. Tokens inside one sequence can be correlated.

## Measurement

Keep model parameters fixed during each event. Each `g_r` contains the full parameter gradient for one global group.
Divide each loss sum by `128 * s`. Use SUM across GPUs without another divisor.
Measure before clipping or optimizer updates. Count tied parameters once.

```text
g_mean = mean(g_1, ..., g_32)
V = sum(||g_r - g_mean||²) / 31
X = 128 * s * V
```

Sum squared distances across parameter tensors. This gives the squared norm of their concatenated gradient difference.
Restore training RNG state and model buffers after probes. Preserve optimizer state, scheduler state, and training counters.

Log `grad_var/x_s{length} = X` against `grad_var/step` in W&B project `Variance_Monitor`.
Recover raw variance with `V = X / (128 * s)`.
Save four estimates from eight groups each in `x_replicates.jsonl`. Each length uses independent draws.

## Start checks

Use account `jason_optai-paid`, QoS `owl_b200_normal_short`, and a five-hour walltime.
Copy data to local NVMe. Check each file hash.
Check four-GPU gradient normalization and document masks. Run four training updates. Complete validation.
Check all 16 probe metrics at updates 0, 2, and 4 locally and in W&B.
Start the full run only after these checks pass. Check that its first complete probe event reaches W&B.
