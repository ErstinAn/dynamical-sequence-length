# Sequence gradient monitor

This historical plan predates the current fixed validation artifact.
Its 22,256-token validation settings do not define the current dataset.
Use [the main guide](../../docs/getting_started.md) for the 1B experiment and 16,384-token validation.

## Experiment

Measure the gradient difference between a short window and its complete parent.
Start training from random weights.

| Setting | Value |
| --- | --- |
| Model / tokenizer | Llama 100M / Llama3 |
| Data | `fineweb_edu_long_30b_shuffled_v1` |
| Training order | Fixed document permutation, seed 42 |
| Training length | 16384 tokens |
| Validation length / windows | 22256 tokens / 936 |
| Training batch, global / local | 32 / 1 |
| Optimizer / constant learning rate | AdamW / 0.0005 |
| Updates / training seed | 3819 / 1 |
| GPUs | 4 B200 |
| Reference length `H` | 16384 |
| Probe batch `N`, global / local | 256 / 1 |
| Probe seed | 0 |
| Probe events | Updates 0, 50, ..., 3800 |

Keep packing and attention blocks between documents for training and validation.
Apply the document restriction below only to the monitor.
Use the dataset package for validation.
Its document order uses seed 43. Its fixed window sample uses seed 44.
Allow the position cache to cover all configured lengths.

Use these candidate lengths:

```text
64, 256, 384, 512, 576, 640, 704, 768,
832, 896, 960, 1024, 1280, 1536, 2048, 4096
```

## Sampling

A parent contains `H` input tokens and their `H` next-token labels.
At each event:

1. Sample 256 spans uniformly with replacement from valid spans of `H+1` raw tokens in the training split.
2. Require each span to remain inside one document with token length greater than `H`.
3. Create parent inputs and shifted labels from each span.
4. For each length `s`, select one uniform random offset per parent from the inclusive range `[0, H-s]`.
5. Extract aligned input and label windows of length `s` from those same parents.
6. Reset window positions to `0, ..., s-1`.

A document of length `L` has weight `L-H` when `L>H`.
Use only causal attention for probes.
Do not add special tokens, pack documents, or ignore targets in probes.
Derive sample seeds from the probe seed, event, length, and global sample index.
Keep probe state separate from the training iterator.
The streams can select overlapping data.

## Gradient calculation

Let `ell(i,k,l)` denote cross-entropy at target `k` in probe sequence `i` of length `l`.
At fixed parameters `theta_t`, calculate:

```text
g_H = gradient_theta [sum_i sum_k=1..H ell(i,k,H) / (N*H)]
g_s = gradient_theta [sum_i sum_k=1..s ell(i,k,s) / (N*s)]
e_t(s) = norm_2(g_s - g_H)
```

Calculate `g_H` once per event with all targets in every complete parent.
Calculate each `g_s` with only its window context.
Average gradients before the norm calculation.
Use raw gradients before clipping or optimizer operations.
Use SUM reduction across GPUs with the global target denominator.
Do not divide the gradients again.

Preserve parameters, optimizer state, scheduler state, buffers, random states, and the training iterator.
Exclude probe tokens from training counters.
Clear temporary gradients and parent batches after each event, including after an error.

This measurement includes sampling noise from the selected long-document population.
It does not establish a population bound.

## Output and acceptance

Use W&B project `dynamic_seq_length/Variance_Monitor`.
Record `context_bias/step` and `context_bias/distance_s{s}` for each candidate.
Keep exactly 17 probe metric keys per event.
Record the sampling rules in the run configuration.

1. Check document containment, window alignment, and target counts.
2. Check gradient means across microbatch sizes and GPU counts.
3. Check causal attention and training state preservation.
4. Run four training updates with probe events at 0, 2, and 4.
5. Check finite values, all validation windows, and uploaded W&B events.
6. Start the full run after acceptance if sufficient allocation time remains.
7. Monitor startup until a probe after training reaches W&B.
8. Check 3819 updates and 77 probe events at completion.

Use `legacy/configs/seq_grad/variance_monitor_shuffled_v1_train16384_val22256.yaml`.
Set `sampling=paired_window` and `parent_population=single_document`.
Use account `jason_optai-paid` and QoS `owl_b200_normal_short`.
