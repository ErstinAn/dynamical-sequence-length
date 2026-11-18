# Long-document loss and gradient experiments

This page describes the historical single-document experiments.
For the packed experiments, read the [packing implementation report](document-packing_masking/implementation-report.md).

These experiments use `LossGradSeqTrainer`.
They replace the cancelled `DynamicSeqBSTrainer` experiments.
Each run starts from seed 1. No run restores the cancelled runs' weights.

## Loss definitions

A parent is a 12,288-token window from one long document.
Each update first measures cross-entropy on every parent, without gradients.
The trainer then selects one contiguous crop from each parent.
It computes crop loss and applies the crop gradients.
Both losses use the weights from the start of the update.
Both losses divide summed cross-entropy by the number of valid labels across ranks.

Crop positions restart at zero.

| W&B metric | Meaning |
| --- | --- |
| `loss_metrics/global_avg_loss` | Mean training loss on the 12,288-token parents |
| `loss_metrics/short_seq_loss` | Mean training loss on the scheduled crops |
| `loss_grad/loss_seq_length` | Always 12,288 |
| `loss_grad/grad_seq_length` | Scheduled crop length, from 1,024 to 12,288 |
| `loss_grad/parent_tokens_seen` | Cumulative tokens for the long forward pass |
| `n_tokens_seen` | Cumulative gradient tokens |
| `validation_metrics/loss` | Loss on the fixed held-out 12,288-token windows |

Use long loss and validation loss to compare context schedules.
The short loss changes its context length during training.
Use cumulative gradient tokens as the comparison axis when per-step token counts differ.
Parent tokens and gradient tokens overlap. Their sum does not measure unique training data.

## Data and model

The private dataset is [fineweb-edu-long-30b](https://huggingface.co/datasets/long-sequence-length-dataset/fineweb-edu-long-30b).
The local artifact is `/projects/jason_optai/kangan/datasets/fineweb-edu-long-30b`.
The artifact contains 30,000,004,070 raw training tokens.
Each document contains at least 12,288 original Llama 3 tokens.
The reader uses local token files. It never joins documents and drops incomplete tails.

The reader adds native BOS and EOS tokens when it creates windows.
The loader repeats the corpus if a run exhausts the eligible windows.
The configuration pins manifest SHA256 `4504314f51733ce085991de14b11ef595424d1a006b1ebb2475e4fc3d29579db`.

All runs use the native Llama 3 tokenizer from `assets/hf/llama3-tokenizer`.
Its vocabulary contains 128,256 entries.
The model contains 103,428,608 parameters, with tied input and output weights.
It uses 12 layers, width 512, eight attention heads, four KV heads, and FlexAttention.
The reference project used another model width and vocabulary.
Loss differences across those projects do not isolate document length.

## Six schedules

The folder `legacy/configs/longseq_loss_grad/` contains six independent YAML files.
Each file has an empty `sweep` field and submits one run.
These files are job configurations, not W&B Sweep specifications.
The generator is `scripts/prepare_longseq_loss_grad_configs.py`.
The CSV files in `outputs/longseq_loss_grad/` specify every optimizer step.

All runs use 3,840 optimizer steps.
The cyclic runs use 12 periods of 320 steps.
Each scheduled length is divisible by eight. Length changes need not equal eight.
For example, a triangle changes by approximately 70.4 tokens per step before rounding.
The monotonic runs use linear crop ramps and inverse batch curves.

The four-stage cycle holds small values, increases batch, holds large values, then resets both values.
Each stage lasts 80 steps.
The two linear-batch runs have identical data schedules and different LR schedules.

Global batch counts parent windows, with one crop per window.
The generator scales each batch trend toward 12,079,595,520 gradient tokens.
It rounds each global batch to eight windows.
The resulting budget error is below 0.04 percent for every run.
The budgets are approximately equal, not exactly equal.

| Profile | Crop trend | Global batch range | Gradient tokens | Parent tokens |
| --- | --- | --- | --- | --- |
| [four_stage_cycle](../legacy/configs/longseq_loss_grad/four_stage_cycle.yaml) | 1,024 or 12,288 | 232–936 | 12,075,073,536 | 23,444,324,352 |
| [linear_bs_cyclic_seq_constant_lr](../legacy/configs/longseq_loss_grad/linear_bs_cyclic_seq_constant_lr.yaml) | 1,024 → 12,288 → 1,024 | 240–712 | 12,079,584,640 | 22,300,852,224 |
| [linear_bs_cyclic_seq_cosine_lr](../legacy/configs/longseq_loss_grad/linear_bs_cyclic_seq_cosine_lr.yaml) | 1,024 → 12,288 → 1,024 | 240–712 | 12,079,584,640 | 22,300,852,224 |
| [bs_cyclic_seq_inverse](../legacy/configs/longseq_loss_grad/bs_cyclic_seq_inverse.yaml) | 12,288 → 1,024 → 12,288 | 256–3,072 | 12,079,541,760 | 32,756,465,664 |
| [bs_decreasing_seq_increasing](../legacy/configs/longseq_loss_grad/bs_decreasing_seq_increasing.yaml) | 1,024 → 12,288 | 256–3,072 | 12,079,494,400 | 32,757,841,920 |
| [bs_increasing_seq_decreasing](../legacy/configs/longseq_loss_grad/bs_increasing_seq_decreasing.yaml) | 12,288 → 1,024 | 256–3,072 | 12,079,494,400 | 32,757,841,920 |

![Crop, loss, and batch schedules](../outputs/longseq_loss_grad/schedules.png)

The constant LR is 0.0005. The cosine run reduces LR from 0.0005 to zero.
All runs use AdamW, BF16 parameters, FP32 reductions, and seed 1.
Checkpoints occur every 100 updates. Each run retains its two latest checkpoints.

## Memory and validation

The two phases use independent microbatches.
The long forward limit is eight windows, or 98,304 tokens, per GPU.
The gradient limit is 48 windows and 49,152 tokens per GPU.
The trainer accumulates gradients to reach the scheduled global batch.
The last microbatch can contain fewer windows.
The loss uses eight chunks so lengths divisible by eight remain valid.

Validation uses 1,704 fixed long windows from the held-out split.
Each evaluation contains 20,938,752 labels at length 12,288.
Every run evaluates the same windows at step 1 and every 20 updates.
The artifact is `assets/data/longseq_validation_llama3_12288`.
The artifact records source windows and verifies tokenizer and tensor hashes.

## Checks and submission

CPU job 8452306 passed 25 tests and checked all six configurations at DP sizes 1, 2, 4, and 8.
Three additional tests checked trends, cycle counts, budget limits, and batch serialization.
GPU test 1406970 passed eight real-model updates on two B200 GPUs.
Both losses were finite. Long loss always used 12,288 tokens.
The full-length crops reproduced parent loss within 0.00001.

Both fixed validation events and checkpoint saves passed.
Peak active GPU memory was 9.31 GiB. Peak reserved memory was 11.09 GiB.
The test took 1 minute 57 seconds, including startup and compilation.
[Debug run](https://wandb.ai/dynamic_seq_length/dynamic-seq-bs-debug/runs/longseq-lossgrad-debug-1406970).
GPU test 8452332 passed four selected schedule positions on one A100.

It checked global batches 256, 472, 2,872, and 3,072.
It checked gradient microbatches 4, 7, 44, and 48, including partial tail batches.
Peak active memory was 9.32 GiB. Peak reserved memory was 10.89 GiB.
The test recorded zero OOM events and took 6 minutes 20 seconds.
The local `capacity-test-8452332/metrics.json` contains all four measurements.

The capacity test sampled steps out of order, so W&B omitted its later rows.
The eight-step smoke run records both losses in order. Formal runs also use increasing steps.
GPU results and submission status appear in `outputs/longseq_loss_grad/jobs.json`.
Both GPU tests passed before formal submission.

The W&B project is [longseq-finwebedu-llama-100m-seqbs-joint-scheduler](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler).
The experiment group is `loss12288-grad1024-12288-3840steps-v2`.
The debug project is `dynamic-seq-bs-debug`.
The cancelled runs retain their history and have a `superseded` tag where a remote run exists.

GPU jobs use account `jason_optai-paid` and the partition's short QoS.
The requested walltime is a limit, not a measured duration.
Queue estimates can change. The job record stores timestamped observations.
[ARC Slurm options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html),
[TinkerCliffs](https://www.docs.arc.vt.edu/resources/compute/00tinkercliffs.html),
and [Owl](https://www.docs.arc.vt.edu/resources/compute/01owl.html) describe the resource settings.

## Submitted jobs

Checked at 2026-11-05T12:48:28.893969+00:00.
All six jobs passed both GPU test gates before submission.
Each job requests four GPUs and 64 GiB of host memory.
All six snapshots match the current source hash.

| Profile | Slurm job | GPUs | State | Requested walltime |
| --- | --- | --- | --- | --- |
| [bs_cyclic_seq_inverse](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/7b9d0941f9144143a11953e3628d308c) | 1407143 | 4 × B200 | PENDING | 24 hours |
| [bs_decreasing_seq_increasing](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/ea24c5ed526341e59d94f920ba613709) | 8452433 | 4 × H200 | PENDING | 24 hours |
| [bs_increasing_seq_decreasing](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/6705d493f92e4b5daf70c81a5347580b) | 8452444 | 4 × H200 | PENDING | 24 hours |
| [four_stage_cycle](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/d2cb4eeaed074d8782a1793da93dc19f) | 8452451 | 4 × A100 | PENDING | 24 hours |
| [linear_bs_cyclic_seq_constant_lr](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/2372d1d5c2f94fdfb9266d60116f8b40) | 1407159 | 4 × B200 | PENDING | 24 hours |
| [linear_bs_cyclic_seq_cosine_lr](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/ec110cfab65f4d74a3434795e3286748) | 8452464 | 4 × A100 | PENDING | 24 hours |

Resource selection used queue estimates and approximate execution times.
A100 and B200 estimates used measured GPU phase times.
The H200 estimate used interpolation between those measurements. It is not an H200 benchmark.
These estimates include extra time for validation, checkpoints, and throughput variation.
W&B run pages appear when training starts.
