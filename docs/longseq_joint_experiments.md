# Superseded: long-document joint schedule experiments

The six formal jobs below were cancelled on 2026-11-05.
They used the wrong trainer for the requested comparison.
Use [the corrected LossGradSeqTrainer experiment](longseq_loss_grad_experiments.md).
The remaining content records the previous setup and historical status.

The user selected joint sequence-length and batch-size training on 2026-11-05.
Each step uses one sequence length for forward and backward.
These experiments use `DynamicSeqBSTrainer`.

## Data and model

The dataset is the private `long-sequence-length-dataset/fineweb-edu-long-30b` corpus.
Each source document contains at least 12,288 original Llama 3 tokens.
The reader uses the verified local token cache and never joins documents.
The experiment configuration pins the local manifest SHA256.
Short tails do not contribute labels. The trainer counts actual label tokens.

The tokenizer has 128,256 entries. Its files are in `assets/hf/llama3-tokenizer`.
The model has **103,428,608 parameters**, with tied input and output weights.
It uses 12 layers, width 512, eight attention heads, and four KV heads.
This width keeps the total near 100M after the vocabulary expansion.
The reference model used width 768 and a 32,000-entry vocabulary.

Cross-project losses therefore do not isolate the effect of document length.
All six new runs use the same model and tokenizer.
All six runs use FlexAttention, as the reference runs did.

## Schedules

Reference project: [finweb-llama-100m-seqbs-joint-scheduler](https://wandb.ai/dynamic_seq_length/finweb-llama-100m-seqbs-joint-scheduler).
New project: [longseq-finwebedu-llama-100m-seqbs-joint-scheduler](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler).
The folder `outputs/longseq_joint/` contains the reference configurations and summaries.

Multiply every reference sequence length by six.
Keep global batch values, step boundaries, periods, and LR schedules.
Each run uses 3,840 updates and **12,079,595,520 training label tokens**.
This experiment budget is smaller than the corpus size of 30B raw tokens.

| Configuration | Sequence length | Global batch | LR |
| --- | --- | --- | --- |
| `bs_increasing_seq_decreasing` | 12,288 down to 48 | 256 up to 65,536 | Constant |
| `bs_decreasing_seq_increasing` | 48 up to 12,288 | 65,536 down to 256 | Constant |
| `bs_cyclic_seq_inverse` | 12,288 down to 48 and back | 256 up to 65,536 and back | Constant |
| `linear_bs_cyclic_seq_cosine_lr` | 48–12,240, repeated | Linear 256–768 | Cosine |
| `linear_bs_cyclic_seq_constant_lr` | 48–12,240, repeated | Linear 256–768 | Constant |
| `four_stage_cycle` | 3,072 or 8,832 | Four stages, 256–1,024 | Constant |

The first three runs use nine length levels: 48, 96, 192, 384, 768, 1,536, 3,072, 6,144, and 12,288.
The cyclic run repeats every 256 updates.
The linear-batch runs retain the symmetric reference anchors at steps 1, 128, 129, and 256.
Their length anchors are 48, 12,240, 12,240, and 48.
The four-stage run repeats every 960 updates. Each stage contains 240 updates.

Its stages hold small values, increase batch, hold large values, then reset both values.
The peaks differ because the reference schedules used different peaks to match token budgets.

The constant LR is 0.0005. The cosine run starts at 0.0005 and ends at zero.
All runs use AdamW, seed 1, BF16 parameters, and FP32 reductions.
Checkpoint saves occur every 200 updates. Each run retains its two latest checkpoints.

## Memory and validation

Microbatch controls memory. It does not change the scheduled global batch.
The initial microbatch limit is 49,152 tokens per GPU, with at most 1,024 sequences.
Choose the largest microbatch within this limit that divides global batch divided by DP size.
Use gradient accumulation for the remaining sequences.
The folder `outputs/longseq_joint/` contains a CSV schedule for each run.

GPU tests must check this capacity before formal runs start.

Loss computation uses 16 chunks. All scheduled lengths are divisible by 16.
The reader uses synchronous reads and captures its state after every batch.
This prevents checkpoint replay from using a stale batch shape.

Validation uses 1,704 fixed windows from the held-out long-document split.
Each window has 12,288 labels. Each evaluation contains **20,938,752 labels**.
The window count divides evenly across two, four, or eight DP ranks.
The files record tokenizer hashes, tensor hashes, and source document positions.
Every run evaluates the same windows at step 1 and every 20 updates.

## Execution records

Use `outputs/longseq_joint/jobs.json` for current job IDs.
Use the GPU test result JSON and Slurm logs for measured memory and throughput.
CPU preparation job 8447970 passed 28 tests and all six configuration checks.
It generated the fixed validation artifact and checked the actual model parameter count.
Job 8448014 checked the final loader snapshot behavior, all six profiles, the GPU test configuration, and validation hashes.

The first H200 test failed during initialization because `flash_attn_interface` was absent.
The new configuration uses FlexAttention. It requires no additional FlashAttention package.
The replacement GPU test is job 8448209 on one A100.
Its per-GPU microbatch limits match the formal runs.

All formal runs use four DP ranks. Different GPU types affect runtime comparisons.

![Exact sequence and batch schedules](../outputs/longseq_joint/schedules.png)

The requested account is `jason_optai-paid`. GPU jobs use the matching short QoS.
Current resource choices use live scheduler queries.
[ARC Slurm options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html),
[TinkerCliffs configuration](https://www.docs.arc.vt.edu/resources/compute/00tinkercliffs.html),
and [Owl configuration](https://www.docs.arc.vt.edu/resources/compute/01owl.html) describe the cluster settings.

## Verified execution status
Checked at 2026-11-05 07:39:03 UTC.
The A100 test completed nine consecutive updates and both fixed validation events.
All losses and gradient norms were finite. The test reported zero OOM events.
Peak active memory was 7.98 GiB. Peak reserved memory was 9.15 GiB.

Checkpoint saves passed. W&B contains all nine training rows and marks the test as finished.
[GPU test](https://wandb.ai/dynamic_seq_length/dynamic-seq-bs-debug/runs/2281c96b-f469-4b30-b005-8376d8d6c49d).
All six formal jobs passed the test gate. Their full training runs are pending.
| Profile | Slurm job | GPUs | Requested walltime | Host RAM | State |
| --- | --- | --- | --- | --- | --- |
| [bs_cyclic_seq_inverse](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/57109e0e61fb4403b97319c96e68ada7) | 8448422 | 4 × A100 | 12 hours | 64 GiB | PENDING |
| [bs_decreasing_seq_increasing](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/2b64d3d5fbb1492a92bd1e49a0611d2c) | 8448423 | 4 × H200 | 12 hours | 64 GiB | PENDING |
| [bs_increasing_seq_decreasing](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/9821cb8f1bed4eac87001be0ae6a8606) | 8448424 | 4 × A100 | 12 hours | 64 GiB | PENDING |
| [four_stage_cycle](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/993c3004fdeb4f1e92b2dcba0fc1a036) | 8448425 | 4 × A100 | 12 hours | 64 GiB | PENDING |
| [linear_bs_cyclic_seq_constant_lr](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/d3519a91290b43aba0bbc067770867bb) | 1403001 | 4 × B200 | 12 hours | 64 GiB | PENDING |
| [linear_bs_cyclic_seq_cosine_lr](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/728c263c38a04177b9c55ade73683619) | 8448426 | 4 × A100 | 12 hours | 64 GiB | PENDING |

Requested walltime is a limit, not an expected duration.
The single-GPU test used about 6.2 GiB of host RAM. The formal requests allow 64 GiB for four ranks.
The YAML resource field `mem` now sets the Slurm host-memory request. Existing configurations without this field retain their previous behavior.
W&B run pages appear after the corresponding training processes start.

The job record contains timestamped queue estimates. These estimates can change.
