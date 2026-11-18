# Separate loss and gradient sequence lengths

`LossGradSeqTrainer` measures loss on a long parent sequence, then updates the
model using one uniformly sampled contiguous crop per parent. Both forwards use
the parameters from the start of the optimizer step. The parent forward uses
`torch.no_grad()` in training mode. It preserves the training RNG and model
buffers. Only the crop loss produces gradients. This is generally not an
unbiased gradient of the long-context loss, because crops remove preceding
context and sample edge tokens at different rates.

## Implementation plan and components

1. Resolve two independent sequence schedules and the global sample batch for
   the current optimizer step. Validate every scheduled step before training.
2. Resolve each phase's own microbatch capacity. Never use the smaller phase's
   capacity as the other phase's microbatch.
3. Read exactly `global_batch / DP` parent sequences onto CPU. Collectively
   reject exhausted data or invalid inputs before starting model work.
4. Sample aligned input/label/position crops with the existing stateless crop
   utility. Regroup the short CPU tensors into gradient microbatches. This
   preserves sample order and does not reread the dataset.
5. Compute the long forward-only loss in its own microbatches. Release GPU
   predictions before proceeding to the gradient phase.
6. Pass the crops to TorchTitan's native update: forward/backward accumulation,
   gradient synchronization, clipping, one optimizer step and one LR step.
7. Log long loss as the primary training loss, and log crop loss separately.
   Save the two schedules, batching protocol signature and token counters in
   checkpoint state.

Production files are `loss_grad_seq/trainer.py`, `batching.py`, and `metrics.py`.
They reuse `DynamicSeqBSDataLoader`, `crop_batches`, the existing length and batch
schedulers, dynamic block compilation, and `FixedBudgetValidator`. Supported
topology is single GPU or replicated DP, with no dataloader prefetch. Existing
Trainers retain their behavior.

## Independent batch calculation

Let `L` be loss length, `G` be gradient length, `B` be global sequences per update,
and `D` be DP size. Require `0 < G <= L <= training.seq_len` and `B % D == 0`.
For each phase independently, apply its scheduled microbatch upper bound and,
when nonzero, its optional token budget divided by that phase's length:

```text
loss_cap = loss_microbatch_scheduler(step)
grad_cap = grad_microbatch_scheduler(step)
if loss_tokens_per_microbatch > 0:
    loss_cap = min(loss_cap, loss_tokens_per_microbatch // L)
if grad_tokens_per_microbatch > 0:
    grad_cap = min(grad_cap, grad_tokens_per_microbatch // G)

loss_microbatch = min(B // D, loss_cap)
grad_microbatch = min(B // D, grad_cap)
loss_forward_batches = ceil((B // D) / loss_microbatch)
gradient_accumulation_steps = ceil((B // D) / grad_microbatch)
```

The final microbatch may be smaller. There is no rounding of the global batch,
no discarded samples, and no requirement to reduce microbatch to a divisor of
the sample count. Each phase's summed loss is normalized by its own global
valid-label count, including unequal tail microbatches and ignored labels.
Capacity below one sequence is an error.

All five schedulers accept the existing constant, increasing, decreasing,
triangular, cyclic, and piecewise-linear schedule types. Token budgets default
to zero (disabled). Explicit microbatch schedules can therefore be used alone.

The `training.global_batch_size` value must match the initial global schedule.
The bootstrap `training.local_batch_size` must divide the initial per-rank sample count, as required by native TorchTitan initialization.
Use one for schedules whose initial sample count cannot be divided by the gradient microbatch.
Actual phase microbatches come from their own schedules.
`training.seq_len` is model capacity, not the current length.

Token budgets are calibrated capacity controls, not universal memory guarantees.
Do not infer their ratio from forward/backward FLOP ratios. Peak GPU memory is
the maximum of the sequential long forward, short forward/backward, and optimizer
phases, with persistent model/optimizer allocations included. CPU memory also
holds one optimizer step's parent tokens and cropped tokens.

## Prepared formal experiment

`legacy/configs/loss_grad_seq/formal_12288.yaml` is prepared for review and is **not
submitted as a full training run**. It uses:

- Constant global batch 512, AdamW LR 0.0005, BF16 parameters, FP32 reductions.
- Fixed loss length 12,288. Gradient length cyclic from 512 to 12,288.
- 3,840 optimizer steps, period 320, exactly 12 complete periods.
- Four loss chunks, and gradient lengths rounded to multiples of four.
- Independent token capacities: 196,608 for loss, 131,072 for gradient, with
  scheduled microbatch upper bounds of 64 for both phases.
- The same 100M model/tokenizer, FineWeb protocol, seed 1, compile, optimizer
  settings, and held-out source shard as the reference run below.
- Validation length equals the long loss length, 12,288. Its rebuilt artifact
  contains 20,963,328 tokens in 1,706 sequences. The old 20,971,520 budget cannot
  be divided by 12,288. The new budget removes the final 8,192 tokens.
- `jason_optai-paid`, two B200 GPUs, 24-hour requested walltime. Runtime is not
  guaranteed by the short tests.

For the ascending half of each cycle:

```text
range = 12288 - 512 = 11776
half_period = 320 / 2 = 160
ideal_increment = 11776 / 160 = 73.6 tokens/step
```

After four-token rounding, increments are 72 or 76. The descending half is
symmetrical. Steps 1, 161 and 321 have gradient lengths 512, 12,288 and 512.
The final step of a period precedes the next minimum. This matches the existing
one-based cyclic scheduler.

Mean gradient length is exactly 6,400.
The full run processes 12,582,912,000 nominal gradient tokens and 24,159,191,040 parent tokens. Valid-label counts are tracked independently.

With two GPUs the loss microbatch is 16 throughout. Gradient microbatch is 64
at length 512 and 10 at length 12,288, with a final tail where needed. Thus a
short gradient step can use four times as many sequences per backward as the
long loss forward. The two phases can also reverse which has the larger batch.

Reference: [llama3-100m-dynamic-seq-cyclic-512-8192-period256-3840steps-lr5e-4_run](https://wandb.ai/dynamic_seq_length/finweb-llama-100m-seqlength-scheduler/runs/d85b196b15c4412dac1a584c6eef2f6c).
Its configuration was retrieved through the authenticated W&B API and saved
as `outputs/loss_grad_seq/reference_run.json` (model serialization omitted).

## Validation length

`LossLengthValidator` evaluates at the current long loss length.
For frozen data, `validator.eval_seq_len` is the maximum scheduled loss length
and must match the artifact capacity. Constant loss length uses the entire artifact.
If loss length changes, validation uses a prefix of that length from each frozen parent sequence.
The held-out parent population stays fixed, and the reported evaluated-token count changes with length.
Source hashes, tokenizer hashes, array hashes and training exclusions remain checked.

## Tests and observability

```bash
OMP_NUM_THREADS=1 python -m unittest tests.test_loss_grad_seq -v
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m tests.loss_grad_distributed
```

CPU tests cover constant batch/cyclic length, dynamic global and phase batches,
both lengths changing, and smaller final microbatches.
They check direct-gradient equivalence, tied-weight chunked CE against full CE, ignored labels,
and RNG/buffer isolation.
They also check checkpoint restoration and coordinated dataset exhaustion. Distributed tests
check global valid-token normalization and collective failure handling.

GPU test configurations:

- `smoke.yaml`: debug model, constant global batch 8, loss microbatch 1 versus
  gradient microbatch 4, nine steps across a full cyclic schedule, native
  checkpoint saves and validation at the current long length.
- `reference_smoke.yaml`: real 100M/FineWeb configuration, constant global batch
  512, fixed loss length 8,192, cyclic gradient length 512-8,192, nine steps with
  a compressed eight-step period. Validation uses length 8,192 and the full
  20,971,520-token budget.
- `long_validation_smoke.yaml`: nine consecutive 100M/FineWeb updates with loss
  length 12,288 and gradient lengths 512-12,288. Validation also uses 12,288.
- `dynamic_smoke.yaml`: both lengths and global batch change. Validation follows
  the current loss length. This also checks unequal tail batches on two GPUs.
- `tests.loss_grad_gpu --schedule-steps ...`: selected positions from the formal
  schedule, including noninteger ideal increments, maximum length, and cycle
  restart. This is a short test with checkpoint/validation disabled, not the
  formal 3,840-step training run.

`tests.loss_grad_gpu` always writes to the W&B project `dynamic-seq-bs-debug`,
including when reading the formal YAML for selected-position tests. Sparse schedule-position probes are diagnostics, not continuous training logs.
Normal smoke runs use consecutive steps and `metrics.log_freq: 1`. GPU result
JSON files record the actual evaluated steps, batch sizes, memory and run URLs.

The [test results](../legacy/reports/loss_grad_seq_results.json) record the completed two-A100 tests
and remote W&B verification. All three revised runs completed nine consecutive
updates, with both training losses and matching validation lengths.
The report also records the failed bootstrap-batch attempt and its verified fix.

Primary `loss_metrics/global_avg_loss` is long loss.
The only additional training loss is `loss_metrics/short_seq_loss`.
Validation keeps `validation_metrics/loss`, with its length in `validation_metrics/seq_len`.
The native W&B logging mechanism remains unchanged.

Other metrics include both sequence lengths, both microbatches, phase batch
counts, parent tokens, gradient tokens, and phase times. Native `n_tokens_seen`
and valid-token counts refer to gradient crops. Parent counters are separate.

MFU includes a conventional estimate of long-forward FLOPs (one third of the
upstream F+B estimate). It is not a measurement of recomputation under activation
checkpointing. Phase timers are host wall times ending at loss/gradient scalar
synchronizations. The gradient phase includes the optimizer update.

ARC scheduling was checked against the [Slurm options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html),
[GPU resources](https://www.docs.arc.vt.edu/resources/gpu.html), and
[Owl configuration](https://www.docs.arc.vt.edu/resources/compute/01owl.html).
Actual availability and job outcomes come from live Slurm queries, not the
documented inventory or prior start estimates.
