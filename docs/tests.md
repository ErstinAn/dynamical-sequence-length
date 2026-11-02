# Test procedure

## CPU tests

Run the unit tests from the repository directory.

```bash
PYTHONPATH=.:cluster/vt_arc OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -t . -v
```

The tests check variance, data identity, parameter groups, configuration, and checkpoint recovery rules.
The local validation arrays are necessary for the artifact test.
The downloaded tokenizer files are necessary for the tokenizer test.

## Distributed CPU tests

Run the tests in an environment that permits local process sockets.

```bash
GRAD_VAR_DISTRIBUTED_TEST=1 FIXED_VALIDATION_GLOO_TEST=1 \
  PYTHONPATH=.:cluster/vt_arc OMP_NUM_THREADS=1 \
  python -m unittest tests.test_grad_var_distributed \
  tests.test_fixed_validation.FixedBudgetValidationTest.test_real_two_rank_reduction_and_partial_batches -v
```

These tests use Gloo with two or four CPU processes.
They check token counts, errors across ranks, and validation reduction.

## GPU tests

Run the following command in an allocation with two visible GPUs.

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m tests.gpu_smoke
```

The test compares training with and without the monitor.
It checks FP32 and BF16 with MSE and chunked cross entropy.
Each case runs for ten steps.
The test checks the variance against separately computed group gradients.

Run the local training smoke test from the project README.
This test also checks the complete model and data launch path.

## Comparison with the previous source

The cleanup keeps the previous source in a local archive.
The archive is not part of the public package.
An extracted copy of that source is necessary for the following commands.

1. Record the previous results.

   ```bash
   python tests/refactor_trace.py --source-root /path/to/original \
     --legacy --output /tmp/before.pt
   ```

2. Record the new results.

   ```bash
   python tests/refactor_trace.py --source-root "$PWD" --output /tmp/after.pt
   ```

3. Compare every tensor and scalar in the two files.

   ```bash
   python tests/compare_traces.py /tmp/before.pt /tmp/after.pt --output /tmp/comparison.json
   ```

The trace uses seed 42 and deterministic operations.
It runs ten steps for AdamW and Muon on a small language model.
It restores optimizer state after step 5.
The JSON result record contains all loss, gradient norm, and variance values.
This comparison does not measure convergence on FineWeb.

## Dynamic sequence length

The CPU tests include checks of token budgets, token packing, length changes, token counts, and checkpoint recovery.
The tests also read each dynamic configuration.
They examine submission commands without job submission.
Run the distributed test of token counts with two CPU processes:

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m tests.dynamic_seq_distributed
```

Within an allocation with two visible GPUs, run the integration comparisons:

```bash
OMP_NUM_THREADS=1 python tests/run_dynamic_seq_gpu.py --nproc 2 --output outputs/dynamic-gpu-check
```

The output directory must be new.
The tests compare the TorchTitan Trainer with constant dynamic training.
They also restore checkpoints across changes in sequence length.
They also run compiled training.

## Sequence gradient distances

Run `python -m unittest tests.test_seq_grad -v` for direct-gradient and training-state comparisons.
Run `python -m torch.distributed.run --standalone --nproc_per_node=2 -m tests.seq_grad_distributed` for replicated CPU checks.
See the [experiment procedure](seq_grad_distance.md) for W&B records and GPU smoke commands.

## Dynamic sequence length and batch sizes

Run `python -m unittest tests.test_dynamic_seq_bs -v` for schedule, data, gradient, metric, and recovery checks.
Run the two-process CPU test with local sockets enabled:

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m tests.dynamic_seq_bs_distributed
```

For a complete GPU launch, use `configs/dynamic_seq_bs/smoke.yaml` with the existing training script.
The [dynamic batch guide](dynamic_seq_bs.md) describes the test scope and configuration.

Checks on 2026-11-02: the unit suite ran 100 tests.
The suite passed 97 tests and skipped three tests.
The two-process test passed the gradient, token count, recovery, and exhaustion checks.
Training and submission commands passed dry runs.
The CPU environment had no visible CUDA devices.
The separate A30 test below supplies GPU results.

## GPU test of increasing batch and cyclic length

On 2026-11-02, Falcon job `742069_0` completed on one NVIDIA A30 with exit code zero.
The requested time limit was 15 minutes.
The job waited one second and ran for two minutes and seven seconds.
It used the 100M model with a local text fixture, FP32, and compilation disabled.
Validation was disabled for this training test.

| Setting | Values |
| --- | --- |
| Optimizer steps | 24 |
| Global batch size | 8, 16, 32, increasing |
| Sequence length | 128 to 512, cyclic, period 8 |
| Microbatch size | 4, 2, 1, decreasing |
| Accumulation groups | 2, 8, 32 |
| Learning rate | 0.0005, constant |
| Total training tokens | 143,360 |
| Peak active GPU memory | 2.659 GiB |

The [W&B run](https://wandb.ai/dynamic_seq_length/dynamic-seq-bs-debug/runs/f21b613d90184b75a406c7810efa9394) contains all 24 training records.
Each record passed checks against the three schedules and the expected token counts.
Loss, gradient norm, throughput, and FLOPs were finite.
No OOM occurred, and the complete checkpoint exists at step 24.
The configuration is `configs/dynamic_seq_bs/gpu_increasing_cyclic.yaml`.

The first test saved correct TensorBoard records but could not initialize W&B.
The submission environment now removes `WANDB_SERVICE` to prevent reuse of a service socket from the login node.
The repeated GPU test confirmed online W&B records after this correction.
All 16 submission tests passed, including the new environment check.

This TorchTitan version uses A100 peak FLOPs for A30 MFU values.
Do not use the recorded MFU for performance comparisons.
This short test does not establish convergence or memory limits at longer sequence lengths.
