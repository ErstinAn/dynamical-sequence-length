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

Run the local training smoke test from the main README.
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
