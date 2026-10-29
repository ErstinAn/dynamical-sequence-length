# Cleanup and experiment record

## Unchanged behavior

The cleanup keeps the 100M model and the three sequence length experiments.
It keeps the batch sizes, learning rate, token budgets, and validation identities.
AdamW and Muon keep their parameter groups and update settings.
The variance calculation does not change.

## Removed components

The new package does not contain SOAP, rotation providers, or the custom ZeRO-1 distributor.
It does not contain the unused Qwen, 430M, or 1B model registrations.
The source archive contains the previous code and experiment records.
The archive also contains the previous job IDs and account-specific files.

## Configuration changes

The new experiment files use `optimizer.method` with `adamw` or `muon`.
They do not use `optimizer.rotation_method` or a separate optimizer YAML file.
The vector group always uses AdamW.
The optimizer configuration is part of the normal TorchTitan configuration log.

W&B is off by default.
The public files do not contain a personal account or storage directory.
Slurm resources belong in a local configuration file.
The launcher no longer changes historical scratch paths to a personal project path.

## Checkpoint scope

The optimizer comparison checks state recovery after step 5.
Both optimizer paths continue to use TorchTitan checkpoint utilities.
The new Slurm tools use manifest version 2 with a source hash.
For previous submission manifests, use the archived source.
The cleanup does not rewrite existing checkpoints or historical manifests.

## Numerical comparison

The CPU comparison used seed 42 and deterministic operations.
It ran ten steps for each optimizer on a small language model.
All loss, gradient norm, and variance values were the same.
All 40 model and optimizer state tensors were the same.
The [result file](refactor_results.json) contains the measured values.

These results do not prove convergence on the full FineWeb experiments.
The cleanup does not change the mathematical training method.

## Validation results

The CPU suite contains 58 tests.
The normal command passed 55 tests and skipped three optional tests.
Separate commands passed all three optional tests.
The installed-source test run then passed all 58 tests with the optional checks enabled.
The optional tests check distributed reductions and the local validation artifacts.

The GPU job used two L40S GPUs on Falcon.
Job 723916 completed with exit code 0 on October 29, 2026.
The requested walltime was 20 minutes.
The actual queue wait was one second.
The execution time was three minutes and four seconds.

The GPU tests passed all four FP32/BF16 and MSE/cross-entropy cases.
Each case used ten training steps.
The full model launch also completed ten steps.
The GPU comparison found the same bytes in all model and optimizer state tensors.
The loss, gradient norm, and variance values also had the same bytes.
The [GPU comparison](gpu_refactor_results.json) contains the measured values.

The wheel installation test passed with the existing runtime dependencies.
The installed wheel loaded the data manifests and completed an optimizer update.
The code checks and text checks passed.
The [validation record](validation_results.json) contains the job times and test measurements.

## Package size

The Python implementation decreased from 6,945 lines to 2,516 lines.
These counts exclude tests and comparison tools.
The data arrays stay outside Git.
The local archive keeps the previous files for recovery.
