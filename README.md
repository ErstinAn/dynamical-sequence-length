# Sequence length experiments

This project measures the effects of sequence length and gradient variance during language model training.
The experiments support 100M and approximately 1B Llama3 models with the Llama2 tokenizer.
TorchTitan controls the training loop.

For a new server, follow [instruction.md](instruction.md).
It includes environment setup, data preparation, a 1B example, and checkpoint recovery.
See the [1B model description](docs/llama3_1b.md) and [delivery inventory](docs/delivery.md).

The fixed-length configurations use one sequence length and batch size per training run.
`DynamicSeqLenTrainer` changes sequence length between optimizer steps and keeps batch sizes fixed.
`DynamicSeqBSTrainer` uses independent schedules for sequence length, global batch size, and microbatch size.
The optimizer can be AdamW or Muon.

Update 2026-10-31: dynamic training uses the primary repository.
The repository includes six experiment configurations and a sweep that compares two cyclic periods.
See the [dynamic training guide](docs/dynamic_seq_len.md) for configuration and submission.

Update 2026-11-02: three independent schedules can change sequence length and both batch sizes.
See the [dynamic batch guide](docs/dynamic_seq_bs.md) for configuration, metrics, and recovery.

## Requirements

- Python 3.12.
- The package versions in `requirements-lock.txt`.
- An NVIDIA GPU for training.
- Network access for the first data download.

The recorded environment uses PyTorch `2.15.0.dev20260830+cu130`.
The TorchTitan source revision is `737594746fda65a6d94dc9482ef07863a80c8588`.
The installed package version alone does not identify this source revision.

The PyTorch build is a nightly build.
The package server can remove previous nightly builds.
A saved wheel for the same build is then necessary for a new installation.
The [environment record](docs/environment.md) gives the test limits.

## Installation

Run these commands from the repository directory.

1. Create a Python environment.

   ```bash
   python3.12 -m venv .venv
   ```

2. Activate the environment.

   ```bash
   source .venv/bin/activate
   ```

3. Install the recorded dependencies.

   ```bash
   python -m pip install -r requirements-lock.txt
   ```

4. Install this package for development.

   ```bash
   python -m pip install --no-deps -e .
   ```

5. Download the tokenizer files.

   ```bash
   python scripts/prepare_tokenizer.py
   ```

The tokenizer script checks the source revision and file hashes.
The repository does not contain the tokenizer files or the validation arrays.

## First training test

The smoke test uses a small model and a local text file.
FineWeb data and a W&B account are not necessary for this test.
It runs for ten steps with one GPU.

1. Check the command.

   ```bash
   python scripts/train.py configs/smoke.yaml --nproc 1 --dry-run
   ```

2. Run the test in a GPU allocation.

   ```bash
   python scripts/train.py configs/smoke.yaml --nproc 1
   ```

The default output directory is `outputs/runs/`.
Each YAML launch creates a new run directory.
The directory contains the experiment settings and the launch command.

## FineWeb experiments

The [data procedure](docs/data.md) prepares the validation arrays.
These arrays are necessary for the training document exclusion check.

| Configuration | Sequence length | Global batch | Local batch | Steps |
| --- | ---: | ---: | ---: | ---: |
| `configs/seq512.yaml` | 512 | 512 | 16 | 7637 |
| `configs/seq1024.yaml` | 1024 | 512 | 16 | 3819 |
| `configs/seq2048.yaml` | 2048 | 512 | 16 | 1910 |

The reference experiments use two GPUs.
Each experiment processes approximately two billion label tokens.
The configuration files keep the previous learning rate and token budgets.
W&B is off by default.

Run one experiment in a two-GPU allocation.

```bash
python scripts/train.py configs/seq512.yaml --nproc 2
```

To use Muon, set `optimizer.method` in the experiment file to `muon`.
The other parameter group continues to use AdamW.
The [method description](docs/method.md) defines the parameter groups and variance measurement.

## Constant learning rate sweep

Use [configs/lr_sweep_constant.yaml](configs/lr_sweep_constant.yaml) to compare five learning rates with fixed batch size and sequence length.
The `llama3_100m_llama2_adamw_constant_lr` entry selects the TorchTitan `Trainer` and `MetricsProcessor`.
It uses the same 100M model and Llama2 tokenizer as the reference experiments.
The AdamW parameter groups and validation procedure are also the same.
It does not calculate gradient variance.

| Setting | Value |
| --- | --- |
| Sequence length | 1024 |
| Global batch, in sequences | 512 |
| Local batch, in sequences | 16 |
| Training steps | 3819 |
| DataLoader workers | 0 |
| Learning rate candidates | 0.0005, 0.001, 0.002, 0.004, 0.008 |
| Warmup steps | 0 |
| Seed | 1 |

Each sweep point uses its selected learning rate from the first optimizer update to the last.
The TorchTitan scheduler gives this constant rate with `warmup_steps=0` and `min_lr_factor=1.0`.
The linear decay calculation gives a multiplier of one during the full training run.
The sweep changes only `optimizer.learning_rate`.
Batch size and sequence length stay fixed.
With two GPUs, each optimizer step accumulates 16 local batches per rank.

Examine the first sweep command without a training run:

```bash
python scripts/train.py configs/lr_sweep_constant.yaml --nproc 2 --index 0 --dry-run
```

Run all five points sequentially inside a two-GPU allocation:

```bash
for index in 0 1 2 3 4; do
  python scripts/train.py configs/lr_sweep_constant.yaml --nproc 2 --index "$index" || break
done
```

Each point creates a independent run directory.
The default configuration does not send data to W&B.
Prepare the 1024-token validation artifact before training.
Compare validation loss at the same step across runs.
For Slurm arrays, follow the [ARC procedure](cluster/vt_arc/README.md) with a local copy of this configuration.

## Dynamic sequence length

`DynamicSeqLenTrainer` uses the existing `scripts/train.py` and `cluster/vt_arc/submit.py` scripts.
The scheduler is a separate component in `torchtitan_dynamic_seq_len/schedulers/sequence_length.py`.

The configurations in `configs/dynamic_seq/` include constant, increasing, decreasing, triangular, and two cyclic settings.
All six full configurations use 3840 optimizer steps and constant LR `0.0005`.
The global batch size is 512 sequences, and the local batch size is 16 sequences.
Their training budget is 2,013,265,920 label tokens.
Validation stays at sequence length 1024 while training length changes.

```bash
python scripts/train.py configs/dynamic_seq/smoke.yaml --nproc 1 --dry-run
python scripts/train.py configs/dynamic_seq/cyclic.yaml --nproc 2
python scripts/train.py configs/dynamic_seq/cyclic_sweep.yaml --nproc 2 --index 1
```

The cyclic sweep compares periods of 256 and 64 steps.
These periods give 15 and 60 cycles, respectively.

Full configurations include Slurm resource settings.
Before submission, replace `resources.account` with your allocation account.
For online W&B records, use the setup procedure in the dynamic training guide.

The [dynamic training guide](docs/dynamic_seq_len.md) gives information about configurations, submission files, metrics, and checkpoint rules.
The [ARC procedure](cluster/vt_arc/README.md) gives instructions for job submission and recovery.

## Paths and configuration

For per-step gradient comparisons across sequence lengths, use the [sequence gradient distance experiment](docs/seq_grad_distance.md).
It reads a shared 16384-token parent batch and updates with random 2048-token windows at global batch size 128.
W&B records all candidate distances every step and sequence-length profiles at configured intervals.

The experiment YAML contains the training settings.
The Python registry defines the models and default values.
Command-line options after the launcher options override the YAML settings.

| Setting | Function |
| --- | --- |
| `DYNAMIC_SEQ_LEN_ASSETS` | Changes the asset directory. |
| `DYNAMIC_SEQ_LEN_OUTPUT` | Changes the output and cache directories. |
| `DYNAMIC_SEQ_LEN_PYTHON` | Selects Python for the shell and Slurm scripts. |
| `paths.validation_dir` | Selects the validation artifact for a YAML run. |

The YAML reader expands `${PROJECT_ROOT}`, `${ASSETS_DIR}`, and `${OUTPUT_ROOT}`.
The default paths are relative to this repository.
Direct registry launches use `DYNAMIC_SEQ_LEN_VALIDATION_DIR` to select validation data.

## Source files

The package groups each Trainer with its metrics and monitor.
Shared schedules are in `schedulers/`, and data components are in `data/`.

```text
torchtitan_dynamic_seq_len/
  config_registry.py
  training_config.py
  optimizer.py
  run_config.py
  dynamic_seq/
    trainer.py
    metrics.py
  dynamic_seq_bs/
    trainer.py
    metrics.py
  grad_var/
    trainer.py
    metrics.py
    monitor.py
  seq_grad/
    trainer.py
    metrics.py
    monitor.py
  schedulers/
    sequence_length.py
    batch_size.py
  data/
```

Both global batch and microbatch schedulers use `schedulers/batch_size.py` with separate configuration objects.
The YAML configuration names and training commands stay the same.
Python imports use the module paths shown below.

| Path | Contents |
| --- | --- |
| `torchtitan_dynamic_seq_len/config_registry.py` | Model definitions and default settings. |
| `torchtitan_dynamic_seq_len/dynamic_seq/` | Sequence length Trainer and metrics. |
| `torchtitan_dynamic_seq_len/dynamic_seq_bs/` | Sequence and batch Trainer and metrics. |
| `torchtitan_dynamic_seq_len/schedulers/sequence_length.py` | Sequence lengths for each optimizer step. |
| `torchtitan_dynamic_seq_len/schedulers/batch_size.py` | Global batch and microbatch schedules with shared calculation rules. |
| `configs/dynamic_seq_bs/` | Independent schedules for sequence length and both batch sizes. |
| `configs/dynamic_seq/` | Dynamic configurations and the sweep for cyclic periods. |
| `torchtitan_dynamic_seq_len/grad_var/` | Gradient variance Trainer, metrics, and monitor. |
| `torchtitan_dynamic_seq_len/seq_grad/` | Sequence gradient Trainer, metrics, and monitor. |
| `torchtitan_dynamic_seq_len/optimizer.py` | AdamW and Muon parameter groups. |
| `torchtitan_dynamic_seq_len/data/` | Data preparation, exclusion checks, and validation. |
| `configs/` | Experiment settings. |
| `scripts/` | Local launch and data tools. |
| `cluster/vt_arc/` | Optional Slurm tools. |
| `tests/` | CPU tests and GPU tests. |

The [ARC procedure](cluster/vt_arc/README.md) describes job arrays and checkpoint recovery.
The [experiment record](docs/experiments.md) describes the test results and migration limits.

## Tests

Install the development tools.

```bash
python -m pip install --no-deps -e '.[dev]'
python -m pip install ruff==0.17.3 pre-commit==4.7.0
```

Run the CPU tests.

```bash
PYTHONPATH=.:cluster/vt_arc OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -t . -v
```

Run the code checks.

```bash
pre-commit run --all-files
```

The [test procedure](docs/tests.md) gives the distributed and GPU commands.
The [writing guide](docs/writing.md) defines the project terms and STE checks.
The [source notice](NOTICE.md) identifies external code and assets.
