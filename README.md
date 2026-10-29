# Sequence length experiments

This project measures gradient variance during language model training.
The experiments use a 100M Llama3 model and the Llama2 tokenizer.
TorchTitan controls the training loop.

Each training run uses one sequence length and one batch size.
The code does not change these values during a run.
The optimizer can be AdamW or Muon.

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

## Paths and configuration

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

| Path | Contents |
| --- | --- |
| `torchtitan_dynamic_seq_len/config_registry.py` | Model definitions and default settings. |
| `torchtitan_dynamic_seq_len/grad_var_*.py` | Variance calculation and training integration. |
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
