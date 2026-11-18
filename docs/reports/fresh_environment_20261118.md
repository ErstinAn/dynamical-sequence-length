# Fresh environment verification: 2026-11-18

## Result

The independent Python environment passed all required dependency imports.
The final `pip check` reported `No broken requirements found.`
The original Torch, TorchTitan, TorchAO, Triton, and CUDA dependency versions remain unchanged.
The subsequent H200 smoke test and exact checkpoint recovery passed.
Read [the final GPU record](fresh_smoke_20261118.md) for results and limits.

Runtime prefix:

```text
/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/environment/conda
```

Read [the machine-readable record](fresh_environment_20261118.json) for exact package versions and Python paths.
Read [the installed package list](fresh_environment_20261118_freeze.txt) for the pip freeze output.

## Starting conditions

The test started in a new directory under the user-approved project storage.
The server runs Rocky Linux 9.8 on x86_64.
The selected runtime uses an independent Python 3.12.14 installation.
Its import paths do not include the original `torchtitan-macro` environment.

The test reused these server facilities:

- The Linux operating system and shell tools.
- The ARC `Miniforge3/25.11.0-1` manager for the first Python creation command.
- Existing network access and the user's W&B credentials for the access check.

The test did not reuse installed packages from the original training environment.
The test did not reuse its Python standard library in the selected runtime.
The test did not install original cached wheels.
It inspected old wheel metadata only to identify exact artifact names and hashes.

A preliminary virtual environment used the original Python executable as a bootstrap.
That virtual environment is not the selected runtime.
Its package-index install failed before runtime installation.

## Executed commands

The test created Python with this command:

```bash
module load Miniforge3/25.11.0-1
export ENV_ROOT=/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/environment
CONDA_PKGS_DIRS="$ENV_ROOT/conda-pkgs" conda create \
  --prefix "$ENV_ROOT/conda" --override-channels --channel conda-forge \
  python=3.12.14 pip --yes > "$ENV_ROOT/logs/python-bootstrap.log" 2>&1
```

The command used a new Conda package cache.
It installed a separate interpreter and standard library.

The initial installation used `requirements-lock.txt`.
It failed because the nightly index did not list `torch==2.15.0.dev20260830+cu130`.
The exact official Torch wheel still existed at its direct URL.
The TorchAO and Triton wheels also remained available.

The successful installation used these commands from the repository root:

```bash
export PYTHONNOUSERSITE=1
"$ENV_ROOT/conda/bin/python" -m pip install --cache-dir "$ENV_ROOT/pip-cache" \
  -r requirements-portable.txt > "$ENV_ROOT/logs/portable-install-03.log" 2>&1
"$ENV_ROOT/conda/bin/python" -m pip --no-cache-dir check
```

The final installation completed with exit code 0.
It downloaded the exact nightly wheels from official PyTorch URLs.
Pip checked the SHA256 values in the portable requirements file.
TorchTitan came from its pinned GitHub commit.
Pip built a wheel from that commit inside the new installation process.

The original lock omitted 16 transitive dependencies.
Five selected versions initially differed from the original environment.
The final constraints restore all 16 original versions.
The restored versions include `nvidia-nvjitlink==13.4.46rc1`.

The final installation reused downloads from earlier attempts in the new test cache.
It did not reuse the original environment's download cache.

## Hopper attention dependency

The GPU preparation review found a missing FlashAttention 3 dependency.
TorchTitan activates FA3 on H100 and H200 GPUs.
CPU model construction did not activate that GPU-specific path.
The original training environment also lacked this dependency.

The pinned TorchTitan source documents an [official PyTorch FA3 index](https://github.com/pytorch/torchtitan/blob/737594746fda65a6d94dc9482ef07863a80c8588/torchtitan/experiments/rl/README.md).
The test installed `flash-attn-3==3.0.0` from that index with an exact SHA256 check.
It also installed the declared dependency `ninja==1.13.0`.
The wheel SHA256 is `44a019ca428d0866fc73ca49b8af02eb4b6a37ae8a1a6b1a07917c383a187fe6`.

The test imported `flash_attn_interface` in the fresh environment.
It called `activate_flash_attention_impl("FA3")` successfully without a visible GPU.
It checked the registered forward and backward operator schemas against the pinned Torch adapter.
The final `pip check` passed after this addition.
The subsequent GPU probe checked real CUDA forward and backward operations at all four sequence lengths.

The attention implementation, model, Torch version, and TorchTitan version did not change.
The test added the missing runtime dependency without replacing the attention backend.
No source build was necessary.
The upstream source checkout was exploratory and is not an installed runtime dependency.

## Standalone bootstrap check

The test downloaded the official Miniforge `25.11.0-1` Linux x86_64 installer.
The official checksum file matched the downloaded installer.
The SHA256 was:

```text
be1bad9d4e67a8753eb76fb4940e9a08036786675c7adf060627e55791bf110d
```

The batch installer completed with exit code 0.
The installed manager reported `conda 25.11.0`.
It did not change shell startup files.

The standalone manager is at:

```text
/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/environment/miniforge
```

The separate Python creation check with this manager passed in CPU job `8796354`.
It used a new package cache at `environment/bootstrap-conda-pkgs`.
It created Python 3.12.14 at `environment/bootstrap-python`.
Independent imports of `ssl` and `sqlite3` passed.
The Python import paths contained only that new prefix and the working directory.

This separate check required no existing Python or Conda manager.
The active Torch prefix still uses the earlier Python creation result.
The test did not reinstall Torch into the separate bootstrap-check prefix.
Earlier direct attempts stopped before dispatch because automatic permission review timed out.
The CPU job completed the same Python creation command.
Slurm recorded `COMPLETED` with exit code `0:0` on node `tc130`.
The job requested one CPU, 4 GB of memory, and ten minutes.
It waited one second and ran for three minutes and thirty seconds.
It ran from `10:29:47` to `10:33:17` UTC on 2026-11-18.

## Imports and W&B

These imports passed in the selected environment:

- `torch`, `torchtitan`, `torchao`, and `triton`.
- `datasets`, `tokenizers`, `numpy`, and `pyarrow`.
- `huggingface_hub` and `wandb`.

Torch reported `2.15.0.dev20260830+cu130` and CUDA runtime `13.0`.
TorchTitan's installed provenance recorded commit `737594746fda65a6d94dc9482ef07863a80c8588`.

The fresh W&B client read project `rotation_optimizer/debug` successfully.
The client used existing credentials from the user's `.netrc` file.
The check did not create a run or check metric uploads.
The subsequent GPU smoke test checked uploads and token-based chart axes successfully.

## Installation logs

The log directory is:

```text
/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/environment/logs
```

| Log | Result |
| --- | --- |
| `pinned-install-01.log` | Original lock failed at the missing nightly index entry |
| `portable-install-01.log` | Direct entries after the lock did not resolve the collection failure |
| `portable-install-02.log` | Direct entries before the lock installed the runtime |
| `portable-install-03.log` | Supplemental constraints restored original transitive versions |
| `python-bootstrap.log` | Fresh Python creation passed |
| `miniforge-bootstrap.log` | Standalone Miniforge installation passed |
| `standalone-python-bootstrap.log` | New standalone manager created Python 3.12.14 with a new cache |

## Independent source and CPU checks

The test copied the current source into a new Git repository.
It created a Git bundle from that repository.
It cloned the bundle into a separate directory.
It excluded local assets and directory links.
It did not change the original Git repository.
These initial test bundles are private local evidence.
A later review found credential-like text in an archived conversation.
The bundles now permit owner access only.
They must not serve as delivery bundles.
The final delivery requires a new Git history that excludes that archive.

The source base was `91ef5297deae1d0cf96d9d22869151e68bd31142`.
The bundle also contained the current uncommitted changes.
The first snapshot commit was `c347606611841072d7aacc24d767393d1d7ba53e`.
Its first CPU suite ran 198 tests and found one error.
The dataset builder ignored the external assets path for its tokenizer.

The path update made the builder use `DYNAMIC_SEQ_LEN_ASSETS`.
It also accepted the downloaded tokenizer provenance after checking its hashes.
This path update did not change tokenization rules or dataset members.

The second snapshot commit is `fe7e5f69c6c38a2b727a8e509c47205c4c81e75a`.
Its bundle SHA256 is `b290a34f273ba6a91f43912fdc8aff88ab3a9bcfd4e31aaf2cf2923f4652fa7b`.
The file `source_snapshot-v2.json` in the test root records all 357 source hashes.
The selected checkout is:

```text
/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118/checkout-v2
```

The test installed this checkout with `pip install --no-deps --no-build-isolation -e`.
The full CPU suite ran 211 tests in 52.418 seconds.
It passed 206 tests and skipped five tests, with no failures or errors.
The skips concern separate opt-in checks and unused historical artifacts.
The opt-in distributed checks subsequently passed in the same fresh environment.
Those checks used the first snapshot because the tokenizer path update did not change their source.

| Check | Result |
| --- | --- |
| Full CPU suite on updated source | 211 tests: 206 passed, five skipped, no failures or errors |
| Distributed gradient variance and fixed validation | Two tests passed in 18.391 seconds |
| Two-rank token progress | Nominal counts, ten complete checkpoints, exact recovery, and validation axes passed |
| Smoke dry run | Eight steps, global batch 512, and 31,457,280 nominal tokens |

The [CPU log](fresh_environment_20261118_cpu_passed.txt) records the updated full suite.
The [Gloo log](fresh_environment_20261118_gloo.txt) records distributed reduction checks.
The [token progress log](fresh_environment_20261118_token_progress_dp.txt) records distributed token and recovery checks.

An additional model check compared the experiment with TorchTitan's native `1B` registry entry.
The model configurations matched after excluding function memory addresses from the comparison.
Construction on the `meta` device counted 1,235,814,400 parameters.
The vocabulary size was 128,256, with tied input and output weights.
This check allocated no parameter storage and ran no optimizer updates.

## Subsequent checks and limits

The independent [dataset reconstruction](fresh_data_20261118.md) passed all shard and validation checks.
The [H200 smoke](fresh_smoke_20261118.md) checked training, full fixed validation, W&B, checkpoints, and exact recovery.
This record does not claim a complete test on another physical server.
No formal 20B-token experiment or learning-rate sweep ran.
