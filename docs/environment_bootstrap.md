# Install the Python environment

Use this procedure on a Linux x86_64 server.
Run each command in Bash.
Use an account that can write to the selected storage directory.
The procedure does not need root access or an existing Python installation.

The tested runtime uses these versions:

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.15.0.dev20260830+cu130 |
| CUDA runtime in PyTorch | 13.0 |
| TorchTitan | Git commit `737594746fda65a6d94dc9482ef07863a80c8588` |
| TorchAO | 0.19.0.dev20260831+cu130 |
| Triton | 3.8.0+gitb252c7c4 |
| FlashAttention 3 | 3.0.0 |
| Ninja | 1.13.0 |
| W&B | 0.29.0 |

The project uses private TorchTitan interfaces.
Keep the specified versions.
Do not replace the pinned nightly wheels with current nightly wheels.

## 1. Check the server

Run these commands from your login directory:

```bash
uname -s
uname -m
ldd --version
command -v bash curl git sha256sum
```

Require Linux and `x86_64`.
Use glibc 2.31 or later for this wheel set.
If a command is missing, ask the server administrator to provide that command.
This procedure does not install system packages or GPU drivers.

Run this command inside a GPU allocation:

```bash
nvidia-smi
```

Check the GPU type, GPU memory, and driver version.
NVIDIA lists driver 580 or later for CUDA 13.x minor compatibility.
JIT compilation can need a newer driver within that family.
Require the GPU smoke test before accepting the driver.
See the [NVIDIA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).

## 2. Select storage

Replace `/absolute/path/to/project-storage` with a writable directory.
Use shared storage if compute nodes must use the environment.
Do not select a login-node-only temporary directory.

```bash
export RUN_ROOT=/absolute/path/to/project-storage
export ENV_ROOT="$RUN_ROOT/environment"
mkdir -p "$ENV_ROOT/bootstrap" "$ENV_ROOT/logs" "$ENV_ROOT/pip-cache" "$ENV_ROOT/conda-pkgs"
df -h "$RUN_ROOT"
```

Check the storage quota separately from filesystem free space.
Keep space for the runtime, downloaded packages, and installation files.
Reserve 20 GiB for these files before installation.
The measured runtime, including FA3, used 7.5 GiB.
Its pip cache used 2.2 GiB.
The reserve also covers Miniforge, Conda packages, temporary files, and logs.

Dataset files and checkpoints need additional space.
See [the main guide](codebase_kick_off/getting_started.md) for those files.

## 3. Install Miniforge

Miniforge supplies Python and the Conda package manager.
This procedure pins the installer to release `25.11.0-1`.
See the [official release](https://github.com/conda-forge/miniforge/releases/tag/25.11.0-1).

Run these commands from the bootstrap directory:

```bash
cd "$ENV_ROOT/bootstrap"
curl --fail --location --retry 3 --output Miniforge3-25.11.0-1-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-25.11.0-1-Linux-x86_64.sh
curl --fail --location --retry 3 --output Miniforge3-25.11.0-1-Linux-x86_64.sh.sha256 \
  https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-25.11.0-1-Linux-x86_64.sh.sha256
sha256sum --check Miniforge3-25.11.0-1-Linux-x86_64.sh.sha256
```

Require `OK` from the checksum command.
The expected SHA256 is `be1bad9d4e67a8753eb76fb4940e9a08036786675c7adf060627e55791bf110d`.
Stop if the hash differs.
Download the installer again after checking the URL and available disk space.

Use an unused installation path:

```bash
bash Miniforge3-25.11.0-1-Linux-x86_64.sh -b -p "$ENV_ROOT/miniforge" \
  > "$ENV_ROOT/logs/miniforge-bootstrap.log" 2>&1
"$ENV_ROOT/miniforge/bin/conda" --version
```

Require exit code 0 and `conda 25.11.0`.
The batch installer does not change shell startup files.
Read `miniforge-bootstrap.log` if installation fails.

## 4. Create an isolated Python environment

Run this command from any directory:

```bash
CONDA_PKGS_DIRS="$ENV_ROOT/conda-pkgs" "$ENV_ROOT/miniforge/bin/conda" create \
  --prefix "$ENV_ROOT/conda" --override-channels --channel conda-forge \
  python=3.12.14 pip --yes > "$ENV_ROOT/logs/python-bootstrap.log" 2>&1
```

Require exit code 0.
Read `python-bootstrap.log` if the package download or installation fails.
The independent Python creation test took three minutes and thirty seconds on one ARC CPU.
That measurement included fresh package downloads and installation on shared storage.
Network speed and filesystem performance can change this time.
Use the explicit interpreter path for all later commands:

```bash
export PYTHONNOUSERSITE=1
export PYTHON="$ENV_ROOT/conda/bin/python"
"$PYTHON" --version
"$PYTHON" -c 'import sys; print(sys.prefix); print(*sys.path, sep="\n")'
```

Require Python `3.12.14`.
Require paths inside the new environment.
The current working directory can also appear in `sys.path`.
Do not add another environment to `PYTHONPATH`.
Do not use `--system-site-packages`.

## 5. Install the locked runtime

First obtain the exact delivery source version from [the main guide](codebase_kick_off/getting_started.md).
Run these commands from that repository root:

```bash
"$PYTHON" -m pip install --cache-dir "$ENV_ROOT/pip-cache" \
  -r requirements-portable.txt > "$ENV_ROOT/logs/runtime-install.log" 2>&1
"$PYTHON" -m pip install --no-deps --no-build-isolation -e . \
  > "$ENV_ROOT/logs/project-install.log" 2>&1
"$PYTHON" -m pip --no-cache-dir check
```

Require `No broken requirements found.`
The portable requirements file preserves the original version lock.
It adds exact wheel URLs and SHA256 checks for three nightly packages.
It also fixes dependency versions omitted from the original lock.
It adds the FlashAttention 3 wheel required by native TorchTitan attention on H100 and H200 GPUs.
The pinned TorchTitan source documents the [official FA3 package index](https://github.com/pytorch/torchtitan/blob/737594746fda65a6d94dc9482ef07863a80c8588/torchtitan/experiments/rl/README.md).
This prebuilt wheel does not require a local CUDA compiler.

The package index no longer lists the pinned Torch nightly build.
Therefore, use `requirements-portable.txt` for a new installation.
Keep direct wheel entries before the original lock include.
This order lets pip resolve the historical nightly versions.

If installation fails, inspect the relevant log.
For a missing wheel, record the package name and exact URL.
For a checksum failure, remove only the damaged downloaded artifact.
Do not upgrade a package to bypass an installation failure.

## 6. Check imports and record versions

Run these commands from the repository root:

```bash
"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import json
import torch
import torchao
import torchtitan
import datasets
import huggingface_hub
import tokenizers
import wandb
import flash_attn_interface
from torch.nn.attention import activate_flash_attention_impl

assert torch.__version__ == "2.15.0.dev20260830+cu130"
assert torch.version.cuda == "13.0"
origin = json.loads(metadata.distribution("torchtitan").read_text("direct_url.json"))
assert origin["vcs_info"]["commit_id"] == "737594746fda65a6d94dc9482ef07863a80c8588"
assert metadata.version("flash-attn-3") == "3.0.0"
activate_flash_attention_impl("FA3")
assert hasattr(torch.ops.flash_attn_3, "fwd")
assert hasattr(torch.ops.flash_attn_3, "bwd")
print("IMPORT_CHECK_PASSED")
print("FA3_REGISTRATION_PASSED")
print(torch.__version__, torch.version.cuda)
print(torchtitan.__file__)
PY
"$PYTHON" -m pip --no-cache-dir freeze > "$ENV_ROOT/logs/pip-freeze.txt"
"$ENV_ROOT/miniforge/bin/conda" list --prefix "$ENV_ROOT/conda" --explicit \
  > "$ENV_ROOT/logs/conda-explicit.txt"
```

Require `IMPORT_CHECK_PASSED`.
Require `FA3_REGISTRATION_PASSED`.
Require a TorchTitan path inside the new environment.
A login node can pass these checks without a visible GPU.
FA3 registration checks the compiled extension and operator definitions.
The GPU probe must also check its forward and backward operations.

Inside a GPU allocation, run:

```bash
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available()
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(index, properties.name, properties.total_memory)
x = torch.ones(1, device="cuda")
assert x.item() == 1
print("CUDA_CHECK_PASSED")
PY
```

Require `CUDA_CHECK_PASSED` and the expected number of allocated GPUs.
If the check fails, compare the driver with the CUDA runtime requirement.
Check Slurm GPU allocation and `CUDA_VISIBLE_DEVICES` next.
Do not run training on a login node.

## VT ARC and verification limits

VT ARC also provides Miniforge modules.
Its [Python documentation](https://www.docs.arc.vt.edu/software/python.html) describes module and isolated-environment workflows.
The standalone installer above also works without ARC modules.

The independent test used a fresh Python prefix and fresh package downloads.
The first Python creation command used the ARC Miniforge manager.
The test also downloaded, checked, and installed the standalone Miniforge manager.
The separate Python creation check with that new manager also passed.
That check used another empty package cache and a new Python prefix.
The test kept the existing fresh Torch prefix for GPU acceptance.

See [the actual verification record](reports/fresh_environment_20261118.md) for exact results and limitations.
