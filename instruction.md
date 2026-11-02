# First run on a new server

This procedure is for a collaborator or an agent working in a fresh checkout.
Run shell commands from the repository root unless a step says otherwise.
Use the same Git commit on both servers. Record every failed or skipped check.
Do not change experiment settings to conceal an installation or data error.

## 1. Check out the agreed source

```bash
git clone https://github.com/ErstinAn/dynamical-sequence-length.git
cd dynamical-sequence-length
git checkout <DELIVERY_COMMIT>
git status --short
git rev-parse HEAD
```

Replace `<DELIVERY_COMMIT>` with the SHA supplied by the maintainer.
The working tree should be clean. Do not copy another project's Python files into this checkout.
Read [the delivery inventory](docs/delivery.md) before transferring additional files.

## 2. Inspect the server

Record the OS, CPU architecture, Python version, GPU model, GPU memory, and driver version.
Check free disk space and compute-node access to the asset and output directories.
Check network access from a compute node, not only the login node.

```bash
uname -sm
python3.12 --version
df -h .
```

Run `nvidia-smi` inside a GPU allocation.
Use the target server's official scheduling documentation and live allocation information.
This launcher supports one node. Start one launcher process, which starts one worker per GPU.
Do not copy VT ARC account, partition, QoS, or module settings to another server.

## 3. Create the Python environment

The reference runtime uses Python 3.12 and the exact versions in `requirements-lock.txt`.
PyTorch is `2.15.0.dev20260830+cu130`.
TorchTitan is fixed at `737594746fda65a6d94dc9482ef07863a80c8588`.
Check driver compatibility with this CUDA build using the target site's documentation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip check
python -c 'import torch, torchtitan; print(torch.__version__, torch.version.cuda); print(torchtitan.__file__)'
```

The existing environment has been tested. A fresh download of all locked dependencies has not been established.
Old nightly wheels may be unavailable. Report the exact missing distribution instead of installing a newer version silently.

Request a matching wheel collection, or test a dependency migration on a separate branch.
Do not copy a virtual environment between unrelated servers.
An offline wheel collection must include the pinned TorchTitan build and all transitive dependencies.

## 4. Select storage paths

The following defaults work inside the checkout. Replace them with suitable absolute paths when necessary.

```bash
export DYNAMIC_SEQ_LEN_ASSETS="$PWD/assets"
export DYNAMIC_SEQ_LEN_OUTPUT="$PWD/outputs"
export HF_HOME="$DYNAMIC_SEQ_LEN_OUTPUT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$DYNAMIC_SEQ_LEN_OUTPUT/cache/torch"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$DYNAMIC_SEQ_LEN_ASSETS" "$DYNAMIC_SEQ_LEN_OUTPUT"
```

Use storage visible to the allocated compute node.
Reserve space for caches and checkpoints in addition to datasets.
The 1B smoke saves two full checkpoints, each containing model and optimizer state.
Allow at least 60 GiB of free output space for the smoke and its recovery check.

## 5. Prepare the tokenizer

Both the 100M experiments and the 1B variant use the existing Llama2 tokenizer.
The source is `hf-internal-testing/llama-tokenizer`, at the revision recorded in `tokenizer_source.json`.
The script downloads three files, verifies their SHA256 values, and copies the source record.
It does not download pretrained model weights.

```bash
python scripts/prepare_tokenizer.py \
  --output "$DYNAMIC_SEQ_LEN_ASSETS/hf/llama-tokenizer"
```

For an offline compute node, run this command on a connected machine and transfer the complete resulting directory.
Include `tokenizer_source.json`. It is part of the validation artifact identity.
Do not substitute the original Llama3 tokenizer for this 32k-vocabulary model variant.

## 6. Run CPU checks

```bash
PYTHONPATH=.:cluster/vt_arc OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -t . -v
python scripts/train.py --list
python scripts/train.py configs/smoke.yaml --nproc 1 --dry-run
```

The extra Python path permits imports of scheduler test helpers. It does not require a Slurm service.
The optional artifact and distributed tests can skip during this first pass.
Investigate any failure. A dry run checks command construction, not GPU execution or dataset availability.

## 7. Run the small GPU example

Obtain one GPU using the target server's normal procedure. Activate the same environment inside the allocation.
Retain the path variables from step 4.

```bash
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python scripts/train.py configs/smoke.yaml --nproc 1
```

Expected result: ten optimizer steps, finite training metrics, and exit code zero.
This uses a tiny model and `tests/fixtures/text.jsonl`. FineWeb and W&B are not required.

## 8. Run the 1B example

This variant reuses TorchTitan's 1B Transformer backbone with a 32000-entry vocabulary.
It has 1,038,682,112 unique parameters. Embedding and output weights are tied.
The original upstream 128256-entry vocabulary is not used in this variant.
See [the model description](docs/llama3_1b.md).

```bash
python scripts/train.py configs/examples/llama3_1b_smoke.yaml --nproc 1 --dry-run
python scripts/train.py configs/examples/llama3_1b_smoke.yaml --nproc 1
```

This runs ten AdamW updates on local text with length 128, global batch four, and microbatch one.
Model storage is FP32, forward parameters are BF16, and gradient reduction is FP32.
Compilation and W&B are disabled. Full checkpoints are saved at steps five and ten.

The run prints its output directory under `$DYNAMIC_SEQ_LEN_OUTPUT/runs/llama3-1b-first-run/`.
Require finite loss and gradient norms, both completed checkpoints, and exit code zero.
Record peak GPU memory and step time. Do not infer full-length memory requirements from this short example.

The verified device was one 48 GB L40S. Training used 21.81 GiB and recovery peaked at 24.68 GiB.
A 24 GB GPU is not sufficient for the measured recovery peak.

Each YAML launch creates a new run directory.
`experiment.yaml` records YAML settings. `command.txt` also records command-line overrides.
Preserve both files and the native configuration log when reporting the effective experiment settings.

## 9. Test checkpoint recovery

Use the completed 1B smoke from step 8. Set `FIRST_RUN` to its absolute run directory.
Use a new recovery directory and keep the original experiment for comparison.

```bash
export FIRST_RUN=/absolute/path/to/llama3-1b-first-run/RUN_ID
export RECOVERY_ROOT="$DYNAMIC_SEQ_LEN_OUTPUT/recovery-check"
test ! -e "$RECOVERY_ROOT"
mkdir -p "$RECOVERY_ROOT"
cp -a "$FIRST_RUN/checkpoint/step-5" "$RECOVERY_ROOT/step-5"
python scripts/train.py configs/examples/llama3_1b_smoke.yaml --nproc 1 \
  --checkpoint.folder "$RECOVERY_ROOT" --checkpoint.load_step 5
```

Expected result: load step five, perform steps six through ten, and save a complete step-ten checkpoint.
This is a recovery branch with a new log directory. It is not an in-place continuation of the original run ID.
Keep the original training horizon, tokenizer, batch topology, source, and optimizer settings.
For dynamic experiments, schedule signatures also enforce these constraints.
Do not run two writers against the same checkpoint directory.

## 10. Prepare FineWeb validation

This step is required before any FineWeb training, even if evaluation is disabled.
The training loader needs the validation document exclusion index.
Run downloads on a connected node and transfer files if compute nodes are offline.

```bash
export SOURCE_PARQUET="$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id="HuggingFaceFW/fineweb",
    repo_type="dataset",
    revision="9bb295ddab0e05d785b879661af7260fed5140fc",
    filename="sample/100BT/000_00000.parquet",
))
PY
)"
python scripts/prepare_fineweb100_validation.py \
  --source-parquet "$SOURCE_PARQUET" \
  --tokenizer "$DYNAMIC_SEQ_LEN_ASSETS/hf/llama-tokenizer" \
  --seq-len 1024 \
  --output "$DYNAMIC_SEQ_LEN_ASSETS/data/fineweb100_validation_llama2_1024"
```

The source parquet is approximately 2 GiB. The generated artifact is approximately 484 MiB.
The preparation script refuses to overwrite an existing artifact.
If the artifact already exists, verify it using the following command.
Do not delete or modify reference hashes to make a mismatch pass.

```bash
python - <<'PY'
import os
from pathlib import Path
from torchtitan_dynamic_seq_len.data.fixed_validation import FrozenValidationArtifact
from torchtitan_dynamic_seq_len.data.fineweb100 import load_evaluation_exclusions
assets = Path(os.environ["DYNAMIC_SEQ_LEN_ASSETS"])
path = assets / "data/fineweb100_validation_llama2_1024"
artifact = FrozenValidationArtifact(
    path, token_budget=20_971_520, sequence_length=1024,
    tokenizer_path=assets / "hf/llama-tokenizer",
)
ids, hashes = load_evaluation_exclusions(path)
print("Verified:", artifact.num_sequences, "sequences;", len(ids), "excluded documents")
PY
```

Use the artifact length selected by the experiment's validation settings.
The full artifact test in `docs/data.md` requires all five Llama2 lengths.
Training length and validation length can differ.

## 11. Choose training data access

Online mode uses the YAML value `dataloader.dataset_path: HuggingFaceFW/fineweb`.
The compute node must reach Hugging Face throughout streaming.
A partially populated cache is not a complete offline dataset.

Offline mode requires all 149 training files, totaling approximately 279.77 GiB.
Download only the paths listed in the committed manifest, at its fixed revision.
Preserve `<FINEWEB_ROOT>/sample/100BT/*.parquet` and verify every file against the manifest SHA256 before transfer acceptance.
The runtime checks names and sizes, but does not recalculate every training shard hash at each launch.
The held-out `000_00000.parquet` must never enter the training split.

The following optional command downloads and verifies the complete training snapshot.
Run it only after allocating enough storage and download time.

```bash
export FINEWEB_ROOT=/absolute/path/to/fineweb-snapshot
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from torchtitan_dynamic_seq_len.data.fineweb100 import load_manifest, file_sha256
m = load_manifest()
root = Path(os.environ["FINEWEB_ROOT"])
snapshot_download(repo_id=m["repo_id"], repo_type="dataset", revision=m["revision"],
                  allow_patterns=m["training_files"], local_dir=root)
for entry in m["source_files"]:
    if entry["path"] in m["training_files"]:
        path = root / entry["path"]
        assert path.stat().st_size == entry["size"], path
        assert file_sha256(path) == entry["sha256"], path
print("Verified all 149 training shards")
PY
```

## 12. Run a real-data integration check

After step 10, run this command with online training access:

```bash
python scripts/train.py configs/examples/llama3_1b_fineweb.yaml --nproc 1
```

For a verified offline snapshot, add `--dataloader.dataset_path "$FINEWEB_ROOT"`.
This runs ten training updates, saves checkpoints, and evaluates 20,971,520 held-out tokens.
Evaluation can take much longer than the ten training updates. Request sufficient walltime.
This is an integration check, not a full pretraining or convergence experiment.

## 13. Select the full experiment

Choose an existing experiment YAML and copy it into ignored `local/` storage for server-specific settings.
The 1B registry entries are listed in `docs/llama3_1b.md`.
Do not treat a 100M token budget or learning rate as an established 1B training recipe.
For comparable results, agree on model size, tokenizer, optimizer, schedules, seed, and token budget before running.

For replicated data parallelism, every scheduled global batch must be divisible by GPU count times microbatch size.
Dynamic batch training requires zero data-loader workers and no prefetch.
Test the largest scheduled memory combinations before a full run.
The 16384-token gradient-distance experiment performs additional probe passes and needs a separate resource estimate.

Use `--nproc N` only when the allocation exposes exactly the intended N GPUs to the launcher.
The local launcher ignores the YAML `resources` block. Editing that block does not allocate GPUs.
For Slurm, place this command inside a site-specific single-task allocation wrapper.
For sweeps, select each point with `--index`. The local launcher does not run all points automatically.

## 14. Return the acceptance report

Report the Git SHA, dependency versions, TorchTitan revision, GPU model, driver, and effective launch command.
Include asset identities, output paths, completed steps, checkpoint recovery results, peak memory, and measured elapsed time.
Separate passed tests, skipped tests, and untested conditions.
Never include authentication tokens in logs, commits, or the report.
If a required check fails, report the exact failure and the next necessary action.
