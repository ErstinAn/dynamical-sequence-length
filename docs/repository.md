# Repository map

Use [getting_started.md](getting_started.md) for the current experiment.
`python scripts/train.py --list` displays its single entry, `llama3_1b`.
Historical builders remain available through `--list-legacy`.

## Main dependencies

```text
configs/smoke.yaml or local/generated_configs/experiment.yaml
  -> scripts/train.py
     -> experiment.py: SequenceExperimentTrainer
        -> dynamic_seq_bs: batch and microbatch schedules
           -> dynamic_seq: sequence schedules and complete-step checks
              -> pinned TorchTitan: model and forward/backward implementation
        -> token_metrics.py: existing losses against n_tokens_seen
        -> token_progress.py: ten full checkpoint saves
        -> data/long_documents.py -> data/packed_documents.py
           -> dataset manifest, orders/train.npy, and token caches
        -> data/fixed_validation.py -> fixed 936-window artifact
```

Python paths are relative to `torchtitan_dynamic_seq_len/`, except `scripts/train.py`.
The main entry uses the upstream native-vocabulary 1B definition and its existing Varlen attention backend.
It retains the data package's finite ordered stream, packing rules, labels, and document masks.

## Directory decisions

| Path | Decision and reason |
|---|---|
| `configs/` | Keep the eight-step smoke with measured microbatches supplied at execution |
| `local/generated_configs/` | Store configurations generated from the agreed experiment document |
| `experiment.py`, `token_metrics.py`, `token_progress.py` | Keep the main entry and its logging and checkpoint policies together |
| `dynamic_seq/`, `dynamic_seq_bs/`, `schedulers/` | Keep shared implementation because batch scheduling extends sequence scheduling |
| `data/`, `optimizer.py`, `training_config.py` | Keep shared data and optimization behavior |
| `scripts/train.py`, `scripts/gpu_smoke.py`, `scripts/probe_microbatch.py` | Keep training, GPU memory measurement, and complete smoke acceptance commands |
| `scripts/download_dataset.py`, `scripts/rebuild_long_token_cache.py` | Keep pinned downloads and checked token reconstruction |
| Other `scripts/` files | Retain provenance tools, historical generators, and helpers imported by tests |
| `tests/` | Retain boundary, order, validation, token, gradient, schedule, mask, and recovery checks |
| `docs/getting_started.md`, `docs/final_dataset.md` | Keep the current operation guide and dataset description |
| `docs/reports/` | Keep current cleanup, audit, and implementation evidence |
| `legacy/configs/` | Preserve all 76 historical YAML files and their hashes |
| `legacy/reports/` | Preserve historical result files with unchanged hashes |
| `local/private_archive/` | Preserve private conversation provenance locally and exclude it from delivery |
| `cluster/vt_arc/` | Keep ARC tools separate from the portable launcher |

Do not merge `dynamic_seq_bs` into `dynamic_seq` during this delivery.
The inheritance expresses their dependency directly.
Combining them would affect historical trainers without simplifying the current execution path.

## Historical experiment roles

| Module | Role |
|---|---|
| `loss_grad_seq` | Measures loss on parent sequences and trains on shorter gradient crops |
| `seq_grad` | Computes independent gradients to measure context effects |
| `grad_var` | Computes gradient variance and X diagnostics |
| `config_registry.py` | Registers historical models and experiment builders |

The main entry does not select these historical trainers.
Their tests check shared behavior and preserve earlier experiments.
This phase retained all training capabilities.

The main entry and smoke contain no 22,256-token validation setting.
Historical packed builders, diagnostics, and reports retain that older value for their original experiments.
Some archived filenames retain `val22256` while their stored settings use the updated 16,384-token artifact.
Do not infer validation length from those filenames.

## Retained manifests

The current dataset uses manifests under `assets/data/fineweb_edu_long_30b_shuffled_v1/`.
The files under `torchtitan_dynamic_seq_len/data/manifests/` serve earlier datasets and tests.

| Files | Dependency |
|---|---|
| `fineweb_100bt_manifest.json` | Historical FineWeb source loader and preparation checks |
| `fineweb_100bt_validation_reference.json` | Historical fixed validation identity |
| `fineweb_100bt_validation_llama2_*_reference.json` | Exact identities for historical lengths 256, 512, 1024, 2048, 4096, 8192, and 12288 |
| `tokenizer_source.json` | Historical Llama2 tokenizer preparation |

Keep these files while the corresponding loaders and regression tests remain.
Do not substitute their identities for the current Llama3 tokenizer or validation arrays.

## Remaining portability limits

The smoke configuration contains the agreed eight-step schedule and global batch size 512.
The target GPU probe supplies measured microbatches.
The target server must provide the launch example and persistent storage paths.
Local `data/` and `tokens/` links are not portable assets.
The collaborator must restore standalone directories.
The new environment passed installation and import checks.

Complete data reconstruction and independent validation-window checks passed.
The real-data smoke, exact checkpoint recovery, and online W&B checks passed on two H200 GPUs.
See the reports for current evidence and limitations.
