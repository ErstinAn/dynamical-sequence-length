# Repository map

Use [getting_started.md](codebase_kick_off/getting_started.md) for the current experiment.
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

## Files for the experiment

| Path | Use |
|---|---|
| `configs/smoke.yaml` | Run the agreed eight steps |
| `local/generated_configs/` | Write formal YAML files from the experiment document |
| `scripts/train.py` | Start or resume training |
| `scripts/gpu_smoke.py`, `scripts/probe_microbatch.py` | Measure microbatches and run the GPU checks |
| `scripts/download_dataset.py`, `scripts/rebuild_long_token_cache.py` | Prepare the pinned data |
| `tests/` | Check data boundaries, validation, gradients, schedules, token counts, and recovery |
| `cluster/vt_arc/smoke_handoff.sbatch` | Submit the smoke on VT ARC |
| `cluster/<site>/run.sbatch` | Submit from another server after adapting its working example |

The previous revision contains the [historical YAML files and results](https://github.com/ErstinAn/dynamical-sequence-length/tree/427d321c40e8b82444d2f5c07082e6155c0f914e/legacy).
Use that revision to reproduce historical workflows or resume jobs created with its source hash.

## Historical experiment roles

| Module | Role |
|---|---|
| `loss_grad_seq` | Measures loss on parent sequences and trains on shorter gradient crops |
| `seq_grad` | Computes independent gradients to measure context effects |
| `grad_var` | Computes gradient variance and X diagnostics |
| `config_registry.py` | Registers historical models and experiment builders |

The main entry does not select these historical trainers.

## Retained manifests

The current dataset uses manifests under `assets/data/fineweb_edu_long_30b_shuffled_v1/`.
The files under `torchtitan_dynamic_seq_len/data/manifests/` serve earlier dataset loaders.

| Files | Dependency |
|---|---|
| `fineweb_100bt_manifest.json` | Historical FineWeb source loader and preparation checks |
| `fineweb_100bt_validation_reference.json` | Historical fixed validation identity |
| `fineweb_100bt_validation_llama2_*_reference.json` | Exact identities for historical lengths 256, 512, 1024, 2048, 4096, 8192, and 12288 |
| `tokenizer_source.json` | Historical Llama2 tokenizer preparation |

Keep these files while the corresponding loaders remain.
Do not substitute their identities for the current Llama3 tokenizer or validation arrays.

## Remaining portability limits

The smoke configuration contains the agreed eight-step schedule and global batch size 256.
The target GPU probe supplies measured microbatches.
The target server must provide the launch example and persistent storage paths.
Local `data/` and `tokens/` links are not portable assets.
The collaborator must restore standalone directories.
The new environment passed installation and import checks.

Complete data reconstruction and independent validation-window checks passed.
The real-data smoke, exact checkpoint recovery, and online W&B checks passed on two H200 GPUs.
See the reports for current evidence and limitations.
