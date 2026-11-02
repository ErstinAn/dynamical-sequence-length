# Collaboration delivery inventory

Use [instruction.md](../instruction.md) as the first-run procedure.
The collaborator must receive a fixed Git commit, not an uncommitted working directory.

## Files included in Git

| Files | Purpose |
| --- | --- |
| `torchtitan_dynamic_seq_len/` | Training implementation and configuration registry |
| `configs/` | Shared experiment protocols and portable examples |
| `scripts/` and `run.sh` | Launch, data preparation, and checks |
| `tests/`, including `tests/fixtures/text.jsonl` | Correctness tests and a small local training dataset |
| `torchtitan_dynamic_seq_len/data/manifests/*.json` | Dataset identity, tokenizer source, and validation reference hashes |
| `instruction.md`, `README.md`, and `docs/` | Setup, methods, and measured test results |
| `pyproject.toml` and `requirements-lock.txt` | Package metadata and fixed dependency versions |
| `LICENSE` and `NOTICE.md` | Source attribution and asset notices |
| `cluster/vt_arc/` | Optional ARC integration, also imported by launcher tests |
| `.gitignore` and `.pre-commit-config.yaml` | Generated-file exclusions and code checks |

The module relocation must include both added directories and deleted old modules.
Imports, configuration entries, and tests must change together.
The sequence and batch schedules, gradient-distance implementation, and proposal configurations have corresponding tests and documentation.
These files are included as complete features, not isolated registry references.

Shared configurations use allocation placeholders and disable W&B by default.
ARC resource blocks remain examples. Set site-specific resources in an ignored local copy.
Existing local run snapshots and logs retain their original settings.

## Assets kept outside Git

| Asset | How to obtain it |
| --- | --- |
| Llama2 tokenizer files | Run `scripts/prepare_tokenizer.py` with the pinned source record |
| FineWeb training shards | Stream from the pinned revision, or download all 149 listed files |
| Validation source parquet | Download the fixed `sample/100BT/000_00000.parquet` |
| Validation arrays and exclusion index | Build with the preparation script, or transfer the complete verified artifact |
| Checkpoints and training logs | Keep in experiment storage, separate from source control |
| Nightly wheels, when package servers no longer serve them | Transfer a verified wheel collection separately |

The 149 training files total 300,405,932,666 bytes, approximately 279.77 GiB.
The held-out source file is approximately 2 GiB. Each validation artifact is approximately 484 MiB.
These figures exclude caches, environment files, and checkpoints.
Tokenizers and validation arrays are not included in a Git clone.
No pretrained model weights are necessary for the examples.

Do not commit credentials, virtual environments, caches, `outputs/`, `local/`, `archive/`, or W&B run directories.
Use the recorded source revisions and hashes rather than a floating dataset or tokenizer revision.

## Acceptance scope

The training dependency environment already exists on the source server.
Testing against this environment does not prove that every locked wheel remains downloadable.
The delivery checks must record fresh-checkout tests separately from installation against existing dependencies.
GPU smoke results apply only to the recorded hardware, precision, batch size, and sequence length.
The complete FineWeb training budget is not part of the handoff smoke.

The final delivery record below is updated after verification.
