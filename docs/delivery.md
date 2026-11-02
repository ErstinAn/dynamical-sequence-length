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

## Verification on 2026-11-02

The [machine-readable record](delivery_results.json) identifies the tested source commit and measurements.
The source-tree CPU suite ran 113 tests: 110 passed and three optional tests skipped.
The artifact check and both optional distributed checks then passed separately.
Two-rank dynamic sequence, dynamic batch, and sequence-gradient checks also passed.
Ruff, formatting, text checks, and all pre-commit hooks passed.

A clean clone passed the same 113-test suite with assets outside the checkout.
A separate virtual environment installed the built wheel using the existing runtime dependencies.
An isolated import outside the checkout loaded the installed registry and bundled data manifests.
A complete installation from external package servers remains untested.

Falcon job `742362` completed with exit code `0:0` on one NVIDIA L40S.
The requested walltime was 20 minutes. Actual queue waiting time was 63 seconds.
Execution elapsed time was 4 minutes and 54 seconds.

It ran the tiny smoke, ten 1B updates, and recovery from step five through step ten.
The original 1B run peaked at 21.81 GiB. Recovery peaked at 24.68 GiB.

The restored run matched the original logged losses and gradient norms at steps six through ten.
This comparison uses log precision and does not establish tensor-by-tensor checkpoint equality.

The GPU test used FP32 model storage, BF16 forward parameters, FP32 gradient reductions, length 128, and microbatch one.
Both original checkpoints and the recovered step-ten checkpoint contain complete DCP metadata.
The checkpoint directories occupied approximately 48 GiB across the original and recovery runs.
The training implementation and tested configurations match the committed delivery source.

The 1B FineWeb example was parsed but not run against the full validation token budget.
Multi-GPU 1B execution, long sequences, and full training convergence remain untested.
TorchTitan uses A100 peak FLOPs for L40S MFU reporting in this dependency version.
Do not use that fallback MFU value for L40S performance comparisons.
