# Main experiment implementation record

Record date: 2026-11-18.
The repository now has a separate main entry for the agreed 1B experiment.
The full delivery from an independent environment remains incomplete.
See the [structured record](20261118_implementation.json) for source hashes and raw logs.

## Implemented changes

- Added `experiment.py:llama3_1b` with the native Llama3 vocabulary and upstream 1B model definition.
- Used the current data package's finite ordered stream and fixed 936-window validation identity.
- Preserved global average loss, global maximum loss, and validation loss names and calculations.
- Bound all three W&B metrics to cumulative nominal training tokens.
- Included the current token count in validation records between training logs and after recovery.
- Added ten checkpoint milestones at 10% increments of the requested token budget.
- Retained complete synchronous checkpoint payloads and all ten saved states.
- Added configuration checks, a concrete dry-run plan, and a portable recovery command.
- Required online W&B with an explicit entity and project for the main launcher.
- Added W&B to the package dependencies at the version already recorded in the lock file.
- Made data restoration import the repository's cache builder instead of a script from the dataset directory.
- Archived ten historical report files and one conversation file without changing their contents.

The two cache builder copies had identical SHA256 values before the import change.
This change removes a code dependency on the downloaded dataset package.
It does not change tokenization or document order.

The conversation archive contains credential-like text and has no runtime dependency.
Its unchanged bytes now remain under ignored `local/private_archive/` with restricted permissions.
The delivery bundle excludes this private archive.
Earlier test bundles remain local and must not serve as the final delivery bundle.

The new trainer inherits the existing forward, backward, loss, mask, and gradient normalization implementation.
Historical trainers and schedules retain their existing code.
The main configuration has no runnable default schedule.
Its zero steps and token budget require explicit experiment settings.
The user later specified the eight-step smoke in `configs/smoke.yaml`.
It uses lengths 2048, 4096, 8192, and 16384 for two steps each.
Global batch size remains 512, for 31,457,280 nominal tokens.
The GPU probe supplies measured microbatches.

The explicit eight-step smoke saves at steps 2, 4, 6, and 8.
The formal ten-threshold policy remains unchanged.
The acceptance driver compares uninterrupted training with a run interrupted after step 4.
It compares all final checkpoint tensors and state values.

## Checkpoint and recovery behavior

The planner uses integer arithmetic for `ceil(budget * milestone / 10)`.
It saves at the first complete step reaching each threshold.
It reports excess tokens from complete steps.
It rejects schedules that map multiple thresholds to one step.
The final configured step must be the first step reaching the budget.

The save policy changes for the new main entry only.
The payload still contains model, optimizer, LR scheduler, trainer counters, and data reader state.
The main trainer adds budget and global nominal count fields to its state.
It rejects recovery with changed token budgets or inconsistent counters.
The existing schedule signature also checks DP degree, sequence schedules, and batch schedules.

Each recovery uses a new W&B run segment in the original experiment group.
This preserves prior history when a checkpoint precedes the most recent logged step.
Token counts resume from the checkpoint value.
The launcher retains the checkpoint directory and records each segment ID.
Online history and charts still require the agreed GPU smoke check.

## Executed checks

| Check | Result |
|---|---|
| Full CPU suite | 203 tests, 200 passed, 3 skipped, 52.867 seconds |
| Main CPU checks | 11 tests passed, including actual checkpoint writes and recovery |
| Two-rank CPU check | Passed nominal counts, ten complete saves, exact recovery, and validation token axes |
| New test directory | Imports, pip check, launcher listing, data command help, and 11 main tests passed |
| Archived files | All 11 hashes matched the files before archival |
| Historical configurations | All 76 YAML hashes remain unchanged |
| Later eight-step and comparison checks | 14 tests passed, including changed tensor and counter detection |
| Independent new runtime and updated source | 211 tests, 206 passed, 5 skipped, 52.418 seconds |

The new test directory was `/tmp/sequence-handoff-1bg7tt4_`.
It contained a copy of the current uncommitted code, not a Git clone.
Its virtual environment used `--system-site-packages` and reused the existing runtime dependencies.
The copied repository contained no `assets/` or `outputs/` directory.
The tests created small synthetic data inside temporary directories.
These results do not establish a successful installation or download from zero.

The CPU checkpoint tests used the production optimizer and LR scheduler containers with a scalar model.
They wrote ten distributed checkpoint directories, including metadata and token milestone records.
They loaded model, optimizer, LR scheduler, trainer counters, and data reader state.
Resumed parameter updates matched uninterrupted updates exactly.
The separate Gloo check repeated these checks with two CPU ranks.

The first test iteration exposed an incomplete optimizer fixture and a comparison of function memory addresses.
The final tests use production state containers and ignore only function addresses when comparing model configurations.
The final model comparison preserves the complete serialized model configuration otherwise.

The full suite skipped two opt-in historical distributed checks and one historical local artifact check.
The new two-rank token test ran separately with local socket access.
No GPU allocation or training submission occurred during these initial CPU checks.
The later CPU data job is `8795433`.
See the independent environment and data records for current cluster activity.
The [initial audit](20261118_token_metrics_audit.md) remains a historical record of the gaps before implementation.

## Remaining acceptance work

The user supplied the Slurm reference, persistent storage, smoke schedule, and debug W&B destination.
The new isolated runtime passed installation, imports, and CPU checks.
The standalone Miniforge installation and new Python creation check also passed.
The experiment document still defines the later formal settings.
Measure microbatches on the target GPU.
The independent fixed dataset download and complete token reconstruction passed.
All reconstructed validation windows also passed the independent alignment check.
Run the real 1B smoke, complete fixed validation, online W&B checks, and GPU checkpoint recovery.
Measure runtime and storage before agreeing on the formal submission.
Record the delivery commit after the final source review.

The independent check found a tokenizer path fixed to the source directory.
The preparation tool now honors `DYNAMIC_SEQ_LEN_ASSETS`.
It reads the same checked provenance from the downloaded package when the old local provenance file is absent.
The updated standalone source passed all available CPU checks without directory links.

The [run guide](../codebase_kick_off/getting_started.md) contains the implemented launch and recovery commands.
The [collaborator prompt](../codebase_kick_off/collaborator_prompt.md) identifies the next required actions.
