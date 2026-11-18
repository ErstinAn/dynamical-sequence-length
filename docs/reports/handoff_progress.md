# 1B delivery progress and validation record

Record date: 2026-11-18.
This record describes the initial cleanup phase before the user supplied the eight-step smoke settings.
Its pending items describe that earlier phase.
Use the [final acceptance record](handoff_acceptance_20261118.md) for the completed smoke and current limits.

See [fresh_environment_20261118.md](fresh_environment_20261118.md) and [fresh_data_20261118.md](fresh_data_20261118.md) for later verification.
Configuration archival and the run guide draft are complete.
The full delivery for execution from a new environment remains incomplete.
The current documentation uses English and follows ASD-STE100 writing principles.

The agreed model is dense 1B with the native Llama3 vocabulary, initialized randomly.
The budget sums global batch size times sequence length at each optimizer step.
The exact budget awaits the experiment document.
That document will specify learning rate, global batch, and sequence length schedules.
The microbatch awaits a smoke test on the target GPU.
The experiment requires W&B.

Keep the existing global average loss, global maximum loss, and validation loss metrics.
Use cumulative nominal training tokens as their W&B horizontal axis.
Train loss uses the actual training sequence length.
The [initial audit](20261118_token_metrics_audit.md) records the original gaps.
The [implementation record](20261118_implementation.md) records the new entry, token axes, checkpoints, and checks.

The full run must save a complete checkpoint at each 10% increment of the cumulative nominal token budget.
Keep all ten checkpoints, including the final state.
Do not count the initialization state.
The main entry now implements this checkpoint schedule.

## Initial cleanup changes

- Saved 310 files, file hashes, and the existing Git diff before cleanup.
- Moved 76 historical YAML files into `legacy/configs/` without changing their contents.
- Checked all 76 YAML files against their original SHA256 values.
- Updated configuration references in tests, scripts, and documentation.
- Redirected old generators to `local/generated_configs/` to preserve historical YAML files.
- Retained one smoke configuration in `configs/`, explicitly marked as draft.
- Made the launcher reject draft configurations before constructing a training command.
- Linked the root README to the main run guide.
- Removed the text check that rejected non-ASCII characters.

During the initial cleanup, the model, training loop, optimizer, schedules, and data reader code matched the baseline.
That phase changed only `run_config.py` within the training package, to reject draft configurations.
The cleanup did not change data files or submit GPU training.
The later English documentation update changed prose and YAML comments only.

## Executed checks

| Check | Result |
|---|---|
| CPU unittest | 192 tests, 189 passed, 3 skipped, 51.566 seconds |
| Historical YAML checksums | All 76 matched |
| Python syntax | Passed |
| Draft launch and dry-run | Both exited before launch with code 2 and created no run directory |
| Text checks during cleanup | Passed |
| Git diff whitespace check | Passed |
| Ruff | Not run because the environment and PATH contained no Ruff executable |

The skipped tests were two optional Gloo distributed checks and the historical FineWeb artifact check.
See the [structured record](handoff_progress.json) for commands, environment versions, skip reasons, and hashes.
See the [raw CPU output](20261118_cleanup_cpu.txt) for the complete test log.
The CPU results above precede the English documentation update.

The English documents passed the STE lint checks and the project text checks.
The update preserved command arguments, configuration values, and training source code.
We did not repeat training tests for this documentation update.

Tests reused `/home/kangan/.conda/envs/torchtitan-macro` and local tokenizer files.
TorchTitan used an editable installation outside this repository, at the pinned commit.
The temporary checkout test reused those dependencies.
These results do not establish a successful independent environment installation.

## Pending work at the end of the initial phase

Independent dependency installation, pinned data downloads, and complete token cache reconstruction remain untested.
The new 1B GPU smoke, W&B upload checks, and full training remain unexecuted.
Ten checkpoint saves and recovery passed CPU checks, including two ranks.
The full 1B GPU checkpoint check remains pending.
Target server storage, resource requests, and submission commands remain unspecified.

Obtain the experiment document, target server launch example, and W&B entity and project.
Check the environment and data procedure in a new test directory.
The final delivery commit and complete execution of the run guide remain unfinished.
The [collaborator prompt](../collaborator_prompt.md) identifies pending inputs and acceptance checks.
