# Llama3 1B sequence length experiments

**Built with Meta Llama 3.** The [tokenizer](assets/tokenizer/) ships with its [Meta license](assets/tokenizer/LICENSE).

This repository compares sequence length schedules on a custom FineWeb-Edu dataset of long documents.
The dense 1B model uses the native Llama3 vocabulary and random initialization.
The planned budget is approximately 20 billion nominal tokens.
The experiment document will specify the exact budget and schedules.
Smoke and formal training require online W&B access. The public dataset and bundled tokenizer need no Hugging Face login.

For a meeting, open the [smoke startup prompt](docs/codebase_kick_off/collaborator_prompt.md).
Replace the two quoted paths. Paste the block into Codex or Claude Code on the target server.
The delivered code is on the default `main` branch. Use your own W&B credentials. [Local key setup](docs/codebase_kick_off/getting_started.md#verify-wb-access)

| Document | Reader and purpose |
|---|---|
| [Run guide](docs/codebase_kick_off/getting_started.md) | The agent follows these steps from environment creation through smoke acceptance. |
| [Agent startup prompt](docs/codebase_kick_off/collaborator_prompt.md) | Share this at the meeting. The collaborator fills in two fields. |
| [Local verification record](docs/reports/handoff_acceptance_20261118.md) | Optional evidence for the owner and reviewers. Its recorded paths describe the original test server. |

The run guide links to the environment instructions and required check scripts.
Keep those supporting files in the cloned repository.

The new environment and complete data reconstruction passed their checks.
The previous real-data smoke used a global batch of 512 and passed on one node with two H200 GPUs.
Recovered training matched every final checkpoint value from uninterrupted training.
The uploaded W&B losses use cumulative nominal tokens as their horizontal axes.

Smoke and each formal run use one node with eight H200 GPUs by default. Each worker validates 117 of the 936 fixed windows.
See the [target-server checks](docs/codebase_kick_off/getting_started.md#check-the-launch-support-and-the-slurm-cluster).

`configs/smoke.yaml` defines the eight-step smoke with global batch size 256 and 15,728,640 nominal tokens.
Measure microbatches on the allocated GPU before execution.

Use the [experiment launch prompt](docs/sl_schedule_docs/experiment_launch_prompt.md) with the agreed formal specification.

Training counts `global_batch_size * sequence_length` nominal tokens at each optimizer step.
Keep the existing `global_avg_loss`, `global_max_loss`, and validation loss metrics.
Use cumulative nominal training tokens as their W&B horizontal axis.
Train loss uses the actual training sequence length at that step.
The main logger binds all three losses to `n_tokens_seen`.
See the [acceptance record](docs/reports/handoff_acceptance_20261118.md) for executed checks and limits.

The full run must save one complete checkpoint at each 10% increment of the cumulative token budget.
Keep all 10 checkpoints, including the final state.
Save after the first complete optimizer step that reaches each threshold.
The main trainer implements this schedule and checks actual counts against the planned token totals.

The current dataset package is `assets/data/fineweb_edu_long_30b_shuffled_v1`.
Training reads documents in the order stored in `orders/train.npy`.
Each validation uses the same 936 windows, with one source document per window.
Each window has 16,384 input positions.
The complete validation contains 15,335,424 valid prediction targets.
See the [dataset description](docs/final_dataset.md) for identities and evaluation limits.

| Path | Purpose |
|---|---|
| `configs/smoke.yaml` | Eight-step smoke, lengths 2048 through 16384, global batch size 256 |
| `torchtitan_dynamic_seq_len/` | Trainers, model registration, data readers, and schedules |
| `torchtitan_dynamic_seq_len/experiment.py` | Main native-vocabulary 1B entry, named `llama3_1b` |
| `scripts/train.py` | Training launcher for one node |
| `scripts/download_dataset.py`, `scripts/rebuild_long_token_cache.py` | Pinned downloads and checked token reconstruction |
| `scripts/gpu_smoke.py` | GPU microbatch measurement, complete smoke, and recovery comparison |
| `tests/` | Data, gradient, dynamic schedule, and recovery checks |
| `cluster/<site>/run.sbatch` | Create this file from the target server’s working Slurm example |
| `assets/tokenizer/` | Original tokenizer, provenance, license, and attribution |
| `assets/data/`, `assets/hf/`, `outputs/` | Local data, installed assets, and run outputs |

The [environment record](docs/reports/fresh_environment_20261118.md) and [data record](docs/reports/fresh_data_20261118.md) identify verified steps and reuse.
The [repository map](docs/repository.md) explains retained dependencies and entry points.
The [collaborator prompt](docs/codebase_kick_off/collaborator_prompt.md) identifies setup steps and required inputs.
See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for licensing and source attribution.
