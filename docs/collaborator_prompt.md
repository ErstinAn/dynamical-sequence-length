# Collaborator startup prompt

Replace bracketed values before giving this prompt to Codex or Claude Code.
Keep unavailable values marked as pending.
Give the collaborator this prompt and the repository URL.

```text
Prepare this repository for the agreed Llama3 1B sequence length experiment.
Repository: https://github.com/ErstinAn/dynamical-sequence-length.git
Branch: main
Experiment document: [DOCUMENT_PATH_AND_VERSION]
Target server launch example: [LAUNCH_EXAMPLE_PATH]
W&B smoke entity and project: rotation_optimizer / debug

Read docs/getting_started.md, docs/environment_bootstrap.md, and docs/repository.md.
Read the latest verification reports before claiming that a procedure passed.
Clone the default main branch.
Record the resolved Git commit before creating the environment.
Use the source acquisition commands in docs/getting_started.md.
Use English for documents. Follow ASD-STE100 writing principles.

Check the machine and supplied launch example.
Record GPUs, driver, memory, storage quotas, and permitted paths.
Identify persistent checkpoint storage.
For a new Slurm site, create cluster/<site>/run.sbatch from its working launch example.
If the site requires resource YAML, create cluster/<site>/submit.yaml and use its submission tool.
Use the same site launcher for smoke and formal runs.
Use cluster/vt_arc/smoke_handoff.sbatch on VT ARC.

Install Python if the server lacks it.
Create an isolated environment with requirements-portable.txt.
Record reused environments, package caches, and dataset caches.
Do not describe reused dependencies as an installation from zero.
Check imports, pip check, and GPU availability inside the allocation.

Obtain read access to the private dataset.
Use the tokenizer in assets/tokenizer/ from this checkout.
Run scripts/download_dataset.py to install it and check its hashes against the pinned dataset.
Download dataset revision b6ea2fde77bab36dff963aba5e6f482477959933.
Check all reference checksums.
Restore standalone data and token directories.
Preserve orders/train.npy and the fixed 936 validation windows.
Use the repository's token cache builder.

Use the main llama3_1b entry with random initialization.
Keep the native model definition, masks, loss, and gradient normalization.
Write each formal YAML from the agreed experiment document.
Keep configs/smoke.yaml as the only configuration under configs/.
Store generated configurations under local/generated_configs/.
Do not invent missing learning rates or schedules.

Count nominal training tokens as the sum of global batch times actual sequence length.
Do not multiply by GPU count again.
Keep global_avg_loss, global_max_loss, and validation loss.
Use n_tokens_seen as the horizontal axis for all three metrics.
Validation must not advance that counter.
Do not add long/short loss series without an explicit request.

Run the required CPU and distributed checks.
Use the agreed real-data smoke: lengths 2048, 4096, 8192, 16384, each for two steps.
Keep global batch size 512 at every step.
Measure microbatches for this GPU and DP degree.
Use scripts/gpu_smoke.py --probe to execute the full acceptance procedure.
Measure the microbatch, step time, validation time, and checkpoint size.
Check complete fixed validation, W&B upload, checkpoint saving, and recovery.
Check smoke checkpoints at steps 2, 4, 6, 8 and recovery across step 4 to step 5.
Require 31,457,280 nominal training tokens and 15,335,424 validation targets.
Keep native full validation after steps 1 and 8.
Run docs/reports/check_smoke_wandb.py after the GPU driver completes.
For formal training, check ten token thresholds and retain all ten complete checkpoints.
Compare resumed updates and counts with an uninterrupted reference.

Leave formal schedules and learning rate sweeps pending until the user supplies the experiment document.
Then inspect the formal configuration with scripts/train.py --dry-run.
Estimate runtime and storage from smoke measurements.
Present the concrete resource request and experiment scope to the user.
Start formal training after the user accepts that request and scope.
Use the supplied launch example for submission and log capture.

Record commands, versions, results, skipped checks, and failure reasons.
Provide the run directory, W&B run IDs, checkpoint paths, and recovery command.
Do not report delivery complete while required checks remain unexecuted.
```
