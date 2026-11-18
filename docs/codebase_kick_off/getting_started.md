# Run the smoke test on the target server

The collaborator fills in two paths in the [meeting prompt](collaborator_prompt.md). The agent performs the steps below.

| Input | Use |
|---|---|
| `JOB_EXAMPLE` | Absolute path to a working GPU job file on this server. Read its resource requests and submission method. |
| `RUN_STORAGE` | Absolute path to writable persistent storage shared by the login and compute nodes. |

Keep the existing code, tests, and `configs/smoke.yaml` read-only.
Create a new site job file and any new launch settings.
If the existing implementation fails, report the error and its log.

## Prepare the checkout and environment

Use a clean checkout of `main`. If needed, run these commands from a shared parent directory:

```bash
git clone https://github.com/ErstinAn/dynamical-sequence-length.git
cd dynamical-sequence-length
git rev-parse HEAD
```

Run the remaining commands from the repository root.
Export the two paths from the meeting prompt in your shell.
Read the job example. Check network access, storage quota, and the GPU request.

```bash
test -r "$JOB_EXAMPLE"
mkdir -p "$RUN_STORAGE/logs"
df -h "$RUN_STORAGE"
uname -sm
```

Allow about 400 GB for the environment, download cache, complete dataset, and both smoke runs.
The measured dataset used about 173 GB. The two sets of checkpoints used about 127 GB.

Use the [environment guide](../environment_bootstrap.md) to install Python and the locked dependencies.
Use `RUN_ROOT="$RUN_STORAGE"` in that guide. Require the import checks to pass.
Run the CUDA check inside the GPU allocation. The verified hardware was a single node with two H200 GPUs.

```bash
export PATH="$RUN_STORAGE/environment/conda/bin:$PATH"
export DYNAMIC_SEQ_LEN_ASSETS="$RUN_STORAGE/assets"
export DATASET_DIR="$DYNAMIC_SEQ_LEN_ASSETS/data/fineweb_edu_long_30b_shuffled_v1"
export TOKENIZER_DIR="$DYNAMIC_SEQ_LEN_ASSETS/hf/llama3-tokenizer"
hf auth whoami
```

Use `hf auth login` if the account needs authentication. The account needs read access to the private dataset.
Use `wandb login` if needed. The GPU job needs online write access to the configured project.
Enter credentials on the server.

## Download and rebuild the dataset

Run the download on a node with network access.
Run tokenization in a CPU allocation.
The tokenizer ships in `assets/tokenizer/`.

```bash
python scripts/download_dataset.py \
  --revision b6ea2fde77bab36dff963aba5e6f482477959933 \
  --dataset-dir "$DATASET_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
  --cache-dir "$RUN_STORAGE/download-cache"
```

Require `text_download_complete: true` in `$DATASET_DIR/download_receipt.json`.
For 32 allocated CPUs, use four workers with eight tokenizer threads each.
For fewer CPUs, reduce the worker and thread counts to fit the allocation.
The measured 32-CPU rebuild took 88 minutes, excluding the download.

```bash
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=8 \
  python scripts/rebuild_long_token_cache.py --workers 4 --resume \
  --dataset-dir "$DATASET_DIR" --tokenizer-dir "$TOKENIZER_DIR"
```

Require `complete: true` in `$DATASET_DIR/rebuild_progress.json`.
After an interruption, repeat the same command. It checks existing shards before reuse.

## Run automatic checks

```bash
python docs/reports/fresh_data_20261118.py \
  --dataset-dir "$DATASET_DIR" --tokenizer-dir "$TOKENIZER_DIR" > "$RUN_STORAGE/data-window-check.json"
OMP_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
FIXED_VALIDATION_GLOO_TEST=1 OMP_NUM_THREADS=1 python -m unittest \
  tests.test_fixed_validation.FixedBudgetValidationTest.test_real_two_rank_reduction_and_partial_batches -v
OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module tests.token_progress_distributed
```

Require `passed: true`, 236 checked indexes, 936 fixed windows, and 15,335,424 valid targets in the data report.
Require zero test failures or errors. Run the two distributed checks on a node that permits local sockets.
They check validation across ranks and recovery from full checkpoints.

## Submit on the target server

Copy `JOB_EXAMPLE` to a **new** file under `cluster/<site>/`. Use the target site's name for `<site>`.
Keep the account, queue, module setup, and submission tool from the working example.

Request one node. Run one driver with access to all allocated GPUs.
Set `REPO` to the absolute checkout path in the job environment.
Set `RUN_STORAGE` there to the path from the prompt.

Put the following commands in the new job file, after the site's setup.
Run them from the repository root inside the allocation.
For a site resource YAML, put the same commands in its command field or a new job script.

```bash
set -eu
cd "$REPO"
export PATH="$RUN_STORAGE/environment/conda/bin:$PATH"
export DYNAMIC_SEQ_LEN_ASSETS="$RUN_STORAGE/assets"
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
test "$GPU_COUNT" -gt 0
ATTEMPT="$RUN_STORAGE/smoke-attempt-1"
python scripts/train.py configs/smoke.yaml --nproc "$GPU_COUNT" --dry-run
python scripts/gpu_smoke.py --probe --nproc "$GPU_COUNT" \
  --assets "$DYNAMIC_SEQ_LEN_ASSETS" --output "$ATTEMPT" \
  --profile "$ATTEMPT/microbatches.json"
```

Use a new attempt directory for each submission. The driver creates it.
The dry run must report eight steps and 31,457,280 nominal tokens.
The driver measures microbatches for the allocated GPUs. It runs one training process per GPU.
If the site requires a command such as `srun`, use it to run the driver once.

Submit **only the new file** using the target site's command.
For a Slurm Bash job, set `JOB_FILE` to the path of that file and run:

```bash
bash -n "$JOB_FILE"
sbatch --output="$RUN_STORAGE/logs/smoke-%j.log" "$JOB_FILE"
```

Record the returned job ID. For Slurm, monitor it with `squeue -j "$JOB_ID"`.
Use `sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed` to check the outcome.
If the site uses a YAML submission tool, use its own submit and status commands.

## Accept the result

| Setting | Smoke value |
|---|---|
| Model | Native dense Llama 3 1B, random initialization |
| Training | Lengths 2048, 4096, 8192, 16384 for two steps each. Global batch size 512. |
| W&B smoke destination | Entity `rotation_optimizer`, project `debug`, online mode |
| Checkpoints | Full saves at steps 2, 4, 6, 8. Resume from the step 4 checkpoint. |
| Validation | Steps 1 and 8. Each pass uses the same 936 single-document windows of length 16,384. |

After the job finishes, check W&B from the repository root.
Set `ATTEMPT` to the directory used by the job.

```bash
python docs/reports/check_smoke_wandb.py \
  --acceptance "$ATTEMPT/acceptance.json" --output "$ATTEMPT/wandb_verification.json"
```

Require `exact_match: true` in `checkpoint_comparison.json` and `status: passed` in `wandb_verification.json` under `ATTEMPT`.
Check `global_avg_loss`, `global_max_loss`, and validation loss against `n_tokens_seen` in W&B.
Count nominal tokens as global batch size times actual training length at each completed step.

Report the job ID, Git commit, run paths from `acceptance.json`, checkpoint paths, and three W&B links.
Each run stores its checkpoints in `checkpoint/step-<STEP>/`. Record any reused caches or environments.

If data checks fail, inspect the first mismatched file and the pinned revision.
For an out-of-memory error, inspect `microbatch-probe.txt` and the actual GPU allocation.
For a recovery failure, check the recorded source, DP degree, and complete checkpoint metadata.
Keep the failed log. Stop the kickoff after smoke acceptance.

[Earlier verification results](../reports/handoff_acceptance_20261118.md) · [Later formal training](../formal_training.md)
