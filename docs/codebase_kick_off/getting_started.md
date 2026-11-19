# Run the smoke test on the target server

The collaborator fills in two paths in the [meeting prompt](collaborator_prompt.md). The agent performs the steps below.
The default for smoke and each formal run is **one Slurm node with eight NVIDIA H200 GPUs**.
Use another layout only when the user requests it.

| Input | Use |
|---|---|
| `JOB_EXAMPLE` | Absolute path to a working GPU job file on this server. Read its resource requests and submission method. |
| `RUN_STORAGE` | Absolute path to writable persistent storage shared by the login and compute nodes. |

Keep the existing code, tests, and `configs/smoke.yaml` read-only.
Create a new site job file and any new launch settings.
If the existing implementation fails, report the error and its log.

## Check the launch support and the Slurm cluster

The [training launcher](../../scripts/train.py) starts eight workers on the host when you pass `--nproc 8`.
Each worker evaluates 117 of the 936 fixed windows. Together they count 15,335,424 prediction targets.

Use the login node for Slurm discovery. Read the target site's current submission instructions and the supplied job example.
For a standard Slurm installation, these read-only commands show partitions and account associations.

```bash
sinfo -o '%P %a %l %D %G'
sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS,DefaultQOS,MaxWall
```

Check the permitted account, partition, quality of service (QoS), H200 request syntax, and walltime.
Check limits on running jobs, submitted jobs, and GPUs.
Check whether the site requires `sbatch` or a site submission tool.

If accounting queries are restricted, use the site's portal or documented user commands.
Ask the collaborator only when a required choice or permission remains unresolved. [Slurm accounting commands](https://slurm.schedmd.com/sacctmgr.html)

## Prepare the checkout and environment

Use a clean checkout of `main`. If needed, run these commands from a shared parent directory:

```bash
git clone https://github.com/ErstinAn/dynamical-sequence-length.git
cd dynamical-sequence-length
git rev-parse HEAD
```

Run the remaining commands from the repository root.
Export the two paths from the meeting prompt in your shell.
Check network access and the shared storage quota.

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
Run the CUDA check inside the GPU allocation. The earlier smoke record covers one node with two H200 GPUs.

```bash
export PATH="$RUN_STORAGE/environment/conda/bin:$PATH"
export DYNAMIC_SEQ_LEN_ASSETS="$RUN_STORAGE/assets"
export DATASET_DIR="$DYNAMIC_SEQ_LEN_ASSETS/data/fineweb_edu_long_30b_shuffled_v1"
export TOKENIZER_DIR="$DYNAMIC_SEQ_LEN_ASSETS/hf/llama3-tokenizer"
```

## Verify W&B access

Smoke requires online write access to `dynamic_seq_length/debug`. Formal runs use the project in the experiment specification.
Use the site's secret store to supply `WANDB_API_KEY`, or use an existing W&B login in your private `.netrc`.

To use a local key file, run the following commands from the repository root.
The copy uses owner-only permissions and preserves an existing file.

```bash
(umask 077 && cp -n docs/API-KEYS.example.md docs/API-KEYS.md)
chmod 600 docs/API-KEYS.md
git check-ignore -- docs/API-KEYS.md
```

Require the last command to print `docs/API-KEYS.md`. Open that local file in your editor.
Enter your own W&B key in the second column of the `WANDB_API_KEY` row. Leave `HF_TOKEN` empty for anonymous downloads.
If your private file is elsewhere, set `API_KEYS_FILE` to its absolute path.

Run this block in Bash from the repository root. Use the same credential setup in the job before starting workers.
Keep keys out of prompts, job files, and logs.

```bash
set +x
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
key_file="${API_KEYS_FILE:-docs/API-KEYS.md}"
if [[ -z "${WANDB_API_KEY:-}" && -r "$key_file" ]]; then
  WANDB_API_KEY=$(awk -F'|' '$2 ~ /^ *WANDB_API_KEY *$/ {gsub(/[[:space:]]/, "", $3); print $3; exit}' "$key_file")
  export WANDB_API_KEY
fi
python scripts/check_wandb_access.py dynamic_seq_length debug
```

Require `W&B login OK` before submission. Run the same check once inside the allocation before training.
Stop the job if the check fails. Use the same credentials when you verify the uploaded smoke history.

| Result | Next action |
|---|---|
| Login fails | Check that this shell or job received your current W&B key. |
| Project access fails | Check the exact entity and project. Ask the owner for write access. |
| Network error or timeout | Check the node's route or approved proxy to `api.wandb.ai`. |

Retry after you correct the reported cause. Keep W&B online.

## Download and rebuild the dataset

Run the download on a node with network access.
Run tokenization in a CPU allocation.

Download the public dataset anonymously. The command pins the dataset commit and verifies its checksums.
The tokenizer ships in `assets/tokenizer/`. Both steps work without a Hugging Face login. [Pinned dataset](https://huggingface.co/datasets/Erstin/fineweb-edu-long-30b/tree/b6ea2fde77bab36dff963aba5e6f482477959933)

```bash
python scripts/download_dataset.py \
  --revision b6ea2fde77bab36dff963aba5e6f482477959933 \
  --dataset-dir "$DATASET_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
  --cache-dir "$RUN_STORAGE/download-cache"
```

Require `text_download_complete: true` in `$DATASET_DIR/download_receipt.json`.
Check that the receipt names `Erstin/fineweb-edu-long-30b` and the commit from the command.
After an interrupted download, repeat the command with the same directories.

For HTTP 401, 403, or 404, check the repository name, commit, and Hub endpoint first.
For HTTP 429, wait for the retry time specified by the Hub.
If authentication is necessary, use a read token for one retry. Run the following block from the same shell.
For a checksum failure, inspect the named file before reuse.

```bash
(
  set +x
  export HF_ENDPOINT=https://huggingface.co
  key_file="${API_KEYS_FILE:-docs/API-KEYS.md}"
  if [[ -z "${HF_TOKEN:-}" && -r "$key_file" ]]; then
    HF_TOKEN=$(awk -F'|' '$2 ~ /^ *HF_TOKEN *$/ {gsub(/[[:space:]]/, "", $3); print $3; exit}' "$key_file")
    export HF_TOKEN
  fi
  python scripts/download_dataset.py --authenticated \
    --revision b6ea2fde77bab36dff963aba5e6f482477959933 \
    --dataset-dir "$DATASET_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
    --cache-dir "$RUN_STORAGE/download-cache"
)
```

If the retry fails, report the repository, revision, and error. Omit credentials from the report.

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

## Submit and verify the allocation

Use this procedure after the support checks above pass.
Adapt `JOB_EXAMPLE` in a new file under `cluster/<site>/`. Set `<site>` to the target site's name.
Request one node with eight H200 GPUs. Use the account, QoS, walltime, and submission command verified on this cluster.

Run one driver with all eight GPUs visible. The driver starts one training worker per GPU.
Use this command on the login node to check the plan. Require eight steps and 15,728,640 nominal tokens.

```bash
python scripts/train.py configs/smoke.yaml --nproc 8 --dry-run
```

### Run the eight-GPU smoke

Put the following block in the job file. Run it once inside the GPU allocation from the repository root.
Load the environment and asset paths above. Set `ATTEMPT` to a new path under `RUN_STORAGE` for each submission.

```bash
set -eu
python -c 'import torch; assert torch.cuda.device_count() == 8 and all("H200" in torch.cuda.get_device_name(i) for i in range(8)), "Expected eight visible H200 GPUs"'
python scripts/gpu_smoke.py --probe --nproc 8 \
  --assets "$DYNAMIC_SEQ_LEN_ASSETS" --output "$ATTEMPT" \
  --profile "$ATTEMPT/microbatches.json"
```

The driver measures microbatches and checks uninterrupted and resumed training on the same allocation.

### Submit the job

For a Slurm Bash job, set `JOB_FILE` to the new job file. Use the site's request check when available.

```bash
bash -n "$JOB_FILE"
sbatch --test-only "$JOB_FILE"
```

Resolve any errors before submission. Use the site's submission command. For standard Slurm, run this command.

```bash
sbatch --output="$RUN_STORAGE/logs/smoke-%j.log" "$JOB_FILE"
```

Record the returned job ID as `JOB_ID`. Monitor it with `squeue -j "$JOB_ID"`.
For a pending job, inspect the pending reason. For a rejected request, correct the reported setting before retrying.
Use `sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed` to check the outcome.
If the site requires a submission tool, use its equivalent commands.

Record each worker's hostname, global rank, local rank, GPU model, device UUID, and world size.
Require one hostname, eight distinct H200 devices, and global ranks 0 through 7 in one group.
Check that every worker reports `WORLD_SIZE=8`. An NCCL sum of one from each worker must return eight on every worker.

## Accept the result

| Setting | Smoke value |
|---|---|
| Allocation | One node, eight H200 GPUs, eight training workers |
| Model | Native dense Llama 3 1B, random initialization |
| Training | Lengths 2048, 4096, 8192, 16384 for two steps each. Global batch size 256. |
| W&B smoke destination | Entity `dynamic_seq_length`, project `debug`, online mode |
| Checkpoints | Full saves at steps 2, 4, 6, 8. Resume from the step 4 checkpoint. |
| Validation | Steps 1 and 8. Each pass uses all 936 fixed windows exactly once and counts 15,335,424 prediction targets. |

After the job finishes, check W&B from the repository root.
Set `ATTEMPT` to the directory used by the job.

```bash
python docs/reports/check_smoke_wandb.py \
  --acceptance "$ATTEMPT/acceptance.json" --output "$ATTEMPT/wandb_verification.json"
```

Require `exact_match: true` in `checkpoint_comparison.json` and `status: passed` in `wandb_verification.json` under `ATTEMPT`.
Check `global_avg_loss`, `global_max_loss`, and validation loss against `n_tokens_seen` in W&B.
Count nominal tokens as global batch size times actual training length at each completed step.

Save the working job file path, submission command, Slurm resource settings, allocation log, and shared storage path for formal runs.
Report the job ID, Git commit, run paths from `acceptance.json`, checkpoint paths, and three W&B links.
Each run stores its checkpoints in `checkpoint/step-<STEP>/`. Record any reused caches or environments.

If data checks fail, inspect the first mismatched file and the pinned revision.
For an out-of-memory error, inspect `microbatch-probe.txt` and the actual GPU allocation.
For a recovery failure, check the recorded source, DP degree, and complete checkpoint metadata.
Keep the failed log. Stop the kickoff after smoke acceptance.

[Earlier verification results](../reports/handoff_acceptance_20261118.md) · [Later formal training](../formal_training.md)
