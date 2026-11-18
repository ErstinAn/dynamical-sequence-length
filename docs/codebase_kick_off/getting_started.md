# Run the smoke test on the target server

The collaborator fills in two paths in the [meeting prompt](collaborator_prompt.md). The agent performs the steps below.
The default for smoke and each formal run is **2 Slurm nodes with 8 NVIDIA H200 GPUs each (16 GPUs total)**.
Use another layout only when the user requests it.

| Input | Use |
|---|---|
| `JOB_EXAMPLE` | Absolute path to a working GPU job file on this server. Read its resource requests and submission method. |
| `RUN_STORAGE` | Absolute path to writable persistent storage shared by the login and compute nodes. |

Keep the existing code, tests, and `configs/smoke.yaml` read-only.
Create a new site job file and any new launch settings.
If the existing implementation fails, report the error and its log.

## Check the launch support and the Slurm cluster

Check the target revision before reserving GPUs. This revision has two limits.

- The [training launcher](../../scripts/train.py) treats `--nproc` as workers per host and uses a local rendezvous. The [smoke driver](../../scripts/gpu_smoke.py) uses that launcher for training and recovery.
- The [fixed validator](../../torchtitan_dynamic_seq_len/data/fixed_validation.py) requires equal window counts on all data-parallel ranks. Dividing 936 windows among 16 ranks leaves eight windows.

These limits block the default smoke on this revision. Report them before GPU submission.
Resume the GPU procedure when the checkout supports both nodes and all 936 windows with 16 ranks.
Get permission before changing shared code or using another allocation.

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

## Submit and verify the allocation

Use this procedure after the support checks above pass.
Adapt `JOB_EXAMPLE` in a new file under `cluster/<site>/`. Set `<site>` to the target site's name.
Request two H200 nodes with eight GPUs per node. Use the account, QoS, walltime, and submission command verified on this cluster.

Use the same shared checkout, environment, assets, and attempt directory on both nodes.
Choose a new attempt directory under `RUN_STORAGE` for each smoke submission.

Use the two-node launch command supported by the checked revision.
Launch one distributed training run with one worker per GPU.
Check the eight-step plan for 15,728,640 nominal tokens. You can run this configuration check on the login node without GPUs.
Measure microbatches on the target hardware. Use the same layout for uninterrupted training and recovery.

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
Require eight distinct H200 device UUIDs on each of two hosts and global ranks 0 through 15 in one group.
Check that every worker reports `WORLD_SIZE=16`. Slurm can bind each worker to one visible GPU.

Check GPU visibility inside the Slurm allocation on the compute nodes. Visibility on the login node is irrelevant.
Check communication in the training group. An NCCL sum of one from each worker must return 16 on every worker.

### Single-node override

Use this command only if the user explicitly requests a single node.
The GPU count on that node must divide 936 validation windows and the global batch of 256.
Run from the checkout inside that allocation with the environment and asset paths set above.
Set `ATTEMPT` to a new directory under `RUN_STORAGE`.

```bash
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
python scripts/gpu_smoke.py --probe --nproc "$GPU_COUNT" \
  --assets "$DYNAMIC_SEQ_LEN_ASSETS" --output "$ATTEMPT" \
  --profile "$ATTEMPT/microbatches.json"
```

Record the approved layout in the smoke results.

## Accept the result

| Setting | Smoke value |
|---|---|
| Allocation | Two nodes, eight H200 GPUs per node, 16 training workers by default |
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
