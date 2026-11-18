# Run guide: Llama3 1B sequence length experiments

Use this guide from the delivered repository root.
For a meeting, give the collaborator the [startup prompt](collaborator_prompt.md).
The agent follows this guide after it receives the job example and the storage path.
Use English for new run records.
Keep one action in each instruction.

The [acceptance record](../reports/handoff_acceptance_20261118.md) contains the actual verification results.
A new environment, complete data reconstruction, and the two-H200 smoke passed.
Checkpoint recovery produced exactly the same final state as uninterrupted training.
Independent W&B checks passed for the uploaded losses and token axes.
The record identifies reused credentials and other limits of this same-cluster test.

## 1. Experiment definitions

| Item | Definition |
|---|---|
| Model | Native TorchTitan dense Llama3 1B, vocabulary size 128,256 |
| Initialization | Random weights, without pretrained weights |
| Dataset | `fineweb_edu_long_30b_shuffled_v1` |
| Training reader | Finite ordered document stream with the existing packing and boundary masks |
| Validation | 936 fixed windows, each with 16,384 prediction targets |
| W&B smoke destination | Entity `rotation_optimizer`, project `debug`, online mode |
| Formal experiment | Approximately 20 billion nominal tokens, with exact settings pending |
| Formal checkpoints | Ten complete states at 10% through 100% of the nominal budget |

The smoke has eight optimizer steps.
It uses global batch size 512 at every step.

| Steps | Training length | Tokens per step | Cumulative tokens after the phase |
|---|---:|---:|---:|
| 1–2 | 2,048 | 1,048,576 | 2,097,152 |
| 3–4 | 4,096 | 2,097,152 | 6,291,456 |
| 5–6 | 8,192 | 4,194,304 | 14,680,064 |
| 7–8 | 16,384 | 8,388,608 | 31,457,280 |

Smoke saves occur at steps 2, 4, 6, and 8.
Eight steps cannot produce ten distinct completed training states.
The explicit smoke policy does not change the formal policy.
The recovery check stops after step 4 and resumes at step 5.
TorchTitan runs complete fixed validation after steps 1 and 8.
The step 1 event is native validator behavior, even when `validator.freq` is 8.

The smoke learning rate is `0.0001`, with no warmup or decay.
This value checks execution only.
It does not select a learning rate for the formal experiment.
The GPU probe selects the microbatch for each length.
Gradient accumulation preserves global batch size 512.

### Terms and counters

A **nominal token** is one position in a training input tensor.
Step `t` processes `B_t * L_t` nominal tokens.
`B_t` is the global batch size in sequences.
`L_t` is the actual training sequence length.
The cumulative count is the sum across completed optimizer steps.
Do not multiply that sum by the GPU count.

A **valid target** is a label that contributes to loss.
Masked labels do not contribute to loss.
Their input positions still count as nominal tokens.
Validation, lookahead positions, and diagnostic passes do not advance the training counter.

**Data parallelism**, or **DP**, distributes a global batch across model replicas.
A **rank** is one worker process.
A **microbatch** contains the sequences processed by one rank in one forward and backward pass.
An **accumulation count** is the number of these passes per optimizer step.
For DP degree `D`, microbatch `M_t`, and accumulation count `A_t`, require `B_t = D * M_t * A_t`.

The main experiment uses the training sequence directly for gradients.
Historical gradient crop experiments distinguish parent tokens and gradient fragment tokens.
Those historical counts do not define this experiment's budget.

| W&B key | Meaning |
|---|---|
| `n_tokens_seen` | Cumulative nominal training tokens after a completed step |
| `n_valid_tokens_seen` | Diagnostic cumulative valid prediction targets |
| `loss_metrics/global_avg_loss` | Mean training loss across valid targets on all ranks and accumulation passes |
| `loss_metrics/global_max_loss` | Maximum of the existing per-rank mean training losses |
| `validation_metrics/loss` | Loss across all 15,335,424 fixed validation targets |

The logger uses `n_tokens_seen` as the default horizontal axis for all three loss metrics.
It preserves their existing names and calculations.
Training losses use the actual training length.
Validation length remains 16,384 throughout training.
Do not add long/short loss series without an explicit experiment requirement.

## 2. Obtain the server information

Obtain a working Slurm launch example from the target server.
Record these fields before submission:

- Account, partition, QoS, GPU request syntax, and time limit.
- Python or module commands required by that server.
- Persistent paths for data, checkpoints, logs, and environments.
- Storage quota and temporary storage cleanup rules.
- Network access from compute nodes to Hugging Face and W&B.

Use the provided example to select those fields.
Do not copy VT ARC account names or partitions to another server.
The portable launcher supports one node with one or more GPUs.
Multi-node training is outside the verified launch procedure.
H100 and H200 are the target GPUs for this delivery.
Other hardware requires separate attention backend and memory checks.

Run these commands in the intended parent directory for the repository:

```bash
pwd
hostname
uname -sm
cat /etc/os-release
lscpu
free -h
df -h .
command -v git
command -v curl
command -v python3.12
```

Record the output with the current time.
Ask the server administrator for quota information if the example omits it.
Free filesystem space does not establish your personal allocation.

Run these commands inside a GPU allocation:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Require the requested GPUs and enough free memory.
A login node without GPUs does not establish compute node availability.
If these commands fail, check the allocation and driver before installing another PyTorch version.

## 3. Obtain the code and create the environment

The repository URL is `https://github.com/ErstinAn/dynamical-sequence-length.git`.
Use the default `main` branch.

Run these commands from the repository's parent directory:

```bash
git clone https://github.com/ErstinAn/dynamical-sequence-length.git
cd dynamical-sequence-length
git rev-parse HEAD
git status --short
```

The repository is private.
Use a GitHub account with repository access.
For an authentication failure, check that access before repeating the clone.
Require a clean working tree.
Record the resolved commit with each experiment.
Use that recorded commit when resuming or reproducing an experiment.

Use [environment_bootstrap.md](../environment_bootstrap.md) to install Python when Python 3.12 is absent.
That document specifies a user-owned Miniforge installation.
It does not require an existing Python environment.

If you used that Miniforge procedure, select its runtime in the current shell:

```bash
export PATH="$ENV_ROOT/conda/bin:$PATH"
export PYTHONNOUSERSITE=1
python --version
```

Require Python 3.12.14 from the new prefix.

If Python 3.12 is already available, run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install -r requirements-portable.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -c 'import sys, torch, torchtitan, wandb; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torchtitan.__file__)'
```

Use these exact versions:

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | `2.15.0.dev20260830+cu130` |
| CUDA runtime in the wheel | 13.0 |
| TorchTitan | Commit `737594746fda65a6d94dc9482ef07863a80c8588` |
| FlashAttention 3 for H100/H200 | `3.0.0`, from the pinned official PyTorch wheel |
| W&B | `0.29.0` |

`requirements-portable.txt` contains official wheel URLs with SHA256 values.
The nightly index no longer lists some required wheels.
The exact wheel URLs remain necessary for installation.
`requirements-transitive.txt` pins dependencies omitted from the original lock file.

The portable file also installs FA3 for the native Hopper attention path.
CPU-only model construction does not check this GPU-specific dependency.
Run the explicit FA3 registration check in the environment guide.
Do not substitute a current nightly build without a new verification run.

Require `pip check` to report no broken requirements.
Require imports from the new environment.
Inside the GPU allocation, run:

```bash
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
```

Require a CUDA tensor and the allocated GPU name.
If this check fails, compare the driver with the CUDA requirements in the environment guide.
Do not use a package upgrade as the first action.

## 4. Select persistent paths and authenticate

Run these commands from the repository root in the active environment:

```bash
export RUN_STORAGE=<PERSISTENT_STORAGE_DIRECTORY>
export DYNAMIC_SEQ_LEN_ASSETS="$RUN_STORAGE/assets"
export DYNAMIC_SEQ_LEN_OUTPUT="$RUN_STORAGE/runs"
export DATASET_DIR="$DYNAMIC_SEQ_LEN_ASSETS/data/fineweb_edu_long_30b_shuffled_v1"
export TOKENIZER_DIR="$DYNAMIC_SEQ_LEN_ASSETS/hf/llama3-tokenizer"
mkdir -p "$RUN_STORAGE" "$DYNAMIC_SEQ_LEN_ASSETS" "$DYNAMIC_SEQ_LEN_OUTPUT"
df -h "$RUN_STORAGE"
hf auth login
hf auth whoami
wandb login
```

Replace `<PERSISTENT_STORAGE_DIRECTORY>` with the approved server path.
Use a path visible from both login and compute nodes.
At VT ARC, the user approved `/projects/jason_optai/kangan/` for large files.
The current test uses a separate `dynamic_seq_len_handoff_20261118` directory there.
Another server must supply its own path.

Use your own credentials.
Your Hugging Face account needs read access to the private dataset.
The Git checkout supplies the tokenizer under its [Meta license](../../assets/tokenizer/LICENSE).
Your W&B account needs write access to `rotation_optimizer/debug`, or another explicitly agreed destination.
Keep credentials out of commands, YAML, logs, and Git.
The validation record must identify any reused authentication files.

## 5. Download and rebuild the dataset

The dataset repository is `long-sequence-length-dataset/fineweb-edu-long-30b`.
Use revision `b6ea2fde77bab36dff963aba5e6f482477959933`.

The tokenizer repository is `meta-llama/Meta-Llama-3-8B`.
Its pinned revision is `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`.
The downloader copies the tokenizer from `assets/tokenizer/` to `$TOKENIZER_DIR`.
It checks all three JSON files against the hashes in the pinned dataset manifest.
It also copies the license, attribution, use policy, and source record.

Run this command from the repository root:

```bash
python scripts/download_dataset.py \
  --dataset-dir "$DATASET_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --cache-dir "$RUN_STORAGE/download-cache" \
  --revision b6ea2fde77bab36dff963aba5e6f482477959933 \
  --max-workers 4
```

Require `text_download_complete: true` in `download_receipt.json` under the dataset directory.
The receipt records the revision, caches, and checksums.
The command checks all 236 text shards, document orders, validation arrays, and tokenizer files.
It does not create the training token cache.

For an interrupted download, repeat the same command with the same paths and revision.
Existing complete files can be reused.
The command checks their hashes before acceptance.
For HTTP 401 or 403, check `hf auth whoami` and repository access.
Request access from the repository owner if required.
Do not replace the dataset with a public dataset.

For a hash mismatch, keep the error record and download into a new destination.
Keep reference hashes unchanged.

Allocate CPU resources before tokenization.
The following example uses 32 CPUs with four processes and eight tokenizer threads per process.
Run it from the repository root inside that CPU allocation:

```bash
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=1
```

For a time estimate, run one complete shard first:

```bash
python scripts/rebuild_long_token_cache.py \
  --dataset-dir "$DATASET_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --workers 1 --limit-shards 1
```

This command records benchmark timing in `rebuild_progress.json`.
It does not report a complete cache.
The first shard contains 128,002,324 tokens.
Estimate total time from measured tokens per second and remaining tokens.
Treat linear worker scaling as an estimate, not a measured result.
Then rebuild the complete cache:

```bash
python scripts/rebuild_long_token_cache.py \
  --dataset-dir "$DATASET_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --workers 4 --resume
```

Adjust the process and thread counts to the allocated CPUs.
Their product must not exceed the allocation.
Require `complete: true` in `rebuild_progress.json`.
Require the manifest SHA256 below.
`--resume` reuses a shard only after source, token, and index hash checks.
Incomplete or corrupt shards are rebuilt.

The builder does not publish a complete result after an interrupted run.
Do not run the benchmark command after this completion check.

Preserve this directory structure:

```text
$RUN_STORAGE/
  assets/
    hf/llama3-tokenizer/
    data/fineweb_edu_long_30b_shuffled_v1/
      README.md
      dataset_manifest.json
      manifest.json
      verification.json
      tokenizer_provenance.json
      training_settings.json
      orders/train.npy
      orders/validation.npy
      validation/
      data/
      tokens/
  runs/
  download-cache/
```

Use actual directories for `data/` and `tokens/`.
Do not copy this machine's directory links as dataset contents.
The Parquet physical row order stays unchanged.
The training reader applies `orders/train.npy` before dividing documents among DP ranks.
A direct `datasets.load_dataset()` call does not apply that order.

| File | Required SHA256 |
|---|---|
| `manifest.json` | `4bf6946f7b7f7cbad0799746519baaa9ef2518f974d2239a8df2f184db004806` |
| `dataset_manifest.json` | `7e284ce4fc90a93e5bdcee160a60051dc29ed17775b8ecb261feafd93f047d80` |
| `validation/manifest.json` | `eb2faf62a36e324d07e08da73170e1268182cdfea1f38352362b86232f47155b` |

The manifests specify approximately 52.03 GB of text and 120.25 GB of token binaries.
These values use decimal GB and exclude caches, indexes, environments, and checkpoints.
The independent environment occupies approximately 8.1 GB after FA3 installation, excluding package caches.
The native model contains 1,235,814,400 parameters.
Each measured complete checkpoint occupies approximately 15.884 GB, including its stored state entries and metadata.
Eight retained checkpoints across the two smoke runs therefore require approximately 127.07 GB.

Use 400 GB as an initial storage estimate for the data, environment, caches, and paired smoke.
Retaining ten additional formal checkpoints needs approximately 158.84 GB more with the same model and optimizer state.
These checkpoint values use measured two-H200 smoke files, not only the number of unique model parameters.
Keep additional free space for interrupted downloads and complete checkpoints.
Recheck checkpoint sizes if the optimizer or distributed layout changes.

The first real shard rebuilt 128,002,324 tokens in 67.095 seconds with eight tokenizer threads.
The complete rebuild took 5,295.83 seconds, or 88.26 minutes, with four processes and eight tokenizer threads per process.
The continuation job checked the completed cache in 107.20 seconds without encoding more tokens.
These measurements exclude queue time and the full text download.
Use a longer time request when the target CPU or storage is slower.
See the [data record](../reports/fresh_data_20261118.md) for exact jobs, timings, and cache reuse.

The fixed validation arrays contain 936 windows of length 16,384.
Every target is valid, for 15,335,424 targets per complete validation.
Each window belongs to one document.
Windows from the same document do not overlap.
These arrays have no packing or padding.
This validation measures long-document performance, not the complete mixed data distribution.

Never regenerate these windows for a schedule comparison.

## 6. Run automatic checks

Check the real dataset from the repository root after reconstruction:

```bash
python docs/reports/fresh_data_20261118.py \
  --dataset-dir "$DATASET_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  > "$RUN_STORAGE/data-window-check.json"
```

Require `passed: true`, 236 checked indexes, 936 reconstructed windows, and 15,335,424 valid targets.
The checker compares shifted labels and positions with the newly rebuilt cache.
It also checks complete document permutations and disjoint windows, including each window's extra target position.
It does not change the dataset or select new validation members.

Run all CPU unit tests from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
```

These tests check data order, document boundaries, masks, fixed validation, gradients, schedules, nominal tokens, W&B axes, and recovery.
Require zero failures and zero errors.
The distributed validation test requires local sockets.
Run both distributed checks on a node that permits them:

```bash
FIXED_VALIDATION_GLOO_TEST=1 OMP_NUM_THREADS=1 python -m unittest \
  tests.test_fixed_validation.FixedBudgetValidationTest.test_real_two_rank_reduction_and_partial_batches -v
OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module tests.token_progress_distributed
```

Require both commands to succeed.
The second check writes ten complete checkpoints for a small CPU model.
It compares resumed updates with uninterrupted updates.

Check the concrete smoke plan before submitting GPUs:

```bash
python scripts/train.py configs/smoke.yaml --nproc <GPU_COUNT> --dry-run
```

Replace `<GPU_COUNT>` with the allocated count on one node.
Require eight steps, 31,457,280 nominal tokens, and checkpoints at steps 2, 4, 6, and 8.
The dry run does not execute training or create a run directory.

## 7. Run the GPU smoke

Use a new output directory for each acceptance attempt.
Run this command from the repository root inside the GPU allocation:

```bash
python scripts/gpu_smoke.py \
  --probe \
  --profile "$RUN_STORAGE/smoke-attempt-1/microbatches.json" \
  --assets "$DYNAMIC_SEQ_LEN_ASSETS" \
  --output "$RUN_STORAGE/smoke-attempt-1" \
  --nproc <GPU_COUNT>
```

Replace `<GPU_COUNT>` with the allocated GPU count.
Replace `smoke-attempt-1` for each new attempt.
The output directory must not already exist.
Use a persistent directory with enough capacity for two sets of complete checkpoints.

The command performs these operations:

1. Measure microbatches with the native model and real ordered training data.
2. Preserve a memory reserve for optimizer updates and distributed execution.
3. Run all eight training steps and the complete fixed validation.
4. Execute a second run with the same seed and configuration.
5. Stop the second run after the step 4 checkpoint.
6. Resume the second run through step 8 and complete validation.
7. Compare every final checkpoint tensor and state value on the CPU.

The probe itself does not update model weights.
Its single-GPU memory measurement estimates replicated DP memory requirements.
The actual distributed smoke must check the selected microbatches.
Do not use a profile from another GPU type or DP degree without a new probe.

Inspect these files under the attempt directory:

| File | Required result |
|---|---|
| `microbatches.json` | `status: passed`, matching DP degree, measured values for all four lengths |
| `uninterrupted-train.txt` | Process startup, W&B URL, and any process errors |
| `recovered-train.txt` | First recovery segment's process output |
| `recovered-resume.txt` | Resumed process output |
| `checkpoint_comparison.json` | `exact_match: true`, no differences |
| `acceptance.json` | Both run paths and completed comparison |

Each run path must contain complete checkpoints at steps 2, 4, 6, and 8.
Each checkpoint must contain DCP metadata and `token_milestone.json`.
**DCP** means PyTorch distributed checkpoint.

Training progress appears in `structured_logs/training.global_rank_0.*.jsonl` under each run directory.
The console log can contain only process and W&B output while training continues.
Use the structured logs, checkpoint metadata, and uploaded W&B history to check progress.
Run this command from the repository root:

```bash
tail -n 20 <RUN_DIRECTORY>/structured_logs/training.global_rank_0.*.jsonl
```

Replace `<RUN_DIRECTORY>` with a run path in `acceptance.json`.
During execution, locate the new directory under the attempt's `runs/` directory.
Require steps 1 through 8 for the uninterrupted case.
Require steps 1 through 4, then steps 5 through 8, for the recovery case.

Check actual W&B uploads before accepting the smoke.
The acceptance driver leaves the independent W&B history check marked as pending.

Run the independent checker from the repository root after the GPU driver completes:

```bash
python docs/reports/check_smoke_wandb.py \
  --acceptance "$RUN_STORAGE/smoke-attempt-1/acceptance.json" \
  --output "$RUN_STORAGE/smoke-attempt-1/wandb_verification.json"
```

Require `status: passed` and three W&B run URLs.
The checker reads uploaded records and metric definitions.
It does not create W&B runs.
For missing records, check upload completion and credentials before repeating the checker.

Require all three loss metrics in the selected debug project.
Require their default horizontal axis to be `n_tokens_seen`.
Require the final training count to equal 31,457,280.
Require each validation to report 15,335,424 valid targets without advancing the training count.
The validation token axes must be 1,048,576 at step 1 and 31,457,280 at step 8.
Reject missing uploads, non-finite losses, incomplete checkpoints, or failed comparisons.

Record GPU model, driver, microbatches, peak memory, step time, validation time, checkpoint size, and save time.
Use these measurements for the later formal resource discussion.

The accepted two-H200 run selected these microbatches:

| Sequence length | Microbatch per GPU | Accumulation count | Training seconds per step, two observed steps |
|---|---:|---:|---|
| 2048 | 32 | 8 | 13.55, 11.56 |
| 4096 | 16 | 16 | 23.75, 22.71 |
| 8192 | 8 | 32 | 48.08, 48.19 |
| 16384 | 4 | 64 | 105.84, 107.36 |

These training timings exclude checkpoint saves and validation.
Complete validation took approximately 60 seconds per event, including small loop overhead.
Checkpoint saves took approximately 17 to 23 seconds each.
The uninterrupted run reached 73.07 GiB of peak GPU memory.
The resumed process reached 75.64 GiB.
These values describe this allocation and do not replace the target server's probe.

### Use another Slurm server

Ask the collaborator for a working GPU job script from the target server.
Set `SITE` to a short site name.
For a Bash job script, set `SITE_EXAMPLE` to its absolute path.
From the repository root, create the site file once:

```bash
: "${SITE:?Set SITE to the target site name}"
: "${SITE_EXAMPLE:?Set SITE_EXAMPLE to the site script path}"
mkdir -p "cluster/$SITE"
cp -i "$SITE_EXAMPLE" "cluster/$SITE/run.sbatch"
```

Keep the site's resource directives, module commands, and environment setup.
Select one node and one driver task.
Set the shared checkout, Python environment, asset directory, and persistent output directory in the script.
Keep credentials outside the file.

Change to the checkout before the driver runs.
Make the script pass its arguments to Python.
For example, a site that runs one driver with `srun` can use this final line:

```bash
exec srun --ntasks=1 "$ENV_PREFIX/bin/python" "$@"
```

Set `ENV_PREFIX` in that script to the absolute Python environment prefix.
Follow the site's example if it requires a different command.
Each GPU runs one training process.

For Slurm, set `GPU_COUNT` to the number of GPUs requested on that node.
Set `RUN_STORAGE` and `DYNAMIC_SEQ_LEN_ASSETS` as described in section 4.
Check the script and submit the smoke from the repository root:

```bash
bash -n "cluster/$SITE/run.sbatch"
mkdir -p "$RUN_STORAGE/logs"
sbatch --output="$RUN_STORAGE/logs/smoke-%j.log" "cluster/$SITE/run.sbatch" \
  scripts/gpu_smoke.py --probe --nproc "$GPU_COUNT" \
  --assets "$DYNAMIC_SEQ_LEN_ASSETS" --output "$RUN_STORAGE/smoke-attempt-1" \
  --profile "$RUN_STORAGE/smoke-attempt-1/microbatches.json"
```

Expect a job ID.
Check the output files in section 7 after the job finishes.
Smoke checkpoints are under `$RUN_STORAGE/smoke-attempt-1/runs/<name>/<id>/checkpoint/`.

For formal training, use the same site file with the approved resource request.
Pass `scripts/train.py local/generated_configs/experiment.yaml --nproc "$GPU_COUNT"` after the site filename.
Set the formal YAML field `paths.dump_root` to `${OUTPUT_ROOT}`.
Set `DYNAMIC_SEQ_LEN_OUTPUT` to persistent storage in the job environment.
Section 9 gives the checkpoint path and recovery command.

Some sites accept a resource YAML through a local submission tool.
For that format, save the site file as `cluster/<site>/submit.yaml`.
Use the submission command from the example with the same Python driver and arguments.
Keep the training schedule in its own experiment YAML.
If the site uses another scheduler, use its example and submission command.

## 8. Submit on VT ARC

This section applies only to VT ARC.
Read the current [Slurm options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html).
Read the current [GPU resources](https://www.docs.arc.vt.edu/resources/gpu.html).
Read the current [storage rules](https://www.docs.arc.vt.edu/resources/storage.html).
Then check live availability and account access.

`cluster/vt_arc/smoke_handoff.sbatch` is the repository's GPU submission example.
It uses the supplied rotation optimizer example as a reference.
It keeps environment paths and persistent storage explicit.
Command-line Slurm options can override its resource defaults.
Use `jason_optai-paid` unless the user requests another account.

From the repository root, set these variables:

```bash
export HANDOFF_ROOT="$RUN_STORAGE"
export HANDOFF_CHECKOUT="$PWD"
export HANDOFF_VENV=<ABSOLUTE_ENVIRONMENT_PREFIX>
mkdir -p "$RUN_STORAGE/logs"
```

Replace `<ABSOLUTE_ENVIRONMENT_PREFIX>` with the prefix that contains `bin/python`.
For a repository venv, use `$PWD/.venv`.
For the current isolated Conda installation, use its recorded prefix.

The current two-H200 request uses this command:

```bash
sbatch --account=jason_optai-paid \
  --partition=h200_normal_q --qos=tc_h200_normal_short \
  --nodes=1 --ntasks=1 --gpus-per-node=2 \
  --cpus-per-task=16 --mem=128G --time=04:00:00 \
  --output="$RUN_STORAGE/logs/smoke-%j.log" \
  cluster/vt_arc/smoke_handoff.sbatch \
  scripts/gpu_smoke.py --probe \
  --profile "$RUN_STORAGE/smoke-attempt-1/microbatches.json" \
  --assets "$DYNAMIC_SEQ_LEN_ASSETS" \
  --output "$RUN_STORAGE/smoke-attempt-1" --nproc 2
```

Check the current QoS rules before reusing this request.
A scheduler estimate does not establish an actual execution time.
The wrapper creates one Slurm task.
The Python driver creates one training process per GPU.
Do not add another independent `torchrun` layer around this driver.

Record the returned job ID.
Use these commands from the login node:

```bash
squeue -j <JOB_ID>
scontrol show job <JOB_ID>
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Submit,Start,End,Elapsed,Timelimit
```

Replace `<JOB_ID>` with the returned ID.
Require Slurm `COMPLETED` with exit code `0:0` and all smoke acceptance checks.
Scheduler completion alone does not establish smoke acceptance.
Read the persistent job log for failures.

## 9. Prepare and resume the formal experiment

The formal sequence schedule and learning rate sweep remain pending.
Obtain the experiment document and its version before generating configurations.
Do not submit a formal run from the smoke settings.

Store generated configurations in `local/generated_configs/`.
Keep only `configs/smoke.yaml` under `configs/`.
Use `torchtitan_config: llama3_1b` and `smoke_test: false` for formal runs.
Require online W&B.

Specify these settings explicitly:

| Fields | Required decision |
|---|---|
| `token_budget`, `training.steps` | Exact budget and the first complete final step that reaches it |
| `training.seq_len` | Context capacity for validation and every scheduled training length |
| `training.global_batch_size`, `training.local_batch_size` | Initial global batch and measured microbatch |
| `seq_len_scheduler.*` | Sequence lengths, boundaries, and schedule kind |
| `batch_size_scheduler.*` | Global batch schedule |
| `microbatch_scheduler.*` | Measured microbatches for the target GPU and DP degree |
| `optimizer.learning_rate`, `lr_scheduler.*` | Agreed learning rate and schedule |
| `validator.freq` | Steps between complete fixed validations |

The data stream is finite.
The training manifest contains 1,348,067 documents and 30,000,004,070 raw text tokens.
The raw count excludes beginning-of-document and end-of-document tokens.
Nominal token budget is not identical to an epoch count or the original document token total.
Packing boundaries, special tokens, and final windows affect that relationship.
Preserve the document members and reading order across schedule comparisons.

Do not silently repeat the dataset after exhaustion.

For a constant global batch and length, calculate steps as `ceil(budget / (global_batch * sequence_length))`.
For changing schedules, sum each step's product until the first complete step reaches the budget.
The approximately 20B budget consumes only part of this finite dataset.
Its exact document position depends on the schedules and DP partition.

Run this command from the repository root:

```bash
python scripts/train.py local/generated_configs/experiment.yaml \
  --nproc <GPU_COUNT> --dry-run
```

Require ten distinct save steps at nominal thresholds 10%, 20%, through 100%.
The planner uses the first complete step that reaches each threshold.
It reports any excess above the threshold.
It rejects schedules that cross multiple thresholds in one step.
Keep all ten complete states.
Do not implement this rule as retention of the ten most recent checkpoints.

Discuss runtime, storage, GPU count, and sweep scope with the user after smoke acceptance.
Estimate training time separately for each sequence length and batch setting.
Multiply measured seconds per step by the planned steps for that setting.
Add all phase times, complete validation events, checkpoint saves, and process initialization.
Treat estimates from two steps per length as preliminary.

After agreement, execute the same command without `--dry-run` inside the GPU allocation.
Append `--index <INDEX>` to select a point from an agreed YAML sweep.

The run directory is `<DUMP_ROOT>/<NAME>/<EXPERIMENT_ID>`.
It contains the configuration, command, source hash, token plan, and W&B segment records.
Checkpoint directories are `checkpoint/step-<STEP>/` within that run directory.
Each checkpoint contains model, optimizer, learning rate scheduler, counters, and data reader state.

To resume an interrupted run, use the same code and environment.
Allocate the same DP degree.
Run this command from the repository root:

```bash
python scripts/train.py --resume-dir <RUN_DIRECTORY>
```

Replace `<RUN_DIRECTORY>` with the original run directory.
Do not supply replacement schedules or microbatches.
The launcher checks recorded source and configuration hashes.
To resume a run created before this cleanup, use the original checkout for that run.
It resumes the latest checkpoint with DCP metadata.
It rejects a run that already contains its final checkpoint.

Each recovery creates a new W&B run segment in the same experiment group.
The new segment uses the restored token count for its first updates.
This preserves earlier history when the checkpoint precedes the last uploaded step.
Use `segment-*.json` to identify W&B run IDs and recovery steps.

## 10. Diagnose failures

| Failure | Next action |
|---|---|
| Missing nightly wheel | Use `requirements-portable.txt` and its exact wheel URLs |
| CUDA initialization failure | Check allocation, driver, wheel CUDA version, and the environment guide |
| Hugging Face 401 or 403 | Check the account's read access to the private dataset |
| Download interruption | Repeat the pinned download command with the same destination |
| Data hash mismatch | Preserve the error and download the pinned data into a new directory |
| Tokenizer hash mismatch | Restore the tracked assets from the recorded Git commit. Retry with a clean tokenizer directory. |
| Tokenization time limit | Resume the builder with `--resume` inside a new CPU allocation |
| GPU out of memory | Measure a smaller microbatch and execute a new smoke attempt with the same global batch |
| Validation length mismatch | Restore the pinned arrays and keep context capacity at least 16,384 |
| Multi-GPU startup failure | Check GPU visibility, DP count, one-task launch structure, and the complete worker logs |
| Recovery mismatch | Check source hashes, configuration, DP degree, and complete checkpoint metadata |
| W&B failure | Check online mode, credentials, project access, and compute-node network access |

Do not change the model, data members, reading order, masks, loss, or gradient normalization to bypass a failed check.
Keep raw logs and the failed attempt directory.
Record every reused environment or cache.
Record unexecuted checks and their reasons.
Use [collaborator_prompt.md](collaborator_prompt.md) for the next agent's startup instructions.
