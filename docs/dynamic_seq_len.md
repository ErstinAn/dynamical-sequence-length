# Dynamic sequence length

`DynamicSeqLenTrainer` is a subclass of the TorchTitan Trainer.
It uses the existing model, optimizer, and training scripts.
The primary repository contains all necessary code and configuration files.
The [writing guide](writing.md) gives the meanings of the technical terms in this document.

## Experiment configurations

The six experiment configurations in `configs/dynamic_seq/` have the same training parameters as the completed experiments.
Each configuration uses 3840 optimizer steps.
The global batch size is 512 sequences, and the local batch size is 16 sequences.
The learning rate is constant at `0.0005`, with no warmup.
Each configuration uses 2,013,265,920 training label tokens when all sequences are full.

| File | Training length |
| --- | --- |
| `constant.yaml` | 1024 for all steps. |
| `increasing.yaml` | 512, 768, 1024, 1280, 1536. 768 steps at each level. |
| `decreasing.yaml` | The levels of `increasing.yaml` in the opposite sequence. |
| `triangular.yaml` | 512, 768, 1280, 1536, 1280, 768. 128 steps per level, five complete periods. |
| `cyclic.yaml` | 512 to 1536 and back. change of 8 per step, period of 256 steps, 15 periods. |
| `cyclic_period64.yaml` | 512 to 1536 and back. change of 32 per step, period of 64 steps, 60 periods. |
| `cyclic_sweep.yaml` | Two training runs, with periods of 256 and 64 steps. |
| `smoke.yaml` | Eight steps with a small model and the test data in the repository. |

`triangular` keeps each length constant for several steps.
`cyclic` changes length at each step, including immediately after the maximum or minimum.
The same token budget can give different counts of floating-point operations and different execution times.
The sweep selects the two cyclic configurations in the table.
The smoke test uses a different, smaller budget.

The constant configuration uses `DynamicSeqLenTrainer` for comparison with the other schedules.
`configs/lr_sweep_constant.yaml` uses the TorchTitan Trainer.
That sweep has 3819 steps and five learning rates.
The configurations for gradient variance use a different Trainer.

## Training procedure

### Cyclic experiments with larger length ranges

Two additional configurations keep the period at 256 optimizer steps for 15 periods.
They use two B200 GPUs with a local batch size of 64 sequences.
The global batch size stays at 512 sequences, with four accumulation groups for each optimizer step.
The learning rate stays at `0.0005`, and validation uses sequence length 1024.

| Configuration | Length range | Total training tokens | Slurm time limit |
| --- | --- | --- | --- |
| `cyclic_8_2040.yaml` | 8 to 2040 | 2,013,265,920 | 8 hours |
| `cyclic_512_8192.yaml` | 512 to 8192 | 8,556,380,160 | 24 hours |

The first configuration rounds lengths to multiples of eight and preserves the original token budget.
The second changes length by 60 tokens per step and uses 4.25 times the original token budget.
Their model capacities are 2040 and 8192, respectively.
They use `loss.num_chunks=4` because some lengths in the second configuration are not divisible by eight.
The loss calculation uses the same normalization by valid tokens.

### Prepare and run training

1. Install the dependencies as specified in the project README.
2. Prepare the tokenizer as specified in the project README.
3. For full training, prepare the validation artifact with sequence length 1024 as specified in [data preparation](data.md).
4. If you use shared storage, set `DYNAMIC_SEQ_LEN_ASSETS` to the asset directory.
5. If you use shared storage, set `DYNAMIC_SEQ_LEN_OUTPUT` to the output directory.
6. Examine the smoke test command.

   ```bash
   python scripts/train.py configs/dynamic_seq/smoke.yaml --nproc 1 --dry-run
   ```

7. In a GPU allocation, run the smoke test.

   ```bash
   python scripts/train.py configs/dynamic_seq/smoke.yaml --nproc 1
   ```

8. In an allocation with two GPUs, run the selected experiment.

   ```bash
   python scripts/train.py configs/dynamic_seq/cyclic.yaml --nproc 2
   ```

The following command selects index 1 from the cyclic sweep.
Index 0 selects a period of 256 steps. index 1 selects a period of 64 steps.

```bash
python scripts/train.py configs/dynamic_seq/cyclic_sweep.yaml --nproc 2 --index 1
```

## Slurm submission procedure

Full experiment configurations include a Slurm `resources` block for two A100 GPUs and a limit of four hours.
Local training does not use this block.
The submission tool uses `resources.account` to select the allocation account.
The [ARC job options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html) give information about the resource settings.
The selected short QoS has a higher billing factor.

1. Make a directory for your submission configuration.

   ```bash
   mkdir -p outputs/launch
   ```

2. Copy the selected configuration to this directory.

   ```bash
   cp configs/dynamic_seq/cyclic_sweep.yaml outputs/launch/experiment.yaml
   ```

3. Set `resources.account` to your allocation account in the copied file.
4. Examine the cluster, partition, and QoS for your allocation.
5. Set the Python executable.

   ```bash
   export DYNAMIC_SEQ_LEN_PYTHON="$(command -v python)"
   ```

6. Examine the submission command.

   ```bash
   python cluster/vt_arc/submit.py outputs/launch/experiment.yaml --dry-run
   ```

7. Submit the job.

   ```bash
   python cluster/vt_arc/submit.py outputs/launch/experiment.yaml
   ```

This sweep submits two array tasks.
With the default setting, Slurm runs one task at a time.
`resources.max_concurrent` sets the maximum number of tasks that can run at the same time.
Each task has a independent output directory, checkpoint, and W&B ID.
A configuration with `sweep: {}` submits one task.

The default configurations do not send data to W&B.
The completed dynamic experiments used the W&B project `dynamic_seq_length/finweb-llama-100m-seqlength-scheduler`.
Their allocation account was `jason_optai-paid`.
A collaborator must use an allocation account that gives them access.

### W&B setup procedure

1. Set `wandb.enabled` to `true`.
2. Set `wandb.mode` to `online`.
3. Set `wandb.entity` to your entity.
4. Set `wandb.project` to your project.
5. Run `wandb login` to authenticate your account.

### Resources in the completed experiments

The constant and increasing experiments used H200 GPUs.
The decreasing, triangular, and both cyclic experiments used A100 GPUs.
The LR sweep used B200 GPUs.
Its resource block keeps the B200 selection and the limit of eight hours.
All full dynamic configurations in the primary repository select A100 GPUs for new experiments.

## Scheduler parameters

The first optimizer step has the number 1.

| Kind | Parameters |
| --- | --- |
| `constant` | `length` |
| `increasing`, `decreasing` | `lengths` from minimum to maximum. optional `boundaries` give the last step at each level. |
| `triangular` | `lengths` from minimum to maximum and `period_steps`. |
| `cyclic` | `min_length`, `max_length`, `period_steps`, and optional `round_to_multiple`. |

For a cyclic schedule, the period must be an even number.
Without rounding, the difference between the maximum and minimum lengths must be divisible by half the period.
This gives an integer change in length at each step.
The schedule with a period of 64 steps uses this formula:

```python
phase = (step - 1) % 64
seq_len = 512 + 32 * min(phase, 64 - phase)
```

Set `seq_len_scheduler.round_to_multiple: 8` to round lengths to multiples of eight.
With rounding, the period must be divisible by four.
The endpoints and their midpoint must be divisible by the specified multiple.
The scheduler rounds the lower half of the length range and reflects it about the midpoint.
This keeps the mean length equal to the midpoint over each complete period.

## Data and validation

Before each optimizer step, the Trainer sets the sequence length in the dataset.
Then, the Trainer reads the batches for all accumulation groups.
All ranks and accumulation groups use the same sequence length.
The dataset keeps unused tokens in its buffers.
After checkpoint recovery, it uses these buffers before it reads more documents.

Dynamic training uses `num_workers=0`, with no prefetch.
It operates on one GPU or with replicated data parallelism.
`training.seq_len` sets the model and RoPE capacity.
The full experiment configurations use a capacity of 2048.

The scheduled lengths must be less than or equal to that capacity.
They must be divisible by `loss.num_chunks`, which is eight in these configurations.
Powers of two are not necessary for sequence lengths.
Validation uses `validator.eval_seq_len=1024` for all training lengths.
Each validation operation uses 20,971,520 label tokens from the fixed artifact.

## Metrics

The logger puts `seq_len` after the learning rate metrics in the same record.
`valid_tokens_per_step` gives the count of valid labels across all ranks for the specified step.
`n_tokens_seen` gives the total number of training token positions up to the specified step.
`n_valid_tokens_seen` gives this total without labels that have the ignore value.
Neither training counter includes validation tokens.

When all sequences are full, the counters obey this equation:

```text
n_tokens_seen[t] - n_tokens_seen[t - 1] = global_batch_size * seq_len[t]
```

When length increases linearly, the cumulative token curve becomes steeper.
On a step axis, `valid_tokens_per_step` and `seq_len` have the same shape when all labels are valid.
The throughput calculation uses the token count in each measurement interval.
The FLOP calculation uses the sequence lengths and token counts in that interval.

## Code and tests

The scheduler is in `torchtitan_dynamic_seq_len/schedulers/sequence_length.py`.
The Trainer and metrics modules are in `torchtitan_dynamic_seq_len/dynamic_seq/`.
Data loader modules are in `torchtitan_dynamic_seq_len/data/`.
The other training configurations use their existing Trainer and metrics classes.
The [test procedure](tests.md) gives the CPU, distributed, and GPU checks.

`scripts/benchmark_dataloader.py` compares TorchTitan training with zero workers, four workers, and cached batches on the CPU.
`configs/benchmark_dataloader.yaml` supplies the benchmark configuration.
The benchmark uses the same GPU allocation for all cases.
The measured execution time does not include warmup.
It includes data transfer from the CPU to the GPU.

New experiments use the primary repository.
A development copy in `local/` is not necessary.
Historical manifests and checkpoints keep their initial source identity.
Recovery from a saved checkpoint uses the source and configuration from the same experiment.
A change to the directory structure does not make a saved manifest compatible with a different source hash.
