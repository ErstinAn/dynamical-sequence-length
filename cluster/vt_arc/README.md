# VT ARC jobs

The Slurm tools are optional.
Local training does not import these tools.
The tools run one node with one Slurm task.
Torchrun creates one process for each requested GPU.

## Prepare a job

1. Read the current [ARC job options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html).
2. Check your account, partition, and QoS with the ARC tools.
3. Copy an experiment file to the ignored `outputs/launch/` directory.

   ```bash
   mkdir -p outputs/launch
   cp configs/dynamic_seq/cyclic_sweep.yaml outputs/launch/experiment.yaml
   ```

4. Examine the `resources` block in the copied file.
   If this block is absent, add the block from `resources.example.yaml`.
5. Replace `YOUR_ACCOUNT` with your ARC account.
6. Set the cluster, partition, and QoS for your allocation.
7. Set the paths to storage that the compute nodes can access.
8. Activate the training environment.
9. Set the Python executable.

   ```bash
   export DYNAMIC_SEQ_LEN_PYTHON="$(command -v python)"
   ```

10. Check the submission command.

    ```bash
    python cluster/vt_arc/submit.py outputs/launch/experiment.yaml --dry-run
    ```

11. Submit the job.

    ```bash
    python cluster/vt_arc/submit.py outputs/launch/experiment.yaml
    ```

The resource example does not reserve GPUs or prove current account access.
The scheduler checks the requested account and resources.
The [ARC compute page](https://www.docs.arc.vt.edu/resources/compute.html) identifies the clusters.
The [ARC GPU page](https://www.docs.arc.vt.edu/resources/gpu.html) describes their GPU resources.

## Included experiments

`configs/dynamic_seq/` contains six experiment configurations and a sweep for two cyclic periods.
`configs/lr_sweep_constant.yaml` contains the sweep for five learning rates with the TorchTitan Trainer.
Both use this same submission tool and shell script.
See the [dynamic training guide](../../docs/dynamic_seq_len.md) for the schedule definitions.

The default configurations do not send data to W&B.
Their allocation account field contains `YOUR_ACCOUNT`.

The completed experiments used `jason_optai-paid`.
Before submission, select an account that gives you access.
For online W&B records, use the setup procedure in the dynamic training guide.

Before submission, set `DYNAMIC_SEQ_LEN_ASSETS` to an asset directory that the compute nodes can read.
Set `DYNAMIC_SEQ_LEN_OUTPUT` to an output directory that the compute nodes can write to.
While Slurm has tasks in the queue or in operation, do not change the source files for those tasks.

## Submission records

Each submission saves a resolved YAML file and a manifest.
The manifest records a run ID, configuration hash, source hash, and Git revisions.
The compute process checks the source hash before training.
A source change after submission causes an error.
The source hash does not include changes inside the external TorchTitan installation.

The tools use `--export=ALL` to pass the process environment.
The Slurm step keeps all requested GPUs visible to torchrun.
W&B is off unless the experiment file enables it.

## Restore a checkpoint

1. Select the initial configuration snapshot and manifest.
2. Restore the source files from that submission.
3. Check the recovery command.

   ```bash
   python cluster/vt_arc/submit.py /path/to/config.yaml \
     --resume-manifest /path/to/manifest.yaml --resume-index 0 --dry-run
   ```

4. Run the same command without `--dry-run`.

Complete checkpoint metadata and the initial configuration identity are necessary for recovery.
A writer lock prevents two processes from writing to one run directory.
The `--time` option changes only the requested walltime.
The version 2 manifest belongs to this repository layout.
For historical version 1 runs, use their previous source and configuration.
