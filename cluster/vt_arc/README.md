# VT ARC jobs

The Slurm tools are optional.
Local training does not import these tools.
The tools run one node with one Slurm task.
Torchrun creates one process for each requested GPU.

## Prepare a job

1. Read the current [ARC job options](https://www.docs.arc.vt.edu/usage/job_scheduling/02_slurm_options.html).
2. Check your account, partition, and QoS with the ARC tools.
3. Copy an experiment file to the ignored `local/` directory.

   ```bash
   mkdir -p local
   cp configs/seq512.yaml local/experiment.yaml
   ```

4. Add the block from `resources.example.yaml` to the copied file.
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
    python cluster/vt_arc/submit.py local/experiment.yaml --dry-run
    ```

11. Submit the job.

    ```bash
    python cluster/vt_arc/submit.py local/experiment.yaml
    ```

The resource example does not reserve GPUs or prove current account access.
The scheduler checks the requested account and resources.
The [ARC compute page](https://www.docs.arc.vt.edu/resources/compute.html) identifies the clusters.
The [ARC GPU page](https://www.docs.arc.vt.edu/resources/gpu.html) describes their GPU resources.

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
