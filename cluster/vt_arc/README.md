# VT ARC smoke submission

Start in the repository root.
Use the [ARC submission command](../../docs/reports/arc_smoke_plan_20261118.md#submission-procedure).

1. Submit `cluster/vt_arc/smoke_handoff.sbatch` with `sbatch`.
2. Pass the Python driver and its arguments after the script name.

Slurm allocates one node.
The script selects the Python environment and checks the dataset.
It starts one driver with `srun`.
The driver starts one training process per GPU.

The smoke driver writes checkpoints below its `--output` directory.
Formal training uses the experiment YAML and the persistent output path.

For another server, follow the [site setup procedure](../../docs/codebase_kick_off/getting_started.md#submit-on-the-target-server).
