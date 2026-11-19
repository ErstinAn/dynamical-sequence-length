# VT ARC smoke submission

Start in the repository root.
Use the [smoke procedure](../../docs/codebase_kick_off/getting_started.md#submit-and-verify-the-allocation).

1. Submit `cluster/vt_arc/smoke_handoff.sbatch` with `sbatch`.
2. Pass the Python driver and its arguments after the script name.

The job requests one H200 node with eight GPUs. [ARC GPU resources](https://www.docs.arc.vt.edu/resources/gpu.html)
The script selects the Python environment and checks the dataset.
It starts one driver with `srun`.
The driver starts one training process per GPU.

The smoke driver writes checkpoints below its `--output` directory.
Formal training uses the experiment YAML and the persistent output path.

For another server, follow the [site setup procedure](../../docs/codebase_kick_off/getting_started.md#submit-and-verify-the-allocation).
