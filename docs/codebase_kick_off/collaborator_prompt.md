# Replace the two quoted paths. Paste the whole block into Codex or Claude Code on the new server.

```text
JOB_EXAMPLE="/REPLACE/WITH/YOUR/WORKING_GPU_JOB_FILE"
RUN_STORAGE="/REPLACE/WITH/YOUR/SHARED_STORAGE_DIRECTORY"

Run the smoke test on this Slurm cluster. Clone https://github.com/ErstinAn/dynamical-sequence-length.git on main.
Use 2 nodes with 8 NVIDIA H200 GPUs each (16 GPUs total). Formal runs use the same layout by default.
Follow docs/codebase_kick_off/getting_started.md. Start with JOB_EXAMPLE. Check the site instructions and live Slurm settings yourself.
Identify the submission command, GPU request, permitted account, partition, QoS, and time limit.
Use RUN_STORAGE for data, logs, and checkpoints. Create local job files and settings. Get permission before changing shared code.
Check two-node launch and fixed validation support before submission. Report any blocker that needs a user decision.
Verify training, validation, and recovery on all 16 GPUs. Report the allocation, job ID, checkpoint paths, and W&B links.
Stop after the smoke test.
```
