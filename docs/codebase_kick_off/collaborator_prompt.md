# Replace the two quoted paths. Paste the whole block into Codex or Claude Code on the new server.

```text
JOB_EXAMPLE="/REPLACE/WITH/YOUR/WORKING_GPU_JOB_FILE"
RUN_STORAGE="/REPLACE/WITH/YOUR/SHARED_STORAGE_DIRECTORY"

Run the smoke test on this Slurm cluster. Clone https://github.com/ErstinAn/dynamical-sequence-length.git on main.
Use one node with eight NVIDIA H200 GPUs. Formal runs use the same layout by default.
Follow docs/codebase_kick_off/getting_started.md. Start with JOB_EXAMPLE. Check the site instructions and live Slurm settings yourself.
Identify the submission command, GPU request, permitted account, partition, QoS, and time limit.
Use RUN_STORAGE for data, logs, and checkpoints. Create local job files and settings. Get permission before changing shared code.
Verify online W&B write access to dynamic_seq_length/debug before submission and inside the job.
Follow the guide to use your site credential or your own ignored local key file. Keep key values out of prompts and logs.
Download Erstin/fineweb-edu-long-30b anonymously at the commit in the guide. Use the bundled tokenizer.
If access fails, follow the guide for that service. Report any unresolved error once.
Start one driver with access to all eight GPUs. Use --nproc 8.
Verify training, all 936 validation windows, and recovery on the same eight GPUs. Report the allocation, job ID, checkpoint paths, and W&B links.
Stop after the smoke test.
```
