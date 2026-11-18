# Start the smoke test

Open Codex or Claude Code on the target server.
Copy the text below and fill in the first two lines.
Use absolute paths visible to the agent.
For a site resource YAML, include its submission command on the first line.

```text
Working GPU job example: [PATH TO SITE JOB FILE]
Persistent storage for data, checkpoints, and logs: [WRITABLE SHARED DIRECTORY]

Run the Llama 3 1B smoke test on this server.
Use the private repository https://github.com/ErstinAn/dynamical-sequence-length.git on main.
Clone the repository or use a clean checkout of the current main commit.
Record the commit. Read docs/codebase_kick_off/getting_started.md.

Use the job example to identify the account, queue, GPU request, and submission command.
Use the supplied directory as RUN_STORAGE. Check the quota on that path.
Check GitHub access, private dataset access, and online write access to rotation_optimizer/debug.
If login requires my account, tell me which service to log into on the server.
Keep credentials out of the conversation and job files.

Follow the guide to install the environment and prepare the pinned dataset.
Use the tokenizer from assets/tokenizer/.
Run the required data, CPU, and distributed checks.
Adapt the site job file and use configs/smoke.yaml.
Measure microbatches on the allocated GPUs.
Run scripts/gpu_smoke.py with --probe and monitor the job through completion.

Verify all eight steps, 31,457,280 nominal training tokens, and the fixed validation set.
Check the four smoke checkpoints and the exact recovery comparison.
Verify the uploaded W&B losses against the token axis.
Report the job ID, run directories, checkpoint paths, W&B links, and test results.
Record any reused caches or environments and any unresolved failures.
Finish after the smoke test. Leave the 20B training run and LR sweep for the later experiment document.
```
