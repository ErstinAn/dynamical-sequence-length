# Document packing and masking

The packed reader retains short documents and document tails.
Document masking is configurable and defaults to `true`.
Train and validation use the same packing and attention rules.

- Read the [implementation plan](implementation-plan.md) for the intended behavior.
- Read the [implementation report](implementation-report.md) for changes, checks, and limitations.
- Use the [new experiment configurations](../../legacy/configs/longseq_packed/) for the six schedules.
- Use the [test configurations](../../legacy/configs/longseq_packed_debug/) for short GPU checks.

CPU checks and two B200 GPU jobs passed.
The report records results, job IDs, and W&B links.

Read the [formal submission record](formal-submission.md) for the six production jobs and the new W&B project.
