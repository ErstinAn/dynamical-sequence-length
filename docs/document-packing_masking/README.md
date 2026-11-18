# Document packing and masking

Use the [historical revision](https://github.com/ErstinAn/dynamical-sequence-length/tree/427d321c40e8b82444d2f5c07082e6155c0f914e) for the commands on this page.

The packed reader retains short documents and document tails.
Document masking is configurable and defaults to `true`.
Train and validation use the same packing and attention rules.

- Read the [implementation plan](implementation-plan.md) for the intended behavior.
- Read the [implementation report](implementation-report.md) for changes, checks, and limitations.
- Use the [new experiment configurations](https://github.com/ErstinAn/dynamical-sequence-length/tree/427d321c40e8b82444d2f5c07082e6155c0f914e/legacy/configs/longseq_packed) for the six schedules.
- Use the [test configurations](https://github.com/ErstinAn/dynamical-sequence-length/tree/427d321c40e8b82444d2f5c07082e6155c0f914e/legacy/configs/longseq_packed_debug) for short GPU checks.

CPU checks and two B200 GPU jobs passed.
The report records results, job IDs, and W&B links.

Read the [formal submission record](formal-submission.md) for the six production jobs and the new W&B project.
