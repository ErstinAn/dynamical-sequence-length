# Historical experiment configurations

This directory's `configs/` subdirectory contains the 76 YAML files from before cleanup.
Their contents match the original files.
`config_checksums.json` records the original SHA256 values.
Regression tests continue to use these configurations.

These configurations describe historical experiments.
They can select different models, old validation sets, local absolute paths, or ARC resources.
Some configurations disable W&B or retain only the two most recent checkpoints.
Those settings do not meet the requirements for the current 1B experiment.

For new experiments, use `configs/smoke.yaml` at the repository root.
Read the [main run guide](../docs/getting_started.md).
Do not change historical hashes or dataset identities to launch a new experiment.
To resume an old run, use that run's frozen source and configuration.
The directory changes also change the source digest checked by the launcher.

The old configuration generators remain available for historical review.
They write new results to `local/generated_configs/` without overwriting this directory.
They do not define the schedules for the current experiment.

`reports/` contains historical result files and the earlier W&B loss audit.
The earlier conversation record remains in ignored `local/private_archive/` outside this directory.
It has no runtime dependency and is excluded from the delivery bundle.
`report_checksums.json` records their original paths and SHA256 values.
Their contents remain unchanged.
Old paths inside those historical records describe the repository at the time of the record.

List historical builders with `python scripts/train.py --list-legacy` from the repository root.
The default `--list` shows only the current `llama3_1b` entry.
