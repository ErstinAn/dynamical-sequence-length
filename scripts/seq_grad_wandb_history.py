"""Handle W&B history rows that contain no metrics from the requested group."""


def scan_metric_rows(run, keys):
    rows = []
    for row in run.scan_history(keys=list(keys)):
        values = [row.get(key) for key in keys]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise RuntimeError(f"Incomplete W&B metric group at history step {row.get('_step')}")
        rows.append(row)
    return rows
