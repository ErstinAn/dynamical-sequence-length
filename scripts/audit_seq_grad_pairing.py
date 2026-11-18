#!/usr/bin/env python3
"""Verify real packed parents/crops and exact target counts without model work."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchtitan_dynamic_seq_len.data.packed_documents import PackedDocumentDataset
from torchtitan_dynamic_seq_len.seq_grad.sampler import PackedProbeLoader, PairedWindowProbeLoader, SingleDocumentProbeLoader, stream_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    fixed = config["fixed"]
    assert fixed["seq_grad.sampling"] == "paired_window"
    n, h = fixed["seq_grad.probe_global_batch_size"], fixed["seq_grad.reference_length"]
    seed, degree = fixed["seq_grad.seed"], config["resources"]["gpus_per_node"]
    source = PackedDocumentDataset(fixed["dataloader.dataset_path"], split="train", seq_len=h, infinite=True)
    state = copy.deepcopy(source.state_dict())
    single = fixed.get("seq_grad.parent_population", "packed") == "single_document"
    base = SingleDocumentProbeLoader(source, h) if single else PackedProbeLoader(source)
    sampler = PairedWindowProbeLoader(base, h)
    rows = []
    # The acceptance configuration has events 0/2/4. Restrict to these three
    # events even if the full-run config was supplied for a separate audit.
    for step in (0, 2, 4):
        full = base.sample_batch(length=h, global_sequences=n, local_batch_size=n,
                                rank=0, degree=1, seed=seed, step=step)
        parent_x, parent_y = full[0]
        locations = []
        if single:
            for i in range(n):
                file, row, offset = base.locate(stream_seed(seed, step, h, i, "single-document-parent-v1"))
                size = int(source.indexes[file][row, 1])
                assert size > h and offset + h < size
                locations.append(dict(file=file, row=row, raw_tokens=size, parent_start=offset))
            assert int((parent_y != -100).sum()) == n*h
            torch.testing.assert_close(parent_x["positions"], torch.arange(h).expand(n, h), rtol=0, atol=0)
        digest = hashlib.sha256()
        for value in [parent_x["input"], parent_y, parent_x["positions"]]:
            digest.update(value.numpy().tobytes())
        targets = {str(length): 0 for length in [h, *fixed["seq_grad.lengths"]]}
        checked = 0
        ranges = {str(length): [h, -1] for length in fixed["seq_grad.lengths"]}
        for rank in range(degree):
            kw = dict(global_sequences=n, local_batch_size=1, rank=rank, degree=degree, seed=seed, step=step)
            local_parents = sampler.sample_batch(length=h, **kw)
            for j, (inputs, labels) in enumerate(local_parents):
                i = rank + j * degree
                torch.testing.assert_close(labels[0], parent_y[i], rtol=0, atol=0)
                for name in inputs:
                    torch.testing.assert_close(inputs[name][0], parent_x[name][i], rtol=0, atol=0)
                targets[str(h)] += int((labels != -100).sum())
            for length in fixed["seq_grad.lengths"]:
                for j, (inputs, labels) in enumerate(sampler.sample_batch(length=length, **kw)):
                    i = rank + j * degree
                    offset = int(np.random.default_rng(stream_seed(seed, step, length, i, "paired-window-v1")).integers(h-length+1))
                    assert 0 <= offset <= h-length
                    ranges[str(length)][0] = min(ranges[str(length)][0], offset)
                    ranges[str(length)][1] = max(ranges[str(length)][1], offset)
                    torch.testing.assert_close(inputs["input"][0], parent_x["input"][i, offset:offset+length], rtol=0, atol=0)
                    torch.testing.assert_close(labels[0], parent_y[i, offset:offset+length], rtol=0, atol=0)
                    # Document reset indices define the allowed causal segments.
                    original = parent_x["positions"][i, offset:offset+length]
                    boundaries = torch.zeros(length, dtype=torch.long)
                    boundaries[original == 0] = torch.arange(length)[original == 0]
                    expected_positions = torch.arange(length) - boundaries.cummax(0).values
                    torch.testing.assert_close(inputs["positions"][0], expected_positions, rtol=0, atol=0)
                    targets[str(length)] += int((labels != -100).sum())
                    checked += 1
            sampler.clear_event()
        assert targets[str(h)] == int((parent_y != -100).sum())
        assert source.state_dict() == state
        rows.append(dict(step=step, parent_sha256=digest.hexdigest(), checked_short_windows=checked,
                         valid_targets=targets, window_offset_ranges=ranges, parent_documents=locations))
    report = dict(passed=True, protocol=sampler.identity, reference_targets="all_valid_targets_in_full_parent",
                  global_parent_sequences=n, reference_length=h, dp_degree=degree,
                  parent_source_state_unchanged=True, events=rows)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "events": len(rows), "checked_short_windows": sum(r["checked_short_windows"] for r in rows)}))


if __name__ == "__main__":
    main()
