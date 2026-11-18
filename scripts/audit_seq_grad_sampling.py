#!/usr/bin/env python3
"""Replay a frozen run's probe starts and measure source-position overlap on CPU."""

import argparse
import bisect
import hashlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def pieces(start, size, population):
    end = start + size
    return [(start, end)] if end <= population else [(start, population), (0, end - population)]


class Coverage:
    def __init__(self, intervals):
        self.intervals = merge(intervals)
        self.ends = [end for _, end in self.intervals]

    def overlap(self, intervals):
        count = 0
        for left, right in intervals:
            i = bisect.bisect_right(self.ends, left)
            while i < len(self.intervals) and self.intervals[i][0] < right:
                start, end = self.intervals[i]
                count += max(0, min(right, end) - max(left, start))
                i += 1
        return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    # Import the implementation actually executed by this run, not live edits.
    snapshot = Path(plan["source"]).resolve()
    sys.path.insert(0, str(snapshot))
    from torchtitan_dynamic_seq_len.data.packed_documents import PackedDocumentDataset
    from torchtitan_dynamic_seq_len.seq_grad.sampler import PackedProbeLoader, stream_seed

    stage = plan["full"]
    config = yaml.safe_load(Path(stage["config"]).read_text())
    fixed = config["fixed"]
    if fixed.get("seq_grad.sampling", "independent") != "independent":
        raise ValueError("This overlap audit applies only to independent probes; use audit_seq_grad_pairing.py for paired windows")
    events = [json.loads(line) for line in
              (Path(stage["dump_folder"]) / "seq_grad_events.jsonl").read_text().splitlines()]
    updates = [e["metrics"]["context_bias/step"] for e in events]
    source = PackedDocumentDataset(fixed["dataloader.dataset_path"], split="train",
                                  seq_len=fixed["training.seq_len"], infinite=True)
    sampler = PackedProbeLoader(source)
    original_state = source.state_dict()
    population = int(sampler.file_ends[-1])
    sizes = np.concatenate([np.asarray(index[:, 1], dtype=np.int64) + 2 for index in source.indexes])
    ends = np.cumsum(sizes)
    starts = ends - sizes
    degree = config["resources"]["gpus_per_node"]
    n = fixed["seq_grad.probe_global_batch_size"]
    lengths = fixed["seq_grad.lengths"]
    reference = fixed["seq_grad.reference_length"]
    per_rank = fixed["training.seq_len"] * fixed["training.global_batch_size"] // degree
    train_partitions = []
    for rank in range(degree):
        ids = np.arange(rank, len(sizes), degree)
        train_partitions.append((ids, np.cumsum(sizes[ids])))

    def training_coverage(first_update, last_update):
        spans = []
        for ids, rank_ends in train_partitions:
            left, right = first_update * per_rank, last_update * per_rank + 1
            assert right < int(rank_ends[-1]), "Audit currently expects less than one training epoch"
            begin = int(np.searchsorted(rank_ends, left, side="right"))
            stop = int(np.searchsorted(rank_ends, right - 1, side="right"))
            for i in range(begin, stop + 1):
                base = int(rank_ends[i - 1]) if i else 0
                doc = int(ids[i])
                spans.append((int(starts[doc]) + max(0, left - base),
                              int(starts[doc]) + min(int(sizes[doc]), right - base)))
        return Coverage(spans)

    def documents(spans):
        result = set()
        for left, right in spans:
            a = int(np.searchsorted(ends, left, side="right"))
            b = int(np.searchsorted(ends, right - 1, side="right"))
            result.update(range(a, b + 1))
        return result

    def capture_start(stream):
        # Execute the real _sample() RNG, lookup and cursor isolation, stopping
        # only before token I/O. No model forward or training state is touched.
        assert stream is not source
        file_base = int(sampler.file_ends[stream.file_idx - 1]) if stream.file_idx else 0
        row_base = int(sampler.ends[stream.file_idx][stream.row_idx - 1]) if stream.row_idx else 0
        yield file_base + row_base + stream.offset

    totals = {str(length): dict(samples=0, source_positions=0, equal_reference_start_pairs=0,
                               samples_overlapping_reference_positions=0,
                               positions_overlapping_reference=0,
                               samples_sharing_reference_document=0) for length in lengths}
    previous_training = dict(source_positions=0, overlapping_positions=0, overlapping_samples=0, samples=0)
    seed_set = set()
    event_rows = []
    latest_history = training_coverage(0, updates[-1]) if updates[-1] else Coverage([])
    latest_history_counts = dict(source_positions=0, overlapping_positions=0)
    with patch.object(PackedDocumentDataset, "__iter__", capture_start):
        for update in updates:
            batches = {}
            for length in [reference, *lengths]:
                seeds = [stream_seed(fixed["seq_grad.seed"], update, length, i) for i in range(n)]
                assert not seed_set.intersection(seeds) and len(set(seeds)) == n
                seed_set.update(seeds)
                batches[length] = [int(sampler._sample(length, seed)) for seed in seeds]
            ref_spans = [span for start in batches[reference] for span in pieces(start, reference + 1, population)]
            reference_coverage = Coverage(ref_spans)
            ref_docs = documents(ref_spans)
            ref_starts = set(batches[reference])
            previous = training_coverage(update - 1, update) if update else Coverage([])
            row = dict(update=update, candidate_overlapping_positions=0, candidate_source_positions=0)
            for length, batch_starts in batches.items():
                for start in batch_starts:
                    spans = pieces(start, length + 1, population)
                    if length != reference:
                        overlap = reference_coverage.overlap(spans)
                        total = totals[str(length)]
                        total["samples"] += 1
                        total["source_positions"] += length + 1
                        total["equal_reference_start_pairs"] += batches[reference].count(start) if start in ref_starts else 0
                        total["samples_overlapping_reference_positions"] += int(overlap > 0)
                        total["positions_overlapping_reference"] += overlap
                        total["samples_sharing_reference_document"] += int(bool(documents(spans) & ref_docs))
                        row["candidate_overlapping_positions"] += overlap
                        row["candidate_source_positions"] += length + 1
                    if update:
                        overlap = previous.overlap(spans)
                        previous_training["source_positions"] += length + 1
                        previous_training["samples"] += 1
                        previous_training["overlapping_positions"] += overlap
                        previous_training["overlapping_samples"] += int(overlap > 0)
                    if update == updates[-1]:
                        latest_history_counts["source_positions"] += length + 1
                        latest_history_counts["overlapping_positions"] += latest_history.overlap(spans)
            event_rows.append(row)
    assert source.state_dict() == original_state, "Probe replay changed the training source state"
    # Validate interval arithmetic against a direct small-set calculation.
    check = Coverage([(1, 5), (4, 8), (10, 12)])
    for left in range(15):
        for right in range(left + 1, 16):
            assert check.overlap([(left, right)]) == len(set(range(left, right)) & (set(range(1, 8)) | {10, 11}))
    sampled = sum(item["source_positions"] for item in totals.values())
    overlapping = sum(item["positions_overlapping_reference"] for item in totals.values())
    report = dict(
        source_snapshot=str(snapshot), wandb_id=stage["wandb_id"], observed_updates=updates,
        events=len(updates), population_packed_tokens=population, distinct_sample_seeds=len(seed_set),
        probe_source_state_unchanged=True, training_shuffled_each_epoch=False,
        protocol=sampler.identity, footprint="Input source positions plus one label-lookahead position; includes ignored boundary positions. Counts identify corpus positions, not token-ID equality or semantic duplicates.",
        candidate_reference=dict(source_positions=sampled, overlapping_positions=overlapping,
                                 fraction=overlapping / sampled,
                                 expected_fraction_under_independent_uniform_draws=-math.expm1(n * math.log1p(-(reference + 1) / population)),
                                 by_length=totals),
        probe_vs_previous_training_update=previous_training,
        probe_vs_training_history_at_last_event=latest_history_counts,
        event_rows=event_rows,
        source_sha256={name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest() for name in
                       ["torchtitan_dynamic_seq_len/seq_grad/sampler.py", "torchtitan_dynamic_seq_len/data/packed_documents.py", "torchtitan_dynamic_seq_len/data/long_documents.py"]},
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"events": len(updates), "candidate_reference_fraction": overlapping / sampled,
                      "probe_vs_previous_training_update": previous_training,
                      "probe_vs_training_history_at_last_event": latest_history_counts}, indent=2))


if __name__ == "__main__":
    main()
