"""Reconstruct the submitted packed run without changing training or W&B state."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/packed_loss_diagnosis_20261116"
DATA = Path("/projects/jason_optai/kangan/datasets/fineweb-edu-long-30b")
RUN = Path("/projects/jason_optai/kangan/experiments/longseq_packed/runs/"
           "packed-loss22256-grad2048-8192-bs-increasing-seq-decreasing/"
           "650b1e8f4a114b40bf7f0724cd336f64")


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")


def indexes():
    arrays = np.load(OUT / "data_indexes.npz")
    n = len(arrays["lengths"])
    ranks = []
    for r in range(4):
        selected = np.arange(r, n, 4)
        ranks.append(dict(cum=np.r_[0, np.cumsum(arrays["lengths"][selected], dtype=np.int64)],
                          selected=selected, files=arrays["files"][selected],
                          rows=arrays["rows"][selected]))
    return arrays, ranks


def cursor(rank, position):
    c = rank["cum"]
    epoch, position = divmod(int(position), int(c[-1]))
    idx = int(np.searchsorted(c, position, side="right") - 1)
    return dict(file_idx=int(rank["files"][idx]), row_idx=int(rank["rows"][idx]),
                offset=int(position - c[idx]), epoch=epoch)


def replay():
    import pyarrow.parquet as pq

    h = json.loads((OUT / "history.json").read_text())
    a, ranks = indexes()
    # Determine the exact candidate-pool boundary from the original metadata.
    meta = pq.read_table(DATA / "data/train-00179.parquet",
                         columns=["source_tokens", "llama3_tokens"], use_threads=False)
    short = meta["source_tokens"].to_numpy() < 16384
    cut_row = int(np.flatnonzero(short)[0])
    assert not short[:cut_row].any() and short[cut_row:].all()
    boundary = int(np.flatnonzero((a["files"] == 179) & (a["rows"] == cut_row))[0])
    blocks = []
    for name, sl in [("original_pool", slice(0, boundary)), ("supplement_12k", slice(boundary, None))]:
        lens = a["lengths"][sl] - 2
        blocks.append(dict(name=name, documents=len(lens), raw_tokens=int(lens.sum()),
                           mean_length=float(lens.mean()), min_length=int(lens.min()),
                           max_length=int(lens.max()), median_length=float(np.median(lens))))
    events, rows, mismatches = [], [], []
    full_starts = []
    boundaries = []
    for r, rank in enumerate(ranks):
        total = int(rank["cum"][-1])
        full_starts.append(np.concatenate([rank["cum"][:-1] + e * total for e in range(4)]))
        n = int(np.searchsorted(rank["selected"], boundary))
        boundaries.append(int(rank["cum"][n]))
        for event, position in [("enter_supplement", boundaries[-1]), ("epoch_1_start", total),
                                ("epoch_2_start", total * 2)]:
            step = next(x["_step"] for x in h if int(x["loss_grad/parent_tokens_seen"]) // 4 > position)
            events.append(dict(rank=r, event=event, step=step, rank_stream_position=position))
    for row in h:
        b = row["loss_grad/global_batch_size"]
        length = row["loss_grad/loss_seq_length"]
        after = int(row["loss_grad/parent_tokens_seen"]) // 4
        before = after - b * length // 4
        fragments = ignored = supplement_tokens = 0
        for r, rank in enumerate(ranks):
            starts = full_starts[r]
            ws = before + np.arange(b // 4, dtype=np.int64) * length
            we = ws + length
            ignored += int((np.searchsorted(starts, we, side="right") -
                            np.searchsorted(starts, ws, side="right")).sum())
            fragments += b // 4 + int((np.searchsorted(starts, we, side="left") -
                                       np.searchsorted(starts, ws, side="right")).sum())
            total = int(rank["cum"][-1])
            for ep in range(before // total, after // total + 1):
                supplement_tokens += max(0, min(after, (ep + 1) * total) -
                                         max(before, ep * total + boundaries[r]))
        expected = {"packing/parent_fragments": fragments,
                    "packing/ignored_parent_labels": ignored,
                    "loss_grad/parent_valid_tokens": b * length - ignored,
                    "loss_grad/parent_tokens": b * length,
                    "loss_grad/grad_tokens": b * row["loss_grad/grad_seq_length"]}
        for k, v in expected.items():
            if v != row[k]:
                mismatches.append(dict(step=row["_step"], key=k, expected=v, actual=row[k]))
        rows.append({"step": row["_step"], "long_loss": row["loss_metrics/long_avg_loss"],
                     "short_loss": row["loss_metrics/short_seq_loss"],
                     "validation_loss": row.get("validation_metrics/loss"),
                     "parent_tokens_seen": after * 4, "global_batch": b,
                     "grad_length": row["loss_grad/grad_seq_length"],
                     "supplement_fraction": supplement_tokens / (b * length),
                     "fragments": fragments, "ignored_labels": ignored,
                     "mean_fragment_positions": b * length / fragments,
                     "grad_norm": row["grad_norm"], "lr": row["lr/AdamW"]})
    with (OUT / "replay.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result = dict(candidate_boundary=dict(file=179, row=cut_row, global_document=boundary),
                  blocks=blocks, events=events, steps=len(h),
                  invariant_comparisons=len(h) * 5, mismatches=mismatches,
                  final_cursors=[cursor(r, int(h[-1]["loss_grad/parent_tokens_seen"]) // 4) for r in ranks])
    write_json("replay.json", result)
    print(json.dumps(result, indent=2), flush=True)


def checkpoint_state():
    import torch
    import torch.distributed.checkpoint as dcp

    p = RUN / "checkpoint/step-3840"
    m = dcp.FileSystemReader(p).read_metadata()
    state = {}
    for k, v in m.storage_data.items():
        if k.fqn.startswith(("train_state.", "dataloader.", "lr_scheduler.")):
            with (p / v.relative_path).open("rb") as f:
                f.seek(v.offset)
                value = torch.load(io.BytesIO(f.read(v.length)), weights_only=False)
            # ParallelAwareDataloader stores a pickled dictionary as bytes.
            if isinstance(value, bytes):
                import pickle
                value = pickle.loads(value)
            state[k.fqn] = value
    write_json("checkpoint_state.json", state)
    print(json.dumps(state, indent=2), flush=True)


def data_quality():
    import hashlib
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(ROOT / "assets/hf/llama3-tokenizer/tokenizer.json"))
    selected = [(0, 0), (178, 100), (179, 1885), (179, 1886), (180, 100), (234, 10)]
    checks = []
    for shard, row in selected:
        file = pq.ParquetFile(DATA / f"data/train-{shard:05d}.parquet")
        offset = 0
        for group in range(file.metadata.num_row_groups):
            size = file.metadata.row_group(group).num_rows
            if offset <= row < offset + size:
                doc = file.read_row_group(group, use_threads=False).slice(row - offset, 1).to_pylist()[0]
                break
            offset += size
        index = np.load(DATA / f"tokens/train-{shard:05d}.index.npy", mmap_mode="r")
        start, size = map(int, index[row])
        tokens = np.memmap(DATA / f"tokens/train-{shard:05d}.bin", mode="r", dtype="<u4")[start:start + size]
        encoded = np.asarray(tokenizer.encode(doc["text"], add_special_tokens=False).ids)
        check = dict(shard=shard, row=row, source_tokens=doc["source_tokens"],
                     llama3_tokens=doc["llama3_tokens"], id=doc["id"], url=doc["url"],
                     source_path=doc["source_path"], edu_score=doc["edu_score"],
                     text_sha_matches=hashlib.sha256(doc["text"].encode()).hexdigest() == doc["text_sha256"],
                     token_cache_matches=bool(np.array_equal(tokens, encoded)),
                     vocab_valid=bool(np.all(tokens < 128256)),
                     text_excerpt=doc["text"][:300])
        assert check["token_cache_matches"] and check["text_sha_matches"] and check["vocab_valid"]
        checks.append(check)
    artifact = ROOT / "assets/data/longseq_validation_llama3_packed_22256"
    provenance = json.loads((artifact / "documents.json").read_text())
    docs = {(f["file"], f["row"]) for p in provenance for f in p["fragments"]}
    valfile = pq.read_table(DATA / "data/validation-00000.parquet",
                            columns=["source_tokens", "llama3_tokens", "dump", "url_group"], use_threads=False)
    vals = valfile.to_pylist()
    used = [vals[row] for file, row in docs]
    assert {file for file, row in docs} == {"data/validation-00000.parquet"}
    val_result = dict(unique_documents=len(docs), row_min=min(row for _, row in docs),
                      row_max=max(row for _, row in docs),
                      original_pool_documents=sum(d["source_tokens"] >= 16384 for d in used),
                      supplement_documents=sum(d["source_tokens"] < 16384 for d in used),
                      full_validation_documents=len(vals),
                      full_validation_supplement_documents=sum(d["source_tokens"] < 16384 for d in vals),
                      mean_document_length=float(np.mean([d["llama3_tokens"] for d in used])),
                      source_dumps=sorted({d["dump"] for d in used}))
    write_json("data_quality.json", dict(token_cache_samples=checks, validation=val_result))
    print(json.dumps(val_result, indent=2), flush=True)


def crop_and_schedule_counts():
    import hashlib
    import sys
    sys.path.insert(0, str(ROOT))
    import torch
    from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig
    from torchtitan_dynamic_seq_len.schedulers.batch_size import BatchSizeSchedulerConfig
    from torchtitan_dynamic_seq_len.loss_grad_seq.batching import plan_batches

    run = json.loads((OUT / "run.json").read_text())
    h = json.loads((OUT / "history.json").read_text())
    cfg = run["config"]
    _, ranks = indexes()
    lengths = SeqLenSchedulerConfig(**cfg["grad_seq_len_scheduler"]).build(3840)
    batches = BatchSizeSchedulerConfig(**cfg["batch_size_scheduler"]).build(3840)
    full_starts = [np.concatenate([r["cum"][:-1] + e * r["cum"][-1] for e in range(4)]) for r in ranks]
    before = grad_seen = valid_seen = comparisons = 0
    mismatches = []
    for row in h:
        step = row["_step"]
        b, g = batches.value_for_step(step), lengths.length_for_step(step)
        plan = plan_batches(loss_length=22256, grad_length=g, global_batch=b, dp=4,
                            loss_limit=2, grad_limit=16, loss_token_budget=44512,
                            grad_token_budget=32768)
        ignored = 0
        for r in range(4):
            key = f"seq-grad-v1:1:{step}:{r}:loss-grad-train-v1".encode()
            generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % 2**63)
            offsets = torch.randint(22256 - g + 1, (b // 4,), generator=generator).numpy()
            starts = before + np.arange(b // 4) * 22256 + offsets
            ends = starts + g
            ignored += int((np.searchsorted(full_starts[r], ends, side="right") -
                            np.searchsorted(full_starts[r], starts, side="right")).sum())
        before += b * 22256 // 4
        grad_seen += b * g
        valid_seen += b * g - ignored
        expected = {"loss_grad/global_batch_size": b, "loss_grad/grad_seq_length": g,
                    "loss_grad/loss_microbatch_size": plan.loss_microbatch,
                    "loss_grad/grad_microbatch_size": plan.grad_microbatch,
                    "loss_grad/loss_forward_batches": plan.loss_batches,
                    "loss_grad/gradient_accumulation_steps": plan.grad_batches,
                    "valid_tokens_per_step": b * g - ignored,
                    "n_tokens_seen": grad_seen, "n_valid_tokens_seen": valid_seen,
                    "loss_grad/parent_tokens_seen": before * 4}
        for k, v in expected.items():
            comparisons += 1
            if row[k] != v:
                mismatches.append(dict(step=step, key=k, expected=v, actual=row[k]))
    result = dict(comparisons=comparisons, mismatches=mismatches,
                  grad_tokens_seen=grad_seen, grad_valid_tokens_seen=valid_seen)
    write_json("crop_schedule_replay.json", result)
    print(json.dumps(result, indent=2), flush=True)


def gpu():
    """Fixed checkpoint, fixed samples, paired context and model implementation checks."""
    import sys
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.nn import functional as F
    from torchtitan_dynamic_seq_len.config_registry import _model
    from torchtitan_dynamic_seq_len.data.packed_documents import PackedDocumentDataset
    from torchtitan_dynamic_seq_len.data.seq_grad_windows import crop_batches
    from torchtitan.models.common.attention import create_block_mask

    torch.set_num_threads(4)
    device = torch.device("cuda")
    model = _model("flex", dim=512, num_layers=12, num_heads=8, num_kv_heads=4,
                   vocab_size=128256).build()
    model.init_states(buffer_device=torch.device("cpu"))
    state = model.state_dict()
    dcp.load(state, checkpoint_id=str(RUN / "checkpoint/step-3840"))
    model.load_state_dict(state, strict=True)
    del state
    # bfloat16() preserves the complex RoPE buffers; .to(dtype=...) does not.
    model = model.to(device=device).bfloat16().eval()
    model._skip_lm_head = True
    h = {x["_step"]: x for x in json.loads((OUT / "history.json").read_text())}
    _, ranks = indexes()
    results = []

    @torch.no_grad()
    def evaluate(inputs, labels, *, sliding=None):
        x = inputs["input"].to(device)
        pos = inputs["positions"].to(device)
        y = labels.to(device)
        if sliding is None:
            masks = model.get_attention_masks(positions=pos)
        else:
            docs = (pos == 0).cumsum(1)
            def mask(b, head, q, kv):
                return (q >= kv) & (q - kv < sliding) & (docs[b, q] == docs[b, kv])
            masks = create_block_mask(mask, B=x.shape[0], H=None, Q_LEN=x.shape[1],
                                      KV_LEN=x.shape[1], device=device, _compile=True)
        hidden = model(x, positions=pos, attention_masks=masks)
        losses = []
        for start in range(0, y.shape[1], 512):
            logits = model.lm_head(hidden[:, start:start + 512])
            losses.append(F.cross_entropy(logits.float().flatten(0, 1),
                                          y[:, start:start + 512].flatten(),
                                          reduction="none", ignore_index=-100).reshape(y.shape[0], -1))
        losses = torch.cat(losses, dim=1)
        valid = y != -100
        result = dict(loss_sum=float(losses.double().sum()), valid=int(valid.sum()))
        result["loss"] = result["loss_sum"] / result["valid"]
        result["position_bins"] = []
        for lo, hi in [(0, 2048), (2048, 4096), (4096, 8192), (8192, 12288), (12288, 16384), (16384, 22256)]:
            keep = valid & (pos >= lo) & (pos < hi)
            n = int(keep.sum())
            result["position_bins"].append(dict(lo=lo, hi=hi, count=n,
                                                 loss=float(losses[keep].double().mean()) if n else None))
        return result

    # The checkpoint is held constant across all three regions and context treatments.
    for step in (2080, 2120, 2560):
        row = h[step]
        before = (int(row["loss_grad/parent_tokens_seen"]) - row["loss_grad/parent_tokens"]) // 4
        for rank in range(4):
            ds = PackedDocumentDataset(DATA, split="train", seq_len=22256,
                                       dp_rank=rank, dp_world_size=4, infinite=True)
            for k, v in cursor(ranks[rank], before).items():
                setattr(ds, k, v)
            it = iter(ds)
            parents = [next(it) for _ in range(8)]
            inputs = {k: torch.stack([x[k] for x, _ in parents]) for k in parents[0][0]}
            labels = torch.stack([y for _, y in parents])
            batches = [({k: v[i:i+2] for k, v in inputs.items()}, labels[i:i+2]) for i in range(0, 8, 2)]
            crops = crop_batches(batches, 4096, seed=1, step=step, rank=rank, stream="loss-grad-train-v1")
            for treatment in ("long", "crop4096", "sliding4096"):
                pieces = []
                for x, y in (crops if treatment == "crop4096" else batches):
                    pieces.append(evaluate(x, y, sliding=4096 if treatment == "sliding4096" else None))
                result = dict(sample_step=step, rank=rank, treatment=treatment,
                              loss_sum=sum(x["loss_sum"] for x in pieces),
                              valid=sum(x["valid"] for x in pieces), pieces=pieces)
                result["loss"] = result["loss_sum"] / result["valid"]
                results.append(result)
                write_json("gpu_ablation.json", dict(checkpoint=3840, gpu=torch.cuda.get_device_name(),
                                                      samples_per_region=32, results=results))
                print(json.dumps({k: v for k, v in result.items() if k != "pieces"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["replay", "checkpoint", "data", "counts", "gpu"])
    args = p.parse_args()
    {"replay": replay, "checkpoint": checkpoint_state, "data": data_quality,
     "counts": crop_and_schedule_counts, "gpu": gpu}[args.mode]()
