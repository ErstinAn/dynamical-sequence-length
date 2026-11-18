"""Make a tiny real-text artifact to check the GPU integration before bulk extraction."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from build_fineweb_long import (
    ROOT,
    TOKENIZER,
    candidate_schema,
    normalized_hash,
    url_group,
    verify_stage,
)
from profile_fineweb_edu_lengths import sha256, write_json
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    (root / "tokens").mkdir(exist_ok=True)
    documents = json.loads((ROOT / "outputs/fineweb_edu_profile/document_examples/agency_documents.json").read_text())
    tokenizer = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
    provenance = json.loads((TOKENIZER / "tokenizer_source.json").read_text())
    files, splits = [], {}
    for split, source_length in [("train", 32772), ("validation", 32585)]:
        doc = next(d for d in documents if d["source_tokens"] == source_length)
        text = doc["text"]
        ids = np.array(tokenizer.encode(text, add_special_tokens=False).ids, dtype="<u4")
        assert len(ids) >= 32768
        entry = {"id": doc["id"], "text": text, "url": doc["url"],
                 "source_tokens": doc["source_tokens"], "llama3_tokens": len(ids),
                 "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                 "normalized_sha256": normalized_hash(text), "url_group": url_group(doc["url"], doc["id"]),
                 "source_path": doc["path"], "source_row": doc["row"], "dump": doc["dump"], "edu_score": doc["score"]}
        base = f"{split}-00000"
        pq.write_table(pa.Table.from_pylist([entry], schema=pa.schema([f for f in candidate_schema() if f.name != "input_ids"])), root / "data" / f"{base}.parquet")
        ids.tofile(root / "tokens" / f"{base}.bin")
        np.save(root / "tokens" / f"{base}.index.npy", np.array([[0, len(ids)]], dtype=np.uint64))
        result = {"split": split, "documents": 1, "tokens": len(ids)}
        for kind, name in [("text", f"data/{base}.parquet"), ("token_ids", f"tokens/{base}.bin"), ("index", f"tokens/{base}.index.npy")]:
            result[kind] = {"path": name, "sha256": sha256(root / name), "bytes": (root / name).stat().st_size}
        files.append(result)
        splits[split] = {"documents": 1, "tokens": len(ids)}
    manifest = {"schema_version": 1, "complete": True, "tokenizer": provenance,
                "tokenizer_sha256": provenance["files_sha256"]["tokenizer.json"], "files": files,
                "selection": {"minimum_llama3_tokens": 32768, "splits": splits},
                "purpose": "Tiny real-text integration fixture; not the final 30B corpus"}
    write_json(root / "manifest.json", manifest)
    verify_stage(SimpleNamespace(output=root))
    config = yaml.safe_load((ROOT / "legacy/configs/examples/llama3_1b_long_documents_smoke.yaml").read_text())
    config["name"] = "llama3-1b-long-documents-pilot"
    config["fixed"]["dataloader.dataset_path"] = str(root)
    config["fixed"]["validator.dataloader.dataset_path"] = str(root)
    (root / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
