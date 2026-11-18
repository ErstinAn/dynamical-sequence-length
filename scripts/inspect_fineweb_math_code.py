"""Inspect targeted math/code candidates; do not estimate population statistics."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

from tokenizers import Tokenizer

from prepare_fineweb_edu_examples import ROOT, PROFILE, read_rows


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run in a CPU allocation.")
    source = ROOT / "outputs/fineweb_edu_profile/document_examples"
    output = ROOT / "outputs/fineweb_edu_profile/math_code_inspection"
    output.mkdir(exist_ok=True)
    metadata = json.loads((source / "candidate_metadata.json").read_text())
    manifest = json.loads((PROFILE / "source_manifest.json").read_text())
    patterns = ["clintbellanger.net/articles/isometric_math", "mathematical-algorithms-asymmetric",
                "dreamincode.net/forums/topic/54399", "programming-arcgis-python-beginners",
                "lecture-12-graphs-and-networks", "visual-programming-gis-applications",
                "kvspgtcsamit.wordpress.com/cloud-computing-2", "domain-of-square-root-function.html",
                "getting-started-with-sas-programming-using-sas-studio"]
    selected = []
    for pattern in patterns:
        matches = [d for d in metadata if pattern in d["url"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one candidate for {pattern}, found {len(matches)}")
        selected.extend(matches)
    entries = {e["path"]: e for e in manifest["files"]}
    groups = {}
    for d in selected:
        groups.setdefault(d["path"], []).append(d)
    def fetch(item):
        path, samples = item
        return read_rows(entries[path], samples, manifest,
                         ["text", "id", "url", "dump", "token_count", "score", "int_score"])
    with ThreadPoolExecutor(max_workers=4) as executor:
        documents = [d for batch in executor.map(fetch, groups.items()) for d in batch]
    tokenizers = {name: Tokenizer.from_file(str(ROOT / "assets/hf" / folder / "tokenizer.json"))
                  for name, folder in [("llama2", "llama-tokenizer"), ("llama3", "llama3-tokenizer")]}
    for t in tokenizers.values():
        t.no_truncation()
        t.no_padding()
    for d in documents:
        d["text_sha256"] = hashlib.sha256(d["text"].encode()).hexdigest()
        d["tokenizer_lengths_without_special_tokens"] = {
            name: len(t.encode(d["text"], add_special_tokens=False).ids) for name, t in tokenizers.items()}
    documents.sort(key=lambda d: d["source_tokens"])
    result = {"dataset": {k: manifest[k] for k in ["repo_id", "revision", "config"]},
              "job_id": os.environ["SLURM_JOB_ID"],
              "method": "Targeted URL candidates from existing narrow length bands; not representative.",
              "documents": documents}
    (output / "documents.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    overview = {**result, "documents": [{k:v for k,v in d.items() if k != "text"} for d in documents]}
    (output / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "complete", "documents": len(documents)}), flush=True)


if __name__ == "__main__":
    main()
