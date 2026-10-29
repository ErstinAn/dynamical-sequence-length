"""Bind every server's FineWeb evaluation to a versioned reference identity."""

import json
from pathlib import Path

REFERENCE_PATH = Path(__file__).parent / "manifests/fineweb_100bt_validation_reference.json"
LLAMA2_REFERENCE_PATH = (
    Path(__file__).parent / "manifests/fineweb_100bt_validation_llama2_1024_reference.json"
)
LLAMA2_LENGTH_REFERENCE_PATHS = tuple(
    Path(__file__).parent / f"manifests/fineweb_100bt_validation_llama2_{length}_reference.json"
    for length in (512, 2048, 4096)
)
IDENTITY_FIELDS = (
    "schema_version",
    "token_budget",
    "sequence_length",
    "num_sequences",
    "dtype",
    "arrays",
    "tokenizer_files",
    "source",
    "source_manifest_sha256",
    "evaluation_documents",
    "packing",
)


def validation_identity(manifest: dict) -> dict:
    return {name: manifest[name] for name in IDENTITY_FIELDS}


def verify_validation_identity(manifest: dict) -> None:
    actual = validation_identity(manifest)
    # Each artifact has its own exact, versioned identity. Selecting a new
    # tokenizer never relaxes source, document-exclusion, or array checksums.
    references = [
        json.loads(path.read_text())["artifact_identity"]
        for path in (REFERENCE_PATH, LLAMA2_REFERENCE_PATH, *LLAMA2_LENGTH_REFERENCE_PATHS)
    ]
    mismatches = [
        [name for name in IDENTITY_FIELDS if actual[name] != reference[name]]
        for reference in references
    ]
    if any(not differences for differences in mismatches):
        return
    differences = min(mismatches, key=len)
    raise ValueError(
        "FineWeb 100BT validation differs from the versioned reference: "
        + ", ".join(differences)
        + "; copy/rebuild the approved artifact before training."
    )
