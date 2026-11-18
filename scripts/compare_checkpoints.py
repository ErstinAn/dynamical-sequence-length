"""Compare complete checkpoint values one tensor at a time on the CPU."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def compare(left, right):
    readers = [dcp.FileSystemReader(str(path)) for path in (left, right)]
    metadata = [reader.read_metadata().state_dict_metadata for reader in readers]
    if set(metadata[0]) != set(metadata[1]):
        raise ValueError("Checkpoint state keys differ.")
    tensors = scalars = 0
    differences = []
    for key in sorted(metadata[0]):
        specs = [item[key] for item in metadata]
        tensor = isinstance(specs[0], TensorStorageMetadata)
        if tensor != isinstance(specs[1], TensorStorageMetadata):
            raise ValueError(f"Checkpoint state types differ: {key}")
        if tensor and (specs[0].size != specs[1].size or specs[0].properties.dtype != specs[1].properties.dtype):
            raise ValueError(f"Checkpoint tensor shape or dtype differs: {key}")
        values = []
        for reader, spec in zip(readers, specs):
            value = torch.empty(spec.size, dtype=spec.properties.dtype) if tensor else None
            state = {key: value}
            dcp.load(state, storage_reader=reader,
                     planner=dcp.DefaultLoadPlanner(allow_partial_load=True))
            values.append(state[key])
        if tensor:
            tensors += 1
            equal = torch.equal(*values)
            if not equal:
                delta = (values[0].float() - values[1].float()).abs()
                differences.append({"key": key, "max_absolute_error": float(delta.max())})
        else:
            scalars += 1
            # Dataloader state can contain nested tensors and NumPy arrays.
            def same(a, b):
                if isinstance(a, torch.Tensor):
                    return isinstance(b, torch.Tensor) and torch.equal(a, b)
                if isinstance(a, dict):
                    return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
                if isinstance(a, (list, tuple)):
                    return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
                if hasattr(a, "shape"):
                    import numpy as np
                    return np.array_equal(a, b)
                return a == b
            if not same(*values):
                differences.append({"key": key, "non_tensor_state_differs": True})
        del values
    return {"left": str(left), "right": str(right), "tensors_checked": tensors,
            "non_tensor_values_checked": scalars, "exact_match": not differences,
            "differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
