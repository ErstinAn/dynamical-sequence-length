"""Compare all tensors and scalar values in two training traces."""

import argparse
import json
import struct
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = torch.load(args.before, map_location="cpu", weights_only=False)
    after = torch.load(args.after, map_location="cpu", weights_only=False)
    num_tensors = 0

    def compare(left, right, path):
        nonlocal num_tensors
        if type(left) is not type(right):
            raise AssertionError(f"Different types at {path}")
        if isinstance(left, torch.Tensor):
            if (
                left.dtype != right.dtype
                or left.shape != right.shape
                or not torch.equal(
                    left.contiguous().reshape(-1).view(torch.uint8),
                    right.contiguous().reshape(-1).view(torch.uint8),
                )
            ):
                raise AssertionError(f"Different tensor values at {path}")
            num_tensors += 1
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                raise AssertionError(f"Different keys at {path}")
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                raise AssertionError(f"Different lengths at {path}")
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                compare(a, b, f"{path}.{index}")
        elif isinstance(left, float):
            if struct.pack("!d", left) != struct.pack("!d", right):
                raise AssertionError(f"Different float bytes at {path}")
        elif left != right:
            raise AssertionError(f"Different scalar values at {path}: {left} != {right}")

    compare(before, after, "trace")
    result = {
        "exact_match": True,
        "equal_tensors": num_tensors,
        "traces": {name: values["trace"] for name, values in after.items()},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exact match: {num_tensors} tensors and all scalar values.")


if __name__ == "__main__":
    main()
