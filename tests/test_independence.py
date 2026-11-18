"""Ensure project configs work when all original research packages are blocked."""

import subprocess
import sys
import unittest
from pathlib import Path


class ProjectIndependenceTest(unittest.TestCase):
    def test_configs_and_optimizer_steps_without_original_packages(self):
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                r"""
import importlib.abc
import os
from pathlib import Path
import sys

project = Path(sys.argv[1])
sys.path.insert(0, str(project))
blocked = {"rotation_optimizers", "torchtitan_rotation_optimizers", "matrix_optimizers_distributor"}

class BlockOriginalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError(f"External research package imported: {fullname}")

sys.meta_path.insert(0, BlockOriginalPackages())
os.environ["FINEWEB100_VALIDATION_DIR"] = "/unavailable/original-project-data"
os.environ.pop("DYNAMIC_SEQ_LEN_VALIDATION_DIR", None)

import torch
from torchtitan_dynamic_seq_len import config_registry
from torchtitan_dynamic_seq_len.optimizer import ExperimentOptimizers
from torchtitan_dynamic_seq_len.data.fineweb100 import validation_directory
from torchtitan_dynamic_seq_len.data.fixed_validation import resolve_artifact_path
from torchtitan_dynamic_seq_len.data.asset_paths import resolve_archived_asset

expected_data = resolve_archived_asset(project / "assets/data/fineweb100_validation")
assert validation_directory() == expected_data
assert resolve_artifact_path() == expected_data.resolve()
for name, expected_optimizer in (
    ("llama3_100m_llama2_adamw", torch.optim.AdamW),
    ("llama3_100m_llama2_muon", torch.optim.Muon),
    ("llama3_100m_llama2_adamw", torch.optim.AdamW),
):
    config = getattr(config_registry, name)()
    assets = Path(os.environ.get("DYNAMIC_SEQ_LEN_ASSETS", project / "assets"))
    assert Path(config.hf_assets_path) == assets / "hf/llama-tokenizer"
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    container = ExperimentOptimizers(config.optimizer, model_parts=[model])
    assert type(container.optimizers[0]) is expected_optimizer
    assert type(container.optimizers[1]) is torch.optim.AdamW
    original = model[0].weight.detach().clone()
    model(torch.randn(2, 4)).square().sum().backward()
    for optimizer in container.optimizers:
        optimizer.step()
    assert not torch.equal(original, model[0].weight)

for name, module in tuple(sys.modules.items()):
    assert name.split(".")[0] not in blocked, name
    if name.startswith("torchtitan_dynamic_seq_len") and getattr(module, "__file__", None):
        assert Path(module.__file__).resolve().is_relative_to(project), name
print("Isolated configs, local asset paths, and AdamW/Muon steps passed")
""",
                str(project),
            ],
            cwd=project,
            text=True,
            capture_output=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
