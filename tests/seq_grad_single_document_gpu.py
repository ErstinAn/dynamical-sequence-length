"""Check monitor-only causal attention against SDPA and preserve training masks."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan_dynamic_seq_len.config_registry import _model
from torchtitan_dynamic_seq_len.seq_grad.trainer import SeqGradTrainer


def attention_trainer(model):
    trainer = object.__new__(SeqGradTrainer)
    trainer.config = SimpleNamespace(seq_grad=SimpleNamespace(parent_population="single_document"),
                                     parallelism=SimpleNamespace(spmd_backend="default"))
    trainer._probe_active = True
    trainer.ntokens_seen = 0
    trainer.model_config = model.config
    trainer.model_parts = [model]
    trainer.parallel_dims = SimpleNamespace(cp_enabled=False)
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(19)
    model = _model("flex", dim=64, num_layers=2, num_heads=4, num_kv_heads=2, vocab_size=128).build().to("cuda")
    model.init_states(buffer_device=torch.device("cuda"))
    reference_config = copy.deepcopy(model.config)
    for layer in reference_config.layers:
        layer.attention.inner_attention = ScaledDotProductAttention.Config()
    reference = reference_config.build().to("cuda")
    reference.init_states(buffer_device=torch.device("cuda"))
    reference.load_state_dict(model.state_dict())
    trainer = attention_trainer(model)
    tokens = torch.randint(0, 128, (2, 65), device="cuda")
    errors = []
    for start, length in ((0, 64), (13, 16), (31, 8)):
        x, y = tokens[:, start:start+length], tokens[:, start+1:start+length+1]
        positions = torch.arange(length, device="cuda").expand(2, length)
        x, y, kwargs = trainer.post_dataloading_process({"input": x, "positions": positions}, y)
        actual, expected = model(x, **kwargs), reference(x, positions=positions)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-5)
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        F.cross_entropy(actual.flatten(0, 1), y.flatten()).backward()
        F.cross_entropy(expected.flatten(0, 1), y.flatten()).backward()
        gradients = []
        for a, b in zip(model.parameters(), reference.parameters(), strict=True):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=2e-5)
                gradients.append(float((a.grad-b.grad).abs().max()))
        errors.append(dict(length=length, logit_max_error=float((actual-expected).detach().abs().max()),
                           gradient_max_error=max(gradients)))
        changed = x.clone()
        changed[:, -2:] = (changed[:, -2:] + 1) % 128
        torch.testing.assert_close(model(changed, **kwargs)[:, :-2], actual[:, :-2], rtol=3e-4, atol=2e-5)
    # When ordinary training resumes, native packed-document blocking must remain.
    trainer._probe_active = False
    positions = torch.arange(16, device="cuda").repeat(2).expand(2, 32)
    x, y, kwargs = trainer.post_dataloading_process({"input": tokens[:, :32], "positions": positions}, tokens[:, 1:33])
    original = model(x, **kwargs)
    changed = x.clone()
    changed[:, :16] = (changed[:, :16] + 1) % 128
    torch.testing.assert_close(model(changed, **kwargs)[:, 16:], original[:, 16:], rtol=3e-4, atol=2e-5)
    report = dict(passed=True, gpu=torch.cuda.get_device_name(), causal_logits_and_gradients=errors,
                  future_tokens_blocked=True, training_document_mask_preserved=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
