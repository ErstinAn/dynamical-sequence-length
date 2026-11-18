"""Compare actual Llama attention and gradients with independent documents."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torchtitan_dynamic_seq_len.config_registry import _model
from torchtitan_dynamic_seq_len.loss_grad_seq.attention import select_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(19)
    model = _model("flex", dim=64, num_layers=2, num_heads=4, num_kv_heads=2,
                   vocab_size=128).build().to("cuda")
    model.init_states(buffer_device=torch.device("cuda"))
    model.eval()
    tokens = torch.randint(0, 128, (1, 32), device="cuda")
    labels = torch.randint(0, 128, (1, 32), device="cuda")
    positions = torch.arange(16, device="cuda").repeat(2)[None]

    def forward(x, pos, masked):
        return model(x, positions=pos, attention_masks=select_attention(model, pos, masked))

    def gradient(loss):
        model.zero_grad(set_to_none=True)
        loss.backward()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).detach().clone()

    packed = forward(tokens, positions, True)
    loss = F.cross_entropy(packed.flatten(0, 1), labels.flatten())
    packed_grad = gradient(loss)
    separate = torch.cat([forward(tokens[:, i:i+16], positions[:, i:i+16], True) for i in (0, 16)], dim=1)
    separate_grad = gradient(F.cross_entropy(separate.flatten(0, 1), labels.flatten()))
    torch.testing.assert_close(packed.detach(), separate.detach(), rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(packed_grad, separate_grad, rtol=3e-4, atol=2e-5)
    changed = tokens.clone()
    changed[:, :16] = (changed[:, :16] + 1) % 128
    masked_b = forward(changed, positions, True)
    torch.testing.assert_close(packed.detach()[:, 16:], masked_b.detach()[:, 16:], rtol=2e-4, atol=2e-5)
    grad_b = gradient(F.cross_entropy(masked_b[:, 16:].flatten(0, 1), labels[:, 16:].flatten()))
    original_b = forward(tokens, positions, True)
    original_grad_b = gradient(F.cross_entropy(original_b[:, 16:].flatten(0, 1), labels[:, 16:].flatten()))
    torch.testing.assert_close(grad_b, original_grad_b, rtol=3e-4, atol=2e-5)
    unmasked = forward(tokens, positions, False)
    from torchtitan.models.common.attention import ScaledDotProductAttention
    reference_config = copy.deepcopy(model.config)
    for layer in reference_config.layers:
        layer.attention.inner_attention = ScaledDotProductAttention.Config()
    reference = reference_config.build().to("cuda")
    reference.init_states(buffer_device=torch.device("cuda"))
    reference.load_state_dict(model.state_dict())
    reference.eval()
    causal_reference = reference(tokens, positions=positions)
    torch.testing.assert_close(unmasked.detach(), causal_reference.detach(), rtol=3e-4, atol=2e-5)
    changed_unmasked = forward(changed, positions, False)
    effect = float((unmasked[:, 16:] - changed_unmasked[:, 16:]).abs().max())
    assert effect > 1e-6, effect
    # Future edits cannot affect the preceding predictions in either mode.
    future = tokens.clone()
    future[:, 24:] = (future[:, 24:] + 1) % 128
    for masked in (True, False):
        a, b = forward(tokens, positions, masked), forward(future, positions, masked)
        torch.testing.assert_close(a[:, :24], b[:, :24], rtol=2e-4, atol=2e-5)
    result = dict(status="passed", gpu=torch.cuda.get_device_name(),
                  masked_logit_max_error=float((packed.detach()-separate.detach()).abs().max()),
                  masked_gradient_max_error=float((packed_grad-separate_grad).abs().max()),
                  unmasked_cross_document_effect=effect,
                  logit_rtol=2e-4, gradient_rtol=3e-4, atol=2e-5)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
