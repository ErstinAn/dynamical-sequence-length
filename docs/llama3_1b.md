# Llama3 1B with the Llama2 tokenizer

This experiment variant reuses TorchTitan's `llama3_configs["1B"]` constructor.
The dependency revision is `737594746fda65a6d94dc9482ef07863a80c8588`.
The [upstream definition](https://github.com/pytorch/torchtitan/blob/737594746fda65a6d94dc9482ef07863a80c8588/torchtitan/models/llama3/__init__.py) supplies the complete Transformer backbone.

The adapter changes three vocabulary fields: model vocabulary, embedding entries, and output features.
All three become 32000. The loss uses the same vocabulary size.
The adapter keeps the upstream attention, feed-forward layers, initialization, RoPE, and weight tying.
Each call constructs a new configuration. It does not mutate the upstream registration.

| Property | Value |
| --- | ---: |
| Unique parameters | 1,038,682,112 |
| Layers | 16 |
| Hidden dimension | 2048 |
| Attention heads | 32 |
| KV heads | 8 |
| Feed-forward hidden dimension | 8192 |
| Vocabulary | 32000 |
| Embedding and output weights | Shared |

This is a vocabulary-adapted 1B model, not the unmodified Meta Llama 3.2 model.
Training starts from random initialization. Pretrained weights are not needed.
The upstream 128256-token configuration remains unchanged and has 1,235,814,400 unique parameters.
The existing 100M model and its configuration entries remain unchanged.

## Registered training entries

| Entry | Trainer behavior |
| --- | --- |
| `llama3_1b_llama2_adamw` | Fixed length with gradient variance monitoring |
| `llama3_1b_llama2_adamw_constant_lr` | Native TorchTitan Trainer, constant LR |
| `llama3_1b_llama2_adamw_dynamic_seq` | Dynamic sequence length |
| `llama3_1b_llama2_adamw_dynamic_seq_bs` | Independent sequence, global batch, and microbatch schedules |
| `llama3_1b_llama2_adamw_seq_grad` | Gradient distances from shared parent sequences |

These entries use the existing experiment AdamW parameter groups.
They do not introduce a new optimizer implementation or an established 1B convergence recipe.
Choose batch sizes, schedules, learning rate, and training budget in an experiment YAML.
Use `python scripts/train.py --list` to inspect all registered entries.

## Examples

The [local-text smoke](../configs/examples/llama3_1b_smoke.yaml) runs ten steps at length 128.
It uses global batch four and microbatch one. One, two, or four replicated GPUs satisfy batch divisibility.
The example stores FP32 model parameters and uses BF16 forward parameters with FP32 gradient reductions.
Full checkpoints are saved at steps five and ten.

```bash
python scripts/train.py configs/examples/llama3_1b_smoke.yaml --nproc 1
```

The [FineWeb example](../configs/examples/llama3_1b_fineweb.yaml) runs ten steps at length 1024.
It evaluates the full fixed 20,971,520-token validation artifact.
Evaluation is more expensive than the ten training updates.

```bash
python scripts/train.py configs/examples/llama3_1b_fineweb.yaml --nproc 1
```

Both examples disable W&B and compilation.
Follow [instruction.md](../instruction.md) for environment, assets, offline data, and recovery.
The short examples do not establish memory requirements for longer sequences or gradient probes.
Changing to the 1B model does not automatically rescale a 100M experiment's budget.

## Tests

```bash
PYTHONPATH=.:cluster/vt_arc OMP_NUM_THREADS=1 python -m unittest tests.test_llama3_1b -v
```

The tests compare Transformer parameter shapes with the upstream model on the meta device.
They check exact parameter count, vocabulary, tied weights, and all five registry entries.
They also parse both examples for one and two GPUs and check the downloaded tokenizer when available.
See [the delivery record](delivery.md) for actual runtime checks and their limits.
