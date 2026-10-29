# Method

## Model and optimizer

The reference model has 100,092,672 trainable parameters.
It has 12 layers, width 768, 12 attention heads, and four key/value heads.
The vocabulary contains 32,000 tokens.
The input embedding and output matrix share their parameters.

The matrix group contains the internal matrix weights.
The other group contains the embeddings, output parameters, final normalization, and parameters with fewer than two dimensions.
The existing code calls this other group the vector group.
The vector group can contain matrices.

AdamW uses PyTorch foreach operations for both groups.
Muon uses `torch.optim.Muon` for the matrix group.
The vector group always uses AdamW.
The defaults keep the previous group order and optimizer settings.

## Batch definitions

Let `D` be the number of data-parallel replicas.
Let `B_local` be the number of sequences in one local batch.
Let `K` be the number of gradient accumulation groups.
Let `L` be the sequence length.

```text
B_global = D * B_local * K
Tokens per step = B_global * L
```

The reference configuration uses `D=2`, `B_local=16`, and `K=16`.
Each step contains 512 sequences.
The step count changes with the sequence length to keep approximately the same token budget.

## Gradient variance

The monitor reads the cumulative gradients after each backward operation.
TorchTitan completes the DP gradient sum before this read.
The monitor does not change the gradients.

Let `C_j` be the cumulative gradient after group `j`.
The training loss uses the token count for the complete optimizer step.
For equal token counts, the mean gradient for group `j` is:

```text
h_j = K * (C_j - C_(j-1))
C_0 = 0
G = (h_1 + ... + h_K) / K
V = sum_j ||h_j - G||^2 / (K - 1)
```

The implementation calculates this value with an online variance update.
It uses FP64 for the squared differences and their sum.
It keeps one gradient snapshot per parameter.

`global_grad_var` is the sum of the parameter variance values.
This value is a variance across synchronized accumulation groups.
It is not a variance across individual training examples.
The measurement occurs before gradient clipping.
The monitor uses `metrics.log_freq` as its measurement interval.

## Supported configurations

For the monitor, these conditions are necessary:

- A fixed sequence length and fixed batch sizes within each run.
- At least two accumulation groups.
- Equal positive global token counts for all groups.
- A single model part.
- Replicated data parallelism.
- Dense gradients with complete local parameter values.

The monitor rejects tensor, pipeline, context, expert, and nontrivial parameter sharding.
The fixed validator has a separate DP implementation.
Its broader DP support does not extend the monitor support.
