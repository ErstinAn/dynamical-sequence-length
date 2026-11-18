# 长序列训练中的序列长度与 Batch Size 联合调度实验

这组实验研究：在近似相同的梯度 token 预算下，动态调整训练序列长度和 batch size，会怎样影响语言模型的训练过程与长上下文表现。我们比较四种调度方案，同时用固定长度的窗口测量 loss，便于观察训练长度变化之外的模型表现。

实验记录：[W&B project](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler)。

## 数据与模型

使用从 FineWeb-Edu `default` 中筛选的长文档数据集 `fineweb-edu-long-30b`。每篇文档至少包含 12,288 个 Llama 3 tokens，训练集约有 135 万篇文档、30B 原始 tokens。训练与验证按 URL 分组划分，并去除重复文档 ID 和规范化后的重复文本。

Tokenizer 使用原始 Llama 3 tokenizer，词表大小为 128,256。模型从头训练，采用约 **103M 参数的 Llama 3 架构**：12 层、hidden size 512、8 个 attention heads、4 个 KV heads，输入与输出 embedding 权重共享。

## 训练方法

每个 optimizer step 分为两个阶段：

1. 从数据集中读取一批 **12,288-token 父窗口**，每个窗口来自同一篇文档。先对完整窗口做一次无梯度的 forward，记录固定长度下的训练 loss。
2. 从每个父窗口中随机裁剪一个连续片段，长度由当前调度决定，范围为 **1,024–12,288 tokens**。对这些片段做 forward 和 backward，累积梯度后更新模型。

两阶段使用同一批父窗口和本步更新前的相同模型参数。裁剪片段的位置从 0 开始，不保留裁剪起点之前的上下文。数据不跨文档拼接，不足完整父窗口的尾部被丢弃。

这里的 global batch size 表示每步使用的父窗口数量，每个父窗口对应一个用于梯度更新的片段。因此，每步的梯度 token 数为：

```text
梯度 token 数 = global batch size × 当前裁剪长度
```

四个实验均使用 4 卡数据并行、BF16 混合精度和 AdamW，训练 3,840 步，随机种子为 1。AdamW 的 β₁=0.9、β₂=0.95，weight decay=0.1，梯度裁剪阈值为 1。各方案的全程梯度 token 预算约为 **12.08B**；长窗口测量额外消耗计算，因此相近的梯度 token 预算不代表相同的总计算量。

## 四种调度方案

| 实验 | 梯度序列长度 | Global batch size | 学习率 |
| --- | --- | --- | --- |
| [四阶段循环](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/d2cb4eeaed074d8782a1793da93dc19f) | 每 320 步一个周期：前 160 步为 1,024，中间 80 步为 12,288，最后 80 步回到 1,024 | 前 80 步保持 232，随后 80 步升至 936，保持 80 步，再重置为 232 | 恒定 5e-4 |
| [Batch 递增、长度循环](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/ec110cfab65f4d74a3434795e3286748) | 每 320 步在 1,024 → 12,288 → 1,024 之间循环 | 全程从 240 增至 712 | 从 5e-4 cosine 衰减至 0 |
| [Batch 与长度反向循环](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/7b9d0941f9144143a11953e3628d308c) | 每 320 步从 12,288 降至 1,024，再升回长序列 | 与长度反向变化，范围 256–3,072，使每步梯度 token 数近似固定 | 恒定 5e-4 |
| [长度递增、Batch 递减](https://wandb.ai/dynamic_seq_length/longseq-finwebedu-llama-100m-seqbs-joint-scheduler/runs/ea24c5ed526341e59d94f920ba613709) | 全程从 1,024 逐渐增至 12,288 | 全程从 3,072 降至 256，使每步梯度 token 数近似固定 | 恒定 5e-4 |

序列长度取 8 的倍数，batch size 也经过离散化，因此各方案预算近似相同。第二个方案同时改变了学习率调度，与其他方案的差异不能全部归因于序列长度或 batch size。

## 如何评估

在固定的验证集上，以 **12,288-token** 长度评估 1,704 个窗口，共约 20.94M tokens；第 1 步及之后每 20 步验证一次。比较实验时，主要查看固定验证集 loss，并按累计梯度 tokens 对齐训练进度。

W&B 中的几个训练指标含义如下：

| 指标 | 含义 |
| --- | --- |
| `loss_metrics/short_seq_loss` | 裁剪片段上的 token 平均交叉熵，对应实际优化目标；片段达到 12,288 时仍沿用此名称 |
| `loss_metrics/global_avg_loss` | 完整 12,288-token 父窗口上的全局 token 平均交叉熵，仅用于测量 |
| `loss_metrics/global_max_loss` | 四个数据并行 rank 各自的长窗口平均 loss 中的最大值 |

长窗口 loss 小于裁剪片段 loss 是可能的：完整窗口提供更多左侧上下文，而且两者统计的目标 token 集合不同。它们都按各自的有效 token 数归一化，大小关系并不固定。当裁剪长度也为 12,288 时，两者使用相同输入，loss 应基本一致。
