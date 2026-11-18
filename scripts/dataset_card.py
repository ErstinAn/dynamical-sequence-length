"""Write a dataset card from measured statistics and frozen metadata."""

import json
from pathlib import Path


def write_dataset_card(root):
    root = Path(root)
    stats = json.loads((root / "statistics.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    validation = json.loads((root / "validation/manifest.json").read_text())
    train, val, fixed = (stats[k] for k in ("train", "validation", "fixed_validation"))
    length_report = {
        "manifest_sha256": stats["dataset_manifest_sha256"],
        "unit": "raw Llama 3 tokens; excludes BOS and EOS",
        "splits": {k: {"documents": stats[k]["documents"], "tokens": stats[k]["raw_tokens"],
                       **stats[k]["document_length"]} for k in ("train", "validation")},
        "window_lengths": {
            "train": "Selected by experiment configuration, not by the document length histogram",
            "fixed_validation": {"length": fixed["sequence_length"], "count": fixed["num_sequences"],
                                 "padding_positions": fixed["padding_positions"]},
        },
    }
    (root / "length_statistics.json").write_text(json.dumps(length_report, indent=2, allow_nan=False) + "\n")
    summary = []
    for label, split in (("训练集", train), ("完整验证集", val)):
        length = split["document_length"]
        summary.append(
            f"| {label} | {split['documents']:,} | {split['raw_tokens']:,} | "
            f"{length['minimum']:,} | {length['quantiles']['0.5']:,.0f} | "
            f"{length['mean']:,.2f} | {length['maximum']:,} |")
    labels = ("小于 16,384", "16,384–24,575", "24,576–32,767", "32,768–65,535",
              "65,536–131,071", "至少 131,072")
    doc_rows, token_rows = [], []
    for i, label in enumerate(labels):
        counts = [s["document_length"]["bin_documents"][i] for s in (train, val)]
        doc_rows.append(f"| {label} | {counts[0]:,} | {counts[0] / train['documents']:.2%} | "
                        f"{counts[1]:,} | {counts[1] / val['documents']:.2%} |")
        token_rows.append(f"| {label} | {train['length_token_fractions'][i]:.2%} | "
                          f"{val['length_token_fractions'][i]:.2%} | {fixed['length_token_fractions'][i]:.2%} |")
    means = train["ordered_block_document_length_means"]
    shares = train["ordered_block_supplement_token_fractions"]
    rank_means = [x for rank in train["dp4_ordered_block_document_length_means"] for x in rank]
    orders = manifest["document_orders"]
    sampling = validation["sampling"]
    if validation.get("packing") == "none":
        method = f"""每个窗口只来自一篇文档。长文档可以提供多个窗口。
不同窗口不共享原始 token，连用于最后一个预测目标的额外 token 也不共享。
每个输入有 {fixed['sequence_length']:,} 个位置，另取下一个 token 来生成预测标签。
因此，每个候选块占用 {sampling['block_width_including_lookahead']:,} 个文档位置。

先按已保存的验证文档顺序读取，再在每篇文档内建立互不重叠的候选块。
每篇文档的第一个块使用随机起点。起点在能容纳该文档全部候选块的范围内均匀抽取。
起点 seed 为 {sampling['offset_seed']}。起点不是固定为文档的第零个 token。
最后用 seed {sampling['seed']} 从 {sampling['candidate_sequences']:,} 个候选窗口中抽取 {fixed['num_sequences']:,} 个，不重复抽取。
这是按候选窗口抽样。长文档可以贡献更多候选窗口，因此各篇文档的入选概率不相同。

输入与预测标签均来自同一文档的连续 token。位置编号在每个窗口内从 0 开始。
BOS 和 EOS 只在窗口碰到真实文档边界时出现，不添加到文档中间的随机起点。
本次不做 packing，也不补 padding。
{fixed['documents_with_multiple_selected_windows']:,} 篇文档各提供了多个入选窗口。
每篇文档最多提供 {fixed['max_selected_windows_per_document']} 个入选窗口。
同一文档的不同片段可能主题相近。不重叠不等于统计上完全独立。"""
        coverage = f"""完整验证集有 {val['documents']:,} 篇文档。
其中 {sampling['eligible_documents']:,} 篇能提供完整窗口，{sampling['excluded_short_documents']:,} 篇过短，未参与本次窗口抽样。
资格下限是 {sampling['minimum_raw_tokens']:,} 个原始 token，加上文档自身的 BOS/EOS 后必须能提供完整输入和下一个预测目标。
符合长度条件的文档中，有 {fixed['eligible_original_documents']:,} 篇来自原始池，{fixed['eligible_supplement_documents']:,} 篇来自补充池。

这组验证样本主要衡量长文档上的表现。它不代表完整混合数据的长度分布。
旧版 packed 验证的来源占比与长度占比验收界限，不适用于这次不拼接的验证方案。
新旧验证的长度、样本和 token 数均不同。不要把新旧 loss 直接拼成同一条可比曲线。"""
    else:
        method = f"""先按 seed {orders['validation']['seed']} 打乱完整验证集的文档，再把文档拼成 token 流。
然后按 {fixed['sequence_length']:,} token 切出候选窗口。
最后用 seed {sampling['seed']} 从全部 {sampling['candidate_sequences']:,} 个候选窗口中抽取 {fixed['num_sequences']:,} 个窗口，不重复抽取。"""
        coverage = """抽样前设定的验收界限为：来源占比差不超过 2 个百分点，各长度区间差不超过 3 个百分点。
这组样本覆盖了两批来源，但没有覆盖完整验证集的每篇文档。
文档边界处的跨文档预测目标会被屏蔽。"""
    (root / "README.md").write_text(f"""---
language:
- en
license: odc-by
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
---

# FineWeb-Edu：固定文档顺序与固定验证样本

本数据包供不同 batch size schedule 和 sequence length schedule 共用。
训练集与验证集的文档成员不变。文档内部的 token 顺序不变。
训练文档打乱一次，之后按固定顺序读取。
验证时，每次使用相同的 {fixed['num_sequences']:,} 个窗口。

## 先区分三种长度

- **文档长度**：一篇原始文档经 Llama 3 tokenizer 编码后的 token 数。下表不计 BOS 和 EOS。
- **训练窗口长度**：数据加载器从 token 流中取出的长度。它由实验配置决定。一个窗口可以包含多篇文档的片段。
- **验证窗口长度**：本包固定为 {fixed['sequence_length']:,}。每次验证都不变。

文档打乱只改变顺序。它不会改变文档长度，也不会把长度分布变成均匀分布。
`seq_len` 必须说明指的是上述哪一种长度。

## 原始文档的长度分布

| 数据 | 文档数 | 原始 token 数 | 最短 | 中位数 | 平均长度 | 最长 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary)}

以下比例以**文档数**为分母。

| 文档长度（token） | 训练文档数 | 训练文档占比 | 验证文档数 | 验证文档占比 |
|---|---:|---:|---:|---:|
{chr(10).join(doc_rows)}

以下比例以**实际输入的原始 token 数**为分母。
固定验证窗口中的一个 token，按它所属的完整原始文档长度归类。
这张表不是验证窗口的长度分布。

| 所属文档长度（token） | 训练 token 占比 | 完整验证 token 占比 | 固定验证 token 占比 |
|---|---:|---:|---:|
{chr(10).join(token_rows)}

完整统计见 [statistics.json](statistics.json)。它还包含分位数和固定验证样本涉及的文档统计。

## 固定验证样本

{method}
抽样只在准备数据时做一次。训练中的每次验证都完整读取这同一组窗口。

| 项目 | 数值 |
|---|---:|
| 窗口数 | {fixed['num_sequences']:,} |
| 每个窗口长度 | {fixed['sequence_length']:,} |
| 输入位置总数 | {validation['physical_positions']:,} |
| 原始文本输入 token 数 | {fixed['raw_tokens']:,} |
| 参与 loss 计算的预测目标数 | {fixed['valid_labels']:,} |
| padding 位置数 | {fixed['padding_positions']:,} |
| 输入涉及的不同原始文档数 | {fixed['documents']:,} |
| 其中原始池文档数 | {fixed['original_documents']:,} |
| 其中补充池文档数 | {fixed['supplement_documents']:,} |

原始池和补充池是构建数据时使用的两批来源。
补充池按源数据的 `source_tokens < 16384` 识别。该字段与重新编码后的 Llama 3 文档长度不同。
完整验证集中，补充池占原始 token 的 {val['supplement_token_fraction']:.4%}。
固定验证样本中的比例是 {fixed['supplement_token_fraction']:.4%}。
各长度区间与完整验证集的 token 占比，最大相差 {100 * max(map(abs, fixed['length_token_fraction_differences'])):.3f} 个百分点。
{coverage}

它用于固定成本的重复评估。它不是整个验证集的完整评估。
`validation/manifest.json` 记录输入、标签、位置数组的校验值。
`independent_validation_check.json` 记录对此数据包全部固定窗口的独立重建检查。

## 打乱后，还会出现成段的长度突变吗？

将训练读取顺序分成 {len(means)} 段，每段文档数近似相同，实际统计如下。

- 每段原始文档平均长度：{min(means):,.2f}～{max(means):,.2f} token。
- 每段补充池 token 占比：{min(shares):.4%}～{max(shares):.4%}。
- 按 4 个数据并行 rank 分配后，每个 rank 再分成 32 段；这些段的平均文档长度：{min(rank_means):,.2f}～{max(rank_means):,.2f} token。

这表明补充池已分散到读取顺序中。原先两批来源前后成块排列的问题已消除。
这不表示每个 batch 的长度组成相同，也不保证 loss 永远没有突变。
本次检查使用数据统计，没有运行完整训练来验证新的 loss 曲线。

此前诊断的 run 中，训练父序列一直为 22,256 token。
发生突变的是其输入文档的来源与长度组成。
如果实验使用 loss/gradient 分离，短梯度片段的长度仍由 schedule 控制。
仅打乱数据，不能解决短片段训练与长上下文评估之间的差异。

## 读取规则

`orders/train.npy` 保存全部训练文档编号的排列，seed 为 {orders['train']['seed']}。
`orders/validation.npy` 保存全部验证文档编号的排列，seed 为 {orders['validation']['seed']}。
生成器为 NumPy PCG64。以已保存的数组和 SHA256 为准。
全局文档编号由 `dataset_manifest.json` 中该 split 的文件顺序和文件内行号决定。

**Parquet 文件内的行顺序没有改写。**
只调用 `datasets.load_dataset()` 不会应用上述排列。
要复现本项目的训练读取规则，使用随包代码中的 `LongDocumentBatchDataLoader`，并设置 `packing: stream`。
它先应用文档排列，再按排列位置分配给各个数据并行 rank。
rank `r` 读取 `order[r::world_size]`。
它在每个 rank 的 token 流中顺序取窗口，并保留文档尾部。
它不在每一步随机选择流中的起点，也不打乱文档内部 token。

合作者也可以在自己的训练框架中应用 `orders/train.npy`。
例如，用 Hugging Face Datasets 加载 Parquet 后，通过 `dataset.select(order.tolist())` 应用排列。
加载 Parquet 时，必须按 manifest 中该 split 的文件顺序传入文件列表。
应用排列后，不要再调用 `.shuffle()`。
这种方法只规定文档顺序；窗口切分、文档边界、标签屏蔽和位置编号仍由合作者的加载器负责。

不同实验可以共用同一份文档排列，无需预先为每种 schedule 切好训练窗口。
窗口由各自的数据加载器按实际配置读取。不同 schedule 不会得到相同的 batch 边界。
`training_settings.json` 提供本包所需的数据配置。
现有实验配置不会自动切换到本包；启动前必须应用这些配置。
验证使用固定的 `validator.eval_seq_len` 与 `validator.token_budget`。
使用 `LossLengthValidator` 的 loss/gradient 分离实验，还需设置 `validator.follow_loss_length: false`。
这样验证长度固定，不随训练父序列长度变化。普通固定验证器不需要这个选项。

只训练一遍时，设置 `dataloader.infinite: false`。
检查每个 rank 消耗的父序列位置数是否小于该 rank 可用的位置数。
不能仅用梯度片段的 token budget 判断是否读完一遍。
此前 3,840 步 run 消耗约 60.67B 个父序列位置，不适合直接用于一次遍历。

## 从 Hugging Face 下载并用于训练

仓库为私有仓库：`long-sequence-length-dataset/fineweb-edu-long-30b`。
合作者需要该仓库的读取权限，以及 `tokenizer_provenance.json` 指定的 Meta Llama 3 tokenizer 访问权限。
每人使用自己的凭据。包中不含凭据或模型权重。

先登录 Hugging Face，然后下载元数据、固定验证数组和训练代码。
下面的路径与训练代码中的默认路径一致。

```python
from huggingface_hub import HfApi, snapshot_download

repo = "long-sequence-length-dataset/fineweb-edu-long-30b"
revision = HfApi().dataset_info(repo).sha
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    revision=revision,
    ignore_patterns=["data/*"],
    local_dir="project/assets/data/fineweb_edu_long_30b_shuffled_v1",
)
print("Downloaded revision:", revision)
```

```bash
tar -xzf project/assets/data/fineweb_edu_long_30b_shuffled_v1/training_code.tar.gz
cd project
# Install this project's runtime from requirements-lock.txt.
python assets/data/fineweb_edu_long_30b_shuffled_v1/restore_final_dataset.py \\
  --dataset-dir assets/data/fineweb_edu_long_30b_shuffled_v1 \\
  --tokenizer-dir assets/hf/llama3-tokenizer
python -m pip install --no-deps -e .
```

恢复脚本读取当前仓库版本，先确认远端元数据与本包一致，再下载文本文件。
也可以用 `--revision` 指定本次下载得到的提交编号。
它不依赖旧版数据的提交仍然存在。
脚本使用固定 tokenizer 重新编码，并核对文本、token、索引和最终 manifest 的 SHA256。
需要约 52 GB 文本文件空间，以及约 120 GB token 缓存空间。
固定验证数组已经包含 token，不需要重新抽样。

本机的 `data/` 和 `tokens/` 是指向已验证原始文件的链接。
Hugging Face 提供实际 Parquet 文件，不提供本机链接或 120 GB token 缓存。
`source_upload_receipt.json` 只记录原始数据的上传来源；其旧提交不是恢复依赖。
仓库只提供当前数据包，不另设旧版目录。

数据集 manifest SHA256：`{stats['dataset_manifest_sha256']}`。
固定验证 manifest SHA256：`{fixed['manifest_sha256']}`。

## 来源与许可

原始语料来自 [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)。
数据许可为 ODC-By。使用和再分发时保留来源说明。
`selection.json` 记录选择规则。`tokenizer_provenance.json` 记录 tokenizer 的版本与校验值。
训练代码的许可见代码包内的 `LICENSE` 与 `NOTICE`。
""")
