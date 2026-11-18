"""Plot the fixed loss length and the six crop and batch schedules."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / "outputs/longseq_loss_grad"
names = ["bs_increasing_seq_decreasing", "bs_decreasing_seq_increasing",
         "bs_cyclic_seq_inverse", "linear_bs_cyclic_seq_cosine_lr",
         "linear_bs_cyclic_seq_constant_lr", "four_stage_cycle"]
fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
for ax, name in zip(axes.flat, names):
    with (root / f"{name}_schedule.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(r["step"]) for r in rows]
    ax.plot(steps, [int(r["grad_seq_len"]) for r in rows], color="#1768ac", linewidth=1.2)
    ax.axhline(12288, color="#555555", linestyle="--", linewidth=1)
    ax.set_ylim(0, 13000)
    ax.set_yticks([1024, 4096, 8192, 12288])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Crop length (blue), loss length (dashed)", color="#1768ac")
    ax.grid(alpha=0.2)
    other = ax.twinx()
    other.plot(steps, [int(r["global_batch"]) for r in rows], color="#d56812", linewidth=1, alpha=0.8)
    other.set_yscale("log", base=2)
    other.set_ylim(192, 4096)
    other.set_yticks([256, 512, 1024, 2048, 3072], labels=["256", "512", "1,024", "2,048", "3,072"])
    other.set_ylabel("Global batch (orange, log scale)", color="#d56812")
    ax.set_title(name.replace("_", " "), fontsize=11)
fig.suptitle("LossGradSeqTrainer: fixed 12,288-token loss\n3,840 updates and approximately 12.08B gradient tokens per run", fontsize=15)
fig.savefig(root / "schedules.png", dpi=160)
fig.savefig(root / "schedules.svg")
print(root / "schedules.png")
