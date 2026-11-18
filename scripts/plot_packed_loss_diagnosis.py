"""Render the measured loss discontinuities and fixed-checkpoint intervention."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs/packed_loss_diagnosis_20261116"
rows = list(csv.DictReader((OUT / "replay.csv").open()))
x = np.array([int(r["step"]) for r in rows])
y = np.array([float(r["long_loss"]) for r in rows])
short = np.array([float(r["short_loss"]) for r in rows])
fraction = np.array([float(r["supplement_fraction"]) for r in rows])
v = [(int(r["step"]), float(r["validation_loss"])) for r in rows if r["validation_loss"]]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(13, 9), layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1, 1])
ax = fig.add_subplot(gs[0, :])
ax.plot(x, y, lw=.7, alpha=.7, label="Long training-batch loss", color="#2855a6")
ax.plot(x, short, lw=.6, alpha=.6, label="Random-crop training loss", color="#169582")
ax.plot(*zip(*v), lw=1.8, label="Fixed validation loss", color="#d26827")
ax.axvspan(2094, 2538, color="#f4ce58", alpha=.17, label="All ranks in supplement pool")
ax.set(xlim=(500, 3840), ylim=(3, 5.3), ylabel="Mean token NLL", title="The loss steps track data-pool boundaries")
ax.legend(loc="upper left", ncol=2, fontsize=9)
for slot, lo, hi, title in [(gs[1, 0], 2050, 2130, "Entering the 12k supplement"),
                             (gs[1, 1], 2500, 2580, "Wrapping to the original pool")]:
    a = fig.add_subplot(slot)
    a.plot(x, y, color="#2855a6", lw=1.2)
    a.set(xlim=(lo, hi), ylim=(3.85, 4.65), ylabel="Long loss", title=title)
    b = a.twinx()
    b.plot(x, fraction, color="#ad7a00", lw=1.2, ls="--")
    b.set(ylim=(-.1, 1.1), ylabel="Supplement token fraction")
    b.tick_params(axis="y", colors="#ad7a00")
a = fig.add_subplot(gs[2, 0])
a.plot(x, [float(r["mean_fragment_positions"]) for r in rows], color="#2855a6", lw=.7)
a.set(xlim=(1800, 2800), xlabel="Optimizer step", ylabel="Positions per fragment",
      title="Packed tensor length stays 22,256; fragment lengths change")
a = fig.add_subplot(gs[2, 1])
a.plot(x, [int(r["grad_length"]) for r in rows], color="#169582")
a.set(xlim=(1800, 2800), xlabel="Optimizer step", ylabel="Gradient crop length",
      title="Gradient length changes smoothly across both events")
fig.savefig(OUT / "loss_boundaries.png", dpi=180)
plt.close(fig)

results = json.loads((OUT / "gpu_summary.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
labels = ["Before drop\n(step 2080 data)", "After drop\n(step 2120 data)", "After rise\n(step 2560 data)"]
for i, (treatment, label, color) in enumerate([
    ("long", "Full document context", "#2855a6"),
    ("crop4096", "Random 4096-token crop", "#169582"),
    ("sliding4096", "Same full targets, 4096-token attention", "#d26827")]):
    values = [next(r["loss"] for r in results if r["step"] == s and r["treatment"] == treatment)
              for s in [2080, 2120, 2560]]
    bars = axes[0].bar(np.arange(3) + (i-1)*.25, values, .25, label=label, color=color)
    axes[0].bar_label(bars, fmt="%.3f", fontsize=8)
axes[0].set(xticks=range(3), xticklabels=labels, ylim=(0, 5.4), ylabel="Mean token NLL",
            title="Fixed step-3840 weights; 32 parents per data region")
axes[0].legend(fontsize=8)
for treatment, label, color in [("long", "Full context", "#2855a6"),
                                 ("sliding4096", "4096-token attention", "#d26827")]:
    r = next(r for r in results if r["step"] == 2080 and r["treatment"] == treatment)
    axes[1].plot(range(6), [b["loss"] for b in r["position_bins"]], marker="o", label=label, color=color)
axes[1].set(xticks=range(6), xticklabels=["0–2k", "2–4k", "4–8k", "8–12k", "12–16k", "16–22k"],
            ylabel="Mean token NLL", xlabel="Position within a document fragment",
            title="Paired targets and position IDs; only attention span changes")
axes[1].legend()
fig.savefig(OUT / "fixed_checkpoint_ablation.png", dpi=180)
