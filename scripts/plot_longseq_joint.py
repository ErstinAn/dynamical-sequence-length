"""Plot the exact generated sequence and global-batch schedules."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / 'outputs/longseq_joint'
names = ['bs_increasing_seq_decreasing', 'bs_decreasing_seq_increasing',
         'bs_cyclic_seq_inverse', 'linear_bs_cyclic_seq_cosine_lr',
         'linear_bs_cyclic_seq_constant_lr', 'four_stage_cycle']
fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
for ax, name in zip(axes.flat, names):
    with (root / f'{name}_schedule.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(r['step']) for r in rows]
    lengths = [int(r['seq_len']) for r in rows]
    batches = [int(r['global_batch']) for r in rows]
    ax.plot(steps, lengths, color='#1768ac', linewidth=1.1)
    ax.set_ylim(0, 12800)
    ax.set_yticks([0, 3072, 6144, 9216, 12288])
    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Sequence length', color='#1768ac')
    ax.grid(alpha=0.2)
    other = ax.twinx()
    other.plot(steps, batches, color='#d56812', linewidth=1.1, alpha=0.8)
    other.set_yscale('log', base=2)
    other.set_ylim(128, 131072)
    other.set_yticks([256, 1024, 4096, 16384, 65536],
                    labels=['256', '1,024', '4,096', '16,384', '65,536'])
    other.set_ylabel('Global batch (log scale)', color='#d56812')
    ax.set_title(name.replace('_', ' '), fontsize=11)
fig.suptitle('Long-document joint schedules\n3,840 updates and 12,079,595,520 training labels per run', fontsize=15)
fig.savefig(root / 'schedules.png', dpi=160)
fig.savefig(root / 'schedules.svg')
print(root / 'schedules.png')
