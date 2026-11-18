"""Plot exact source-length tails for default and sample-100BT."""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/fineweb_edu_long/full-default-profile'
SOURCES = [
    ('default', OUTPUT / 'length_histogram.npz', '#2463A0'),
    ('sample-100BT', ROOT / 'outputs/fineweb_edu_profile/full-100bt/length_histogram.npz', '#C56523'),
]

def main():
    x = np.unique(np.geomspace(256, 65536, 450).astype(np.int64))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    provenance = {}
    for label, path, color in SOURCES:
        counts = np.load(path)['counts'].astype(np.int64)
        document_tail = np.cumsum(counts[::-1])[::-1]
        token_tail = np.cumsum((counts * np.arange(len(counts), dtype=np.int64))[::-1])[::-1]
        axes[0].plot(x, document_tail[x] / counts.sum() * 100, color=color, label=label, lw=2.1)
        axes[1].plot(x, token_tail[x] / 1e9, color=color, label=label, lw=2.1)
        provenance[label] = {'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'documents': int(counts.sum()), 'source_tokens': int(token_tail[0])}
    ticks = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    labels = ['256', '512', '1K', '2K', '4K', '8K', '16K', '32K', '64K']
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(ticks, labels)
        ax.set_xlim(256, 65536)
        ax.grid(True, which='major', alpha=0.2)
        ax.set_xlabel('Minimum document length in source tokens')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_title('Share of documents at or above each length')
    axes[0].set_ylabel('Documents (%)')
    axes[1].set_title('Tokens in complete retained documents')
    axes[1].set_ylabel('Source tokens (billions)')
    fig.suptitle('FineWeb-Edu: exact source-length distribution', fontsize=14)
    fig.supxlabel('Source tokens use the dataset token_count field. These are not Llama 3 counts.', fontsize=10)
    for suffix in ['png', 'svg']:
        fig.savefig(OUTPUT / f'length_distribution.{suffix}', dpi=170, bbox_inches='tight')
    figures = ROOT / 'docs/fineweb-edu-stats/figures'
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg']:
        shutil.copy2(OUTPUT / f'length_distribution.{suffix}', figures / f'length-distribution.{suffix}')
    (OUTPUT / 'plot_provenance.json').write_text(json.dumps({'matplotlib': matplotlib.__version__,
        'numpy': np.__version__, 'inputs': provenance}, indent=2)+'\n')
    print(OUTPUT / 'length_distribution.png')

if __name__ == '__main__':
    main()
