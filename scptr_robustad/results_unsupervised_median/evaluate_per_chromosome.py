"""Post-fit per-chromosome ARI evaluation and figure generation."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import truth
from extract import CHROMS

HERE = Path(__file__).resolve().parent

def main():
    out = HERE / 'per_chromosome'
    labels = truth()
    rows = []
    for chrom in CHROMS:
        row = {'chromosome': chrom, 'n_cells': len(labels), 'k': 5}
        for mode in ('chromosome_only', 'shared_graph'):
            pred = np.load(out / mode / chrom / 'fit.npz')['labels']
            row[f'{mode}_ARI'] = float(adjusted_rand_score(labels, pred))
            row[f'{mode}_NMI'] = float(normalized_mutual_info_score(labels, pred))
        rows.append(row)
    (out / 'evaluation.json').write_text(json.dumps(rows, indent=2))
    with (out / 'ari_by_chromosome.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    pos = np.arange(len(CHROMS))
    ax.bar(pos-.2, [r['chromosome_only_ARI'] for r in rows], .4,
           label='Chromosome-only graph and hierarchies')
    ax.bar(pos+.2, [r['shared_graph_ARI'] for r in rows], .4,
           label='Genome-wide graph, single-chromosome hierarchies')
    ax.set(xticks=pos, xticklabels=CHROMS, ylabel='ARI', ylim=(-.05, 1),
           title='Per-chromosome median clustering · 508 cells · k=5 · 1 Mb')
    ax.tick_params(axis='x', rotation=45); ax.legend(fontsize=9)
    fig.savefig(out / 'ari_by_chromosome.png', dpi=180)
    fig.savefig(out / 'ari_by_chromosome.pdf')
    for r in rows:
        print(f"{r['chromosome']:5s} chromosome_only={r['chromosome_only_ARI']:.4f} shared_graph={r['shared_graph_ARI']:.4f}")

if __name__ == '__main__':
    main()
