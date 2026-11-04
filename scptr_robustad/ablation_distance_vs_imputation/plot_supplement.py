#!/usr/bin/env python3
"""The supplementary figure of the ablation: three arms side by side, drawn like Fig. 3b,c.

Columns are imputation + k-means, no imputation + k-means and no imputation + median (TAD-only
starts, the clean ablation); the top row is the genome-wide ARI and the bottom row the ARI of the
20 chromosome-only fits, next to scHiCluster, Higashi and scHiCPTR. The drawing code, colors and
baseline values are the paper's, as in plot_comparison.py.

    python plot_supplement.py   # TAD_Reorganization_latex/figures/embryo/embryo_imputation_ablation.{pdf,png}
"""

import csv
import sys

from common import HERE, REPO_ROOT
from plot_comparison import tadlineage

sys.path.insert(0, str(REPO_ROOT / "TAD_Reorganization_latex"))
import make_embryo_figure as fig3  # noqa: E402  (sets the paper's rcParams)
import matplotlib.pyplot as plt  # noqa: E402

ARMS = [("imputed", "kmeans", "imputation, k-means"),
        ("raw", "kmeans", "no imputation, k-means"),
        ("raw", "median", "no imputation, median")]
OUT = REPO_ROOT / "TAD_Reorganization_latex" / "figures" / "embryo" / "embryo_imputation_ablation.pdf"


def main():
    with open(HERE / "summary.tsv") as handle:
        summary = list(csv.DictReader(handle, delimiter="\t"))
    ari = fig3.load_module("_ari", REPO_ROOT / "scptr_robustad/plot_ari_comparison.py")
    baselines = {name: ari.read(path, kind) for name, path, kind, _ in ari.METHODS[1:]}
    name = ari.METHODS[0][0]

    fig, axes = plt.subplots(2, 3, figsize=(fig3.FIG_WIDTH_IN, 3.9))
    for column, (hierarchy, method, title) in enumerate(ARMS):
        data = {name: tadlineage(summary, hierarchy, method, "tad_only"), **baselines}
        fig3.draw_genome_wide(axes[0, column], ari, data)
        fig3.draw_per_chromosome(axes[1, column], ari, data)
        axes[0, column].set_title(f"{title}\ngenome-wide", fontsize=fig3.PLOT_FONT)
        axes[0, column].text(-0.22, 1.2, f"({'abc'[column]})", transform=axes[0, column].transAxes,
                             fontsize=fig3.PLOT_FONT, fontweight="bold")
        if column:
            for ax in axes[:, column]:
                ax.set_ylabel("")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, _, c in ari.METHODS]
    fig.legend(handles, [n for n, _, _, _ in ari.METHODS], loc="lower center", ncol=4,
               frameon=False, fontsize=fig3.SMALL)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.0)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print(OUT)


if __name__ == "__main__":
    main()
