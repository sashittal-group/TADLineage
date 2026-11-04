#!/usr/bin/env python3
"""One figure per ablation arm, drawn like panels (b, c) of the paper's Fig. 3.

For each arm {imputed, raw} x {median, k-means}: (b) genome-wide ARI per bin size and (c) the
ARI of the 20 chromosome-only fits, next to scHiCluster, Higashi and scHiCPTR. The drawing code,
colors and baseline values are the paper's (TAD_Reorganization_latex/make_embryo_figure.py and
scptr_robustad/plot_ari_comparison.py, imported, not edited); only the TADLineage values come
from summary.tsv. --rule picks the start-selection rule of score.py (default: TAD-only starts,
the clean ablation; paper_starts adds the spectral start from the contact graph).

    python plot_comparison.py                      # figures/compare_<arm>.{pdf,png}
    python plot_comparison.py --rule paper_starts  # figures/compare_<arm>_paper_starts.{pdf,png}
"""

import argparse
import csv
import sys
from collections import defaultdict

from common import HERE, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "TAD_Reorganization_latex"))
import make_embryo_figure as fig3  # noqa: E402  (sets the paper's rcParams)
import matplotlib.pyplot as plt  # noqa: E402

ARMS = [("imputed", "median", "imputed_median", "imputation + median, as in the paper"),
        ("imputed", "kmeans", "imputed_kmeans", "imputation + k-means, no median"),
        ("raw", "median", "raw_median", "no imputation + median"),
        ("raw", "kmeans", "raw_kmeans", "no imputation + k-means")]
RESOLUTION_KEY = {"250000": "250kb", "500000": "500kb", "1000000": "1000kb"}
RULE_NOTE = {"tad_only": "starts drawn under the TAD distance only",
             "paper_starts": "paper's start set (spectral + 4 k-means++)"}


def tadlineage(summary, hierarchy, method, rule):
    """(genome-wide {bin: ARI}, per chromosome {bin: [ARI]}) in plot_ari_comparison's format."""
    genome, per_chrom = {}, defaultdict(list)
    for row in summary:
        if row["hierarchy"] != hierarchy or row["method"] != method or row["rule"] != rule:
            continue
        key = RESOLUTION_KEY[row["resolution"]]
        if row["scope"] == "all":
            genome[key] = float(row["ari"])
        else:
            per_chrom[key].append(float(row["ari"]))
    return genome, dict(per_chrom)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rule", choices=list(RULE_NOTE), default="tad_only")
    args = parser.parse_args()
    with open(HERE / "summary.tsv") as handle:
        summary = list(csv.DictReader(handle, delimiter="\t"))
    ari = fig3.load_module("_ari", REPO_ROOT / "scptr_robustad/plot_ari_comparison.py")
    baselines = {name: ari.read(path, kind) for name, path, kind, _ in ari.METHODS[1:]}
    color = ari.METHODS[0][3]
    suffix = "" if args.rule == "tad_only" else f"_{args.rule}"
    (HERE / "figures").mkdir(exist_ok=True)

    for hierarchy, method, stem, label in ARMS:
        ari.METHODS[0] = (label, None, None, color)
        data = {label: tadlineage(summary, hierarchy, method, args.rule), **baselines}
        fig, (ax_all, ax_chrom) = plt.subplots(1, 2, figsize=(fig3.FIG_WIDTH_IN, 2.5))
        fig3.draw_genome_wide(ax_all, ari, data)
        fig3.draw_per_chromosome(ax_chrom, ari, data)
        for ax, letter in ((ax_all, "a"), (ax_chrom, "b")):
            ax.text(-0.16, 1.08, f"({letter})", transform=ax.transAxes, fontsize=fig3.PLOT_FONT,
                    fontweight="bold")
        fig.suptitle(f"TADLineage, {label}", fontsize=fig3.PLOT_FONT)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, _, c in ari.METHODS]
        fig.legend(handles, [f"TADLineage ({label})" if i == 0 else name
                             for i, (name, _, _, _) in enumerate(ari.METHODS)],
                   loc="lower center", ncol=4, frameon=False, fontsize=fig3.SMALL)
        fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        out = HERE / "figures" / f"compare_{stem}{suffix}.pdf"
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"), dpi=300)
        plt.close(fig)
        genome, per_chrom = data[label]
        print(f"{stem}{suffix}: genome-wide " + ", ".join(f"{k} {v:.3f}" for k, v in genome.items()))


if __name__ == "__main__":
    main()
