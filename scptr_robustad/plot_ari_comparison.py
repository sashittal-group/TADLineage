#!/usr/bin/env python3
"""The embryo clustering accuracy of the three methods, as one two-panel figure.

a  the genome-wide run: every chromosome pooled into one clustering, one adjusted Rand
   index (ARI) per method and bin size.
b  the per-chromosome runs: the TAD hierarchy of a single chromosome clustered on its own,
   one ARI per chromosome, drawn as a box over the 20 chromosomes. The six boundary
   tolerances of a bin size are re-analyses of one clustering, so they are averaged inside
   a chromosome before the box is drawn; the two baselines have no tolerance parameter.

Panel b is the panel that carries the comparison. Pooling every chromosome gives the two
embedding methods far more signal than a single chromosome does, so panel a understates
how much of the separation comes from the genome-wide pooling rather than from the
representation; panel b removes that advantage and the gap widens.

Usage:
    python plot_ari_comparison.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESOLUTIONS = ["250kb", "500kb", "1000kb"]
RESOLUTION_LABEL = {"250kb": "250 kb", "500kb": "500 kb", "1000kb": "1 Mb"}
# Higashi writes its bin size in base pairs
ALIASES = {"250000": "250kb", "500000": "500kb", "1000000": "1000kb"}

METHODS = [
    ("TADLineage", REPO_ROOT / "scptr_robustad/results_updated/figure/ari_summary.tsv",
     None, "#4c72b0"),
    ("scHiCluster", REPO_ROOT / "schiccluster_scptr/results/figure/ari_summary.tsv",
     "cluster", "#dd8452"),
    ("Higashi", REPO_ROOT / "higashi_scptr/results/figure/ari_summary.tsv",
     "cluster", "#55a868"),
    # scHiCPTR is the method that was published on this dataset; its "cluster" rows carry the
    # better of the two clusterings it is scored under (see schicptr_scptr/README.md)
    ("scHiCPTR", REPO_ROOT / "schicptr_scptr/results/figure/ari_summary.tsv",
     "cluster", "#c44e52"),
]


def read(path, kind):
    """``(genome_wide, per_chromosome)`` keyed by bin size, ARI averaged over tolerances."""
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    if kind is not None:
        rows = [r for r in rows if r.get("kind") == kind]
    pooled = collections.defaultdict(list)
    chroms = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        resolution = ALIASES.get(row["resolution"], row["resolution"])
        if resolution not in RESOLUTIONS:
            continue
        if row["chromosome"] == "all":
            pooled[resolution].append(float(row["ari"]))
        else:
            chroms[resolution][row["chromosome"]].append(float(row["ari"]))
    genome_wide = {r: statistics.mean(v) for r, v in pooled.items()}
    per_chromosome = {r: [statistics.mean(v) for v in by_chrom.values()]
                      for r, by_chrom in chroms.items()}
    return genome_wide, per_chromosome


def panel_genome_wide(ax, data, letter):
    width = 0.78 / len(METHODS)
    centre = (len(METHODS) - 1) / 2
    positions = np.arange(len(RESOLUTIONS))
    for offset, (name, _, _, color) in enumerate(METHODS):
        heights = [data[name][0].get(r, np.nan) for r in RESOLUTIONS]
        bars = ax.bar(positions + (offset - centre) * width, heights, width, color=color,
                      edgecolor="white", linewidth=0.6, label=name)
        for bar, height in zip(bars, heights):
            if height == height:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([RESOLUTION_LABEL[r] for r in RESOLUTIONS])
    ax.set_xlabel("bin size")
    ax.set_ylabel("ARI against the five stages")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{letter}  Every chromosome pooled into one clustering",
                 loc="left", fontsize=10)


def panel_per_chromosome(ax, data, letter):
    width = 0.78 / len(METHODS)
    centre = (len(METHODS) - 1) / 2
    positions = np.arange(len(RESOLUTIONS))
    for offset, (name, _, _, color) in enumerate(METHODS):
        values = [data[name][1].get(r, []) for r in RESOLUTIONS]
        centres = positions + (offset - centre) * width
        boxes = ax.boxplot(values, positions=centres, widths=width * 0.86,
                           patch_artist=True, manage_ticks=False,
                           medianprops=dict(color="black", linewidth=1.1),
                           flierprops=dict(marker="o", markersize=2.2,
                                           markerfacecolor=color, markeredgecolor="none",
                                           alpha=0.8))
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor("0.3")
            patch.set_linewidth(0.6)
        for element in ("whiskers", "caps"):
            for line in boxes[element]:
                line.set_color("0.3")
                line.set_linewidth(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([RESOLUTION_LABEL[r] for r in RESOLUTIONS])
    ax.set_xlabel("bin size")
    ax.set_ylabel("ARI of one chromosome\n(box over the 20 chromosomes)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{letter}  One chromosome clustered on its own",
                 loc="left", fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(HERE / "results_updated" / "figure"
                                                / "ari_comparison.png"))
    args = parser.parse_args()

    data = {}
    for name, path, kind, _ in METHODS:
        if not Path(path).exists():
            print(f"missing {path}")
            return
        data[name] = read(path, kind)

    for name, (genome_wide, per_chromosome) in data.items():
        for resolution in RESOLUTIONS:
            values = per_chromosome.get(resolution, [])
            print(f"{name:12s} {resolution:7s} genome-wide "
                  f"{genome_wide.get(resolution, float('nan')):.3f}   "
                  f"per-chromosome mean {statistics.mean(values):.3f} over {len(values)}"
                  if values else f"{name} {resolution}: no per-chromosome runs")

    figure, axes = plt.subplots(1, 2, figsize=(8.1, 3.8))
    panel_genome_wide(axes[0], data, "a")
    panel_per_chromosome(axes[1], data, "b")
    figure.tight_layout(w_pad=2.0, rect=(0, 0.07, 1, 1))
    # one legend under both panels: the four methods no longer fit across the top of panel a
    # without crowding the value labels, and both panels use the same colours
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white", linewidth=0.6)
               for _, _, _, color in METHODS]
    figure.legend(handles, [name for name, _, _, _ in METHODS], loc="lower center",
                  ncol=len(METHODS), frameon=False, fontsize=9, columnspacing=2.0,
                  handlelength=1.4, bbox_to_anchor=(0.5, 0.0))

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
