#!/usr/bin/env python3
"""The six panels that carry the argument of the report, as one figure.

Read left to right, top to bottom, they go from what the events are to whether anything
biological follows from them:

a. how much of a stage median survives into the next stage
b. what the events are made of, by bin size
c. how many loci are recovered at more than one bin size
d. gene and CTCF density at the boundaries, against the null of relocating them
e. transition specificity of the transcriptional response, one point per (bin size,
   transition) cell
f. chrX boundary loss against the autosomes of the same run

Panels d and f both have a reference line at the value a null result takes, because in both
cases the null result is the outcome.

Usage:
    python plot_summary.py
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from summarise_results import median_turnover, read_tsv  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STAGE_TRANSITIONS = ["1-cell->2-cell", "2-cell->4-cell", "4-cell->8-cell",
                     "8-cell->64-cell"]
SHORT = {t: t.replace("-cell", "").replace("->", "→") for t in STAGE_TRANSITIONS}
RESOLUTIONS = ["250kb", "500kb", "1000kb"]
RESOLUTION_COLORS = {"250kb": "#4c72b0", "500kb": "#dd8452", "1000kb": "#55a868"}


def panel_turnover(ax):
    shared = median_turnover()
    width = 0.26
    positions = np.arange(len(STAGE_TRANSITIONS))
    for offset, resolution in enumerate(RESOLUTIONS):
        values = [100 * np.mean(shared.get((resolution, t), [0]))
                  for t in STAGE_TRANSITIONS]
        ax.bar(positions + (offset - 1) * width, values, width,
               label=resolution, color=RESOLUTION_COLORS[resolution])
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT[t] for t in STAGE_TRANSITIONS])
    ax.set_ylabel("domains shared with the\nnext median (%)")
    ax.set_title("a  Most of a median is replaced,\nnot rearranged", loc="left",
                 fontsize=10)
    ax.set_ylim(0, 40)
    ax.legend(fontsize=7, frameon=False)


def panel_event_mix(ax):
    counts = collections.defaultdict(collections.Counter)
    for path in RESULTS.glob("events/*/*/tol_*_events.tsv"):
        for row in read_tsv(path):
            counts[row["resolution"]][row["event_type"]] += 1
    kinds = ["domain_gain", "domain_loss", "split", "merge"]
    colors = ["#c7c7c7", "#9e9e9e", "#4c72b0", "#c44e52"]
    bottom = np.zeros(len(RESOLUTIONS))
    for kind, color in zip(kinds, colors):
        values = np.array([counts[r][kind] for r in RESOLUTIONS], dtype=float)
        totals = np.array([sum(counts[r].values()) for r in RESOLUTIONS], dtype=float)
        share = 100 * values / np.maximum(totals, 1)
        ax.bar(RESOLUTIONS, share, bottom=bottom, label=kind.replace("_", " "),
               color=color)
        bottom += share
    ax.set_ylabel("share of events (%)")
    ax.set_title("b  Splits and merges are a\nfew per cent of events", loc="left",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="center right")


def panel_reproducibility(ax):
    rows = read_tsv(RESULTS / "consensus" / "consensus_events.tsv")
    support = collections.Counter(int(row["resolutions"]) for row in rows)
    labels = ["1 bin size", "2 bin sizes", "3 bin sizes"]
    values = [support.get(n, 0) for n in (1, 2, 3)]
    ax.bar(labels, values, color=["#c7c7c7", "#4c72b0", "#55a868"])
    for index, value in enumerate(values):
        ax.text(index, value, f" {value}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("loci with an adjacent\nsplit or merge")
    ax.set_title("c  Almost nothing is recovered\nat a second bin size", loc="left",
                 fontsize=10)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)


def panel_density(ax):
    enrichment = read_tsv(RESULTS / "enrichment" / "all_configurations_enrichment.tsv")
    ctcf = read_tsv(RESULTS / "ctcf" / "ctcf_at_boundaries.tsv")
    wanted = ["boundary_gained", "boundary_lost", "boundary_stable",
              "split_boundary", "merge_boundary"]
    genes = collections.defaultdict(list)
    for row in enrichment:
        if row["gene_set"] == "all_genes" and row["fold"]:
            genes[row["feature_set"]].append(float(row["fold"]))
    peaks = collections.defaultdict(list)
    for row in ctcf:
        if row["fold"]:
            peaks[row["feature_set"]].append(float(row["fold"]))

    positions = np.arange(len(wanted))
    width = 0.38
    ax.bar(positions - width / 2, [np.mean(genes[n]) if genes[n] else 0 for n in wanted],
           width, label="genes", color="#4c72b0")
    ax.bar(positions + width / 2, [np.mean(peaks[n]) if peaks[n] else 0 for n in wanted],
           width, label="CTCF peaks", color="#c44e52")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([n.replace("boundary_", "").replace("_boundary", "")
                        for n in wanted], rotation=20, ha="right")
    ax.set_ylabel("density vs relocated windows")
    ax.set_ylim(0, 1.3)
    ax.set_title("d  These boundaries avoid genes\nand CTCF", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)


def panel_specificity(ax):
    rows = read_tsv(RESULTS / "expression" / "all_configurations_contrasts.tsv")
    cells = collections.defaultdict(list)
    for row in rows:
        if row["statistic"] != "mean_abs_change" or row["matched_minus_other"] == "":
            continue
        cells[(row["feature_set"], row["resolution"], row["transition"])].append(
            float(row["matched_minus_other"])
        )
    wanted = ["boundary_gained", "boundary_lost", "boundary_stable",
              "split_boundary", "merge_boundary"]
    for index, feature in enumerate(wanted):
        values = [np.mean(v) for (f, _, _), v in cells.items() if f == feature]
        if not values:
            continue
        jitter = (np.random.default_rng(index).random(len(values)) - 0.5) * 0.26
        ax.scatter(np.full(len(values), index) + jitter, values, s=16, alpha=0.75,
                   color="#4c72b0", edgecolor="none")
        ax.plot([index - 0.22, index + 0.22], [np.mean(values)] * 2,
                color="black", linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(wanted)))
    ax.set_xticklabels([n.replace("boundary_", "").replace("_boundary", "")
                        for n in wanted], rotation=20, ha="right")
    ax.set_ylabel("matched minus other\n(mean |log2 FC|)")
    ax.set_title("e  Transition specificity is weak\nand carried by one bin size",
                 loc="left", fontsize=10)


def panel_chrx(ax):
    rows = read_tsv(RESULTS / "chrx" / "chrx_boundary_rates.tsv")
    positions = np.arange(len(STAGE_TRANSITIONS))
    for offset, resolution in enumerate(RESOLUTIONS):
        values = []
        for transition in STAGE_TRANSITIONS:
            subset = [float(r["ratio"]) for r in rows
                      if r["transition"] == transition and r["change"] == "lost"
                      and r["resolution"] == resolution and r["ratio"]]
            values.append(np.mean(subset) if subset else np.nan)
        ax.plot(positions, values, marker="o", markersize=4,
                color=RESOLUTION_COLORS[resolution], label=resolution)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT[t] for t in STAGE_TRANSITIONS])
    ax.set_ylabel("chrX boundary loss /\nautosome mean")
    ax.set_ylim(0, 1.6)
    ax.set_title("f  chrX does not shed boundaries\nfaster than the autosomes",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(RESULTS / "figures" / "summary.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for panel, ax in zip(
        (panel_turnover, panel_event_mix, panel_reproducibility,
         panel_density, panel_specificity, panel_chrx),
        axes.ravel(),
    ):
        panel(ax)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
