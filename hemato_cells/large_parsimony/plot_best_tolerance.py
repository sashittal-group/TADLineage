#!/usr/bin/env python3
"""
plot_best_tolerance.py

Bar charts of how often the large parsimony ILP recovers the correct quartet
{LT, CLP} | {leaf1, leaf2}, from aggregate_best_tolerance.py's per-instance CSV
(one instance = chromosome x resolution x leaf pair, each at its best tolerance).

Figures per analysis folder:
  correct_per_resolution.png  % correct instances for each resolution
  correct_per_leaf_pair.png   % correct instances for each leaf pair
  correct_genomewide.png      the same two breakdowns genome-wide, from the
                              genome-wide CSV (one instance = resolution x leaf pair,
                              cost summed over chromosomes, at its best tolerance)

Usage:
    python plot_best_tolerance.py --analysis ontad_weighted robustad_weighted
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES_ORDER = ["100kb", "250kb", "500kb", "1mb", "2mb"]
LEAF_ORDER = ["MEP+G", "MEP+GMP", "MK+G", "MK+GMP", "MKP+G", "MKP+GMP"]
TITLES = {
    "ontad_weighted": "OnTAD, weighted",
    "robustad_weighted": "RobusTAD, weighted",
    "ontad_unweighted": "OnTAD, unweighted",
}

BAR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def group_stats(df, key, order):
    g = df.groupby(key).agg(n=("correct_best_tol", "size"),
                            correct=("correct_best_tol", "sum"))
    return g.reindex([o for o in order if o in g.index])


def bar_panel(ax, stats, xlabel, title, subtitle, counts_on_top=False):
    x = range(len(stats))
    pct = 100 * stats["correct"] / stats["n"]
    ax.bar(x, pct, width=0.62, color=BAR, zorder=2)
    for i, (p, c, n) in enumerate(zip(pct, stats["correct"], stats["n"])):
        if counts_on_top:
            ax.text(i, p + 1.5, f"{int(c)}/{n}", ha="center", va="bottom", fontsize=10, color=INK)
            continue
        ax.text(i, p + 1.5, f"{p:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
        ax.text(i, 2, f"{int(c)}/{n}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stats.index)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel("correct instances (%)", color=MUTED)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=9, color=MUTED, ha="left", va="bottom")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def bar_figure(stats, xlabel, title, subtitle, out_path):
    fig, ax = plt.subplots(figsize=(1.1 * len(stats) + 2.2, 4.2))
    bar_panel(ax, stats, xlabel, title, subtitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def genomewide_figure(df, label, out_path):
    res = group_stats(df, "resolution", RES_ORDER)
    leaf = group_stats(df, "leaf_pair", LEAF_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(1.1 * (len(res) + len(leaf)) + 4.4, 4.2),
                             gridspec_kw={"width_ratios": [len(res) + 2, len(leaf) + 2]})
    bar_panel(axes[0], res, "resolution",
              f"{label}, genome-wide: per resolution",
              f"6 leaf pairs per bar", counts_on_top=True)
    bar_panel(axes[1], leaf, "leaf pair",
              f"{label}, genome-wide: per leaf pair",
              f"5 resolutions per bar", counts_on_top=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", nargs="+", default=["ontad_weighted", "robustad_weighted"])
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    for name in args.analysis:
        bt = os.path.join(here, name, "results", "best_tolerance")
        df = pd.read_csv(os.path.join(bt, "best_tolerance_per_instance.csv"))
        outdir = os.path.join(bt, "figures")
        os.makedirs(outdir, exist_ok=True)
        label = TITLES.get(name, name)
        n_chrom = df["chrom"].nunique()

        res = group_stats(df, "resolution", RES_ORDER)
        bar_figure(res, "resolution",
                   f"{label}: correct quartet per resolution",
                   f"{n_chrom} chromosomes x 6 leaf pairs per bar",
                   os.path.join(outdir, "correct_per_resolution.png"))

        leaf = group_stats(df, "leaf_pair", LEAF_ORDER)
        bar_figure(leaf, "leaf pair",
                   f"{label}: correct quartet per leaf pair",
                   f"{n_chrom} chromosomes x 5 resolutions per bar",
                   os.path.join(outdir, "correct_per_leaf_pair.png"))

        gw_csv = os.path.join(bt, "best_tolerance_genomewide_per_instance.csv")
        if os.path.exists(gw_csv):
            genomewide_figure(pd.read_csv(gw_csv), label,
                              os.path.join(outdir, "correct_genomewide.png"))
        print(f"{name}: wrote {outdir}")


if __name__ == "__main__":
    main()
