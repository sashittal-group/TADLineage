#!/usr/bin/env python3
"""Figures of the ablation from summary.tsv and per_start.tsv.

figures/ablation_genome_wide.png   (a) TAD-only starts, (b) the paper's start set, as bars per bin
                                   size for the four arms {imputed, raw} x {median, k-means};
                                   (c) the ARI of every TAD-only start, to show the landscape
figures/ablation_per_chromosome.png the same four arms per chromosome (chromosome-only fits)

Reference lines: the contact graph alone (the spectral start, no TADs) and the best of
scHiCluster / Higashi / scHiCPTR at that bin size, read from their summaries as in the paper.

    python plot.py
"""

import csv
import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import HERE, LABEL, REPO_ROOT, RESOLUTIONS  # noqa: E402

ARMS = [("imputed", "median", "imputed + median (paper)"),
        ("imputed", "kmeans", "imputed + k-means (no median)"),
        ("raw", "median", "no imputation + median"),
        ("raw", "kmeans", "no imputation + k-means")]
COLOR = {"imputed": "#2a78d6", "raw": "#eb6834"}
CONTACT_GRAY = "#8a8a85"
RULE_TITLE = {"tad_only": "TAD-only starts (20 k-means++ under the TAD distance)",
              "paper_starts": "paper's start set (spectral + 4 k-means++)"}


def read(file):
    with open(HERE / file) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def baselines():
    """Best genome-wide ARI and per-chromosome ARIs of the three published methods per bin size."""
    spec = importlib.util.spec_from_file_location("_ari", REPO_ROOT / "scptr_robustad/plot_ari_comparison.py")
    ari = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ari)
    key = {250_000: "250kb", 500_000: "500kb", 1_000_000: "1000kb"}
    best = {}
    for name, path, kind, _ in ari.METHODS[1:]:
        genome, _ = ari.read(path, kind)
        for r in RESOLUTIONS:
            if best.get(r, (-1,))[0] < genome.get(key[r], -1):
                best[r] = (genome[key[r]], name)
    return best


def arm_style(hierarchy, method):
    filled = method == "median"
    return dict(color=COLOR[hierarchy] if filled else "white", edgecolor=COLOR[hierarchy],
                hatch=None if filled else "////", linewidth=1.0)


def style(ax, title):
    ax.set_xticks(range(len(RESOLUTIONS)), [LABEL[r] for r in RESOLUTIONS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ARI vs embryo stage")
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="y", alpha=0.3, ls=":", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def genome_wide(summary, per_start, best):
    rows = [r for r in summary if r["scope"] == "all"]
    contact = next(float(r["ari"]) for r in rows if r["method"] == "spectral (contacts only)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    width = 0.19
    for ax, rule, letter in zip(axes[:2], ("tad_only", "paper_starts"), "ab"):
        for offset, (hierarchy, method, label) in enumerate(ARMS):
            values = [float(next(r["ari"] for r in rows if int(r["resolution"]) == res
                                 and r["hierarchy"] == hierarchy and r["method"] == method
                                 and r["rule"] == rule)) for res in RESOLUTIONS]
            x = np.arange(len(RESOLUTIONS)) + (offset - 1.5) * width
            ax.bar(x, values, width * 0.92, label=label, **arm_style(hierarchy, method))
            for xi, v in zip(x, values):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="0.2")
        ax.axhline(contact, color=CONTACT_GRAY, ls="--", lw=1.2,
                   label=f"contact graph alone, no TADs ({contact:.2f})")
        for i, res in enumerate(RESOLUTIONS):
            ax.hlines(best[res][0], i - 2 * width, i + 2 * width, color="black", lw=1.6,
                      label="best of scHiCluster / Higashi / scHiCPTR" if i == 0 else None)
        style(ax, f"({letter}) genome-wide, {RULE_TITLE[rule]}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3, frameon=False)

    ax = axes[2]
    rng = np.random.default_rng(0)
    for offset, (hierarchy, method, label) in enumerate(ARMS):
        for i, res in enumerate(RESOLUTIONS):
            values = [float(r["ari"]) for r in per_start if r["scope"] == "all"
                      and int(r["resolution"]) == res and r["hierarchy"] == hierarchy
                      and r["method"] == method and r["start"] != "spectral"]
            x = i + (offset - 1.5) * width
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(values)), values, s=12,
                       color=COLOR[hierarchy], marker="o" if method == "median" else "^",
                       alpha=0.75, linewidths=0, zorder=3)
            ax.hlines(np.median(values), x - 0.07, x + 0.07, color="black", lw=1.2, zorder=4)
    ax.axhline(contact, color=CONTACT_GRAY, ls="--", lw=1.2)
    style(ax, "(c) every TAD-only start (genome-wide; bar = median)")
    handles = [plt.Line2D([], [], color=COLOR[h], marker="o" if m == "median" else "^", ls="", label=l)
               for h, m, l in ARMS]
    ax.legend(handles=handles, fontsize=7.5, loc="lower left", frameon=True, framealpha=0.95)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(HERE / "figures/ablation_genome_wide.png", dpi=200)
    fig.savefig(HERE / "figures/ablation_genome_wide.pdf")


def per_chromosome(summary):
    rows = [r for r in summary if r["scope"] != "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    width = 0.15
    rng = np.random.default_rng(0)
    for ax, rule, letter in zip(axes, ("tad_only", "paper_starts"), "ab"):
        arms = ARMS + [("none", "spectral (contacts only)", "contact graph alone (spectral start)")]
        for offset, (hierarchy, method, label) in enumerate(arms):
            values = [[float(r["ari"]) for r in rows if int(r["resolution"]) == res
                       and r["hierarchy"] == hierarchy and r["method"] == method
                       and (r["rule"] == rule or hierarchy == "none")] for res in RESOLUTIONS]
            x = np.arange(len(RESOLUTIONS)) + (offset - 2) * width
            box = ax.boxplot(values, positions=x, widths=width * 0.85, patch_artist=True,
                             manage_ticks=False, showfliers=False,
                             medianprops=dict(color="black", lw=1.0))
            color = COLOR.get(hierarchy, CONTACT_GRAY)
            for patch in box["boxes"]:
                patch.set_facecolor(matplotlib.colors.to_rgba(color, 0.5 if method != "kmeans" else 0.12))
                patch.set_edgecolor(color)
                if method == "kmeans":
                    patch.set_hatch("////")
            for xi, v in zip(x, values):
                ax.scatter(xi + rng.uniform(-0.3, 0.3, len(v)) * width, v, s=5, color=color,
                           alpha=0.6, linewidths=0, zorder=3)
            ax.bar([np.nan], [np.nan], color=matplotlib.colors.to_rgba(color, 0.5 if method != "kmeans" else 0.12),
                   edgecolor=color, hatch="////" if method == "kmeans" else None, label=label)
        style(ax, f"({letter}) single chromosome, {RULE_TITLE[rule]}")
    axes[0].legend(fontsize=7.5, loc="upper right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(HERE / "figures/ablation_per_chromosome.png", dpi=200)
    fig.savefig(HERE / "figures/ablation_per_chromosome.pdf")


def main():
    (HERE / "figures").mkdir(exist_ok=True)
    summary, per_start = read("summary.tsv"), read("per_start.tsv")
    genome_wide(summary, per_start, baselines())
    per_chromosome(summary)


if __name__ == "__main__":
    main()
