#!/usr/bin/env python3
"""The two results of this folder, one single-panel figure each.

``ctcf_boundaries.png``
    CTCF and gene density by boundary class, each class scored against the stable
    boundaries of its own medians, so the bar for stable is one by construction.

``merge_split.png``
    The merge and split boundaries of the same medians, against the stable boundaries and
    against the other reorganised boundaries.

Both use one axis convention: every bar is a ratio between two boundary sets drawn from
the same pair of medians, and one is no difference. Significance is a two-sided exact sign
test over **chromosomes**, which is the only replicate here that is independent -- the six
tolerances of a bin size are re-analyses of one clustering and the three bin sizes are
re-analyses of one dataset, so both are averaged inside a chromosome instead of being
counted as replicates.

Two further panels are kept behind ``--all-panels`` rather than deleted: the contrast one
configuration at a time, and the cross-caller agreement on merge and split loci. Neither is
in the main figures.

Usage:
    python plot_boundary_results.py
    python plot_boundary_results.py --all-panels
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from summarise_results import read_tsv  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
CALLERS = ["RobusTAD", "OnTAD"]
CALLER_COLORS = {"RobusTAD": "#4c72b0", "OnTAD": "#dd8452"}
FEATURE_LABEL = {"ctcf": "CTCF", "genes": "genes"}
# matplotlib's default font has no thin space; keep labels plain ASCII
CLASS_LABEL = {"stable": "stable", "gainloss": "gain / loss", "ms": "merge / split"}

# These panels are printed at \textwidth in an 11pt document, which shrinks them to about
# four fifths of the size matplotlib draws them. The sizes live here so that ratio stays
# adjustable in one place rather than being buried in each panel.
TITLE_SIZE = 10
LEGEND_SIZE = 8
STAR_SIZE = 7.5
NOTE_SIZE = 7.5


def classes_table():
    rows = read_tsv(RESULTS / "turnover" / "boundary_classes.tsv")
    return {(r["caller"], r["feature"], r["contrast"]): r for r in rows}


SIGNIFICANCE_NOTE = ("sign test over chromosomes\n"
                     "***  p \u2264 0.001      **  p \u2264 0.01\n"
                     "*  p \u2264 0.05      n.s.  not significant")


def stars(p):
    p = float(p)
    return "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else "n.s."


def significance_legend(ax, x=0.5, y=0.985):
    """The star key, in the axes so the figure stands on its own."""
    ax.text(x, y, SIGNIFICANCE_NOTE, transform=ax.transAxes, ha="center", va="top",
            fontsize=NOTE_SIZE, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7",
                      linewidth=0.6))


def headroom(heights, floor, ceiling, keep):
    """Axis limits: the published ones when both callers are drawn, else fitted.

    Dropping a caller halves the bars but not the axis, and the panel then reads as
    mostly empty space with a legend floating in it. The top is refitted with room left
    for the stars and the two keys; the floor is kept so that one is still the reference
    the eye lands on.
    """
    if keep or not heights:
        return floor, ceiling
    # doubling the drawn span puts every bar top in the lower half of the axis, which is
    # what leaves the upper half free for the series legend and the star key
    return floor, max(heights) + (max(heights) - floor)


def panel_classes(ax, table, letter="", callers=None, drop_stable=False, title=None):
    """Each boundary class against the stable boundaries of its own medians.

    ``drop_stable`` removes the stable bars, which are one by construction; the dashed
    line at one is the same reference and carries it without spending a column.
    """
    groups = [("stable", None), ("gainloss", "gainloss / stable"), ("ms", "ms / stable")]
    if drop_stable:
        groups = groups[1:]
    callers = list(callers or CALLERS)
    # two bars per caller, one per feature; 0.8 of a slot shared between them, so the
    # four-bar figure keeps the width it had and a one-caller figure is not left with
    # two hairlines in the middle of an empty slot
    series = 2 * len(callers)
    width = 0.8 / series
    positions = np.arange(len(groups))
    drawn, offset = [], 0
    for caller in callers:
        for feature in ("ctcf", "genes"):
            heights, labels = [], []
            for name, contrast in groups:
                if contrast is None:
                    heights.append(1.0)
                    labels.append("")
                    continue
                row = table.get((caller, feature, contrast))
                heights.append(float(row["ratio"]) if row else np.nan)
                labels.append(stars(row["sign_p"]) if row else "")
            bars = ax.bar(positions + (offset - (series - 1) / 2) * width, heights, width,
                          color=CALLER_COLORS[caller],
                          alpha=1.0 if feature == "ctcf" else 0.55,
                          edgecolor="white", linewidth=0.6,
                          label=(f"{caller}, {FEATURE_LABEL[feature]}"
                                 if len(callers) > 1 else FEATURE_LABEL[feature]))
            drawn += [h for h in heights if np.isfinite(h)]
            for bar, text in zip(bars, labels):
                if text:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                            text, ha="center", va="bottom", fontsize=STAR_SIZE)
            offset += 1
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels([CLASS_LABEL[name] for name, _ in groups])
    ax.set_ylabel("density, over stable boundaries\nof the same medians")
    ax.set_ylim(*headroom(drawn, 0.9, 1.76, len(callers) == len(CALLERS)))
    title = title or "Reorganised boundaries are CTCF and gene dense"
    ax.set_title(f"{letter}  {title}" if letter else title,
                 loc="left", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))
    significance_legend(ax, x=0.70,
                        y=0.80 if len(callers) == len(CALLERS) else 0.88)


def panel_per_configuration(ax, turnover):
    """The contrast one configuration at a time, which is what the sign test counts."""
    rows = [r for r in turnover
            if r["denominator"] == "boundary_stable"
            and r["numerator"] in ("boundary_gained", "boundary_lost")]
    labels, heights, colors, notes = [], [], [], []
    for caller in CALLERS:
        for numerator in ("boundary_gained", "boundary_lost"):
            match = [r for r in rows
                     if r["caller"] == caller and r["numerator"] == numerator]
            if not match:
                continue
            row = match[0]
            labels.append(f"{caller}\n{numerator.replace('boundary_', '')}")
            heights.append(float(row["ratio"]))
            colors.append(CALLER_COLORS[caller])
            notes.append(f"{row['above_one']}/{row['configurations']}")
    positions = np.arange(len(labels))
    bars = ax.bar(positions, heights, 0.6, color=colors, edgecolor="white", linewidth=0.6)
    for bar, note in zip(bars, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, note,
                ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("CTCF at changed boundaries,\nover stable boundaries")
    ax.set_ylim(0.98, 1.25)
    ax.set_title("b  Every configuration of both callers agrees\n"
                 "     (configurations above one shown)", loc="left", fontsize=9)


def panel_merge_split(ax, table, letter="", callers=None):
    contrasts = [("ms / stable", "vs stable"), ("ms / gainloss", "vs other\nreorganised")]
    callers = list(callers or CALLERS)
    series = 2 * len(callers)
    width = 0.8 / series
    positions = np.arange(len(contrasts))
    drawn, offset = [], 0
    for caller in callers:
        for feature in ("ctcf", "genes"):
            heights, labels = [], []
            for contrast, _ in contrasts:
                row = table.get((caller, feature, contrast))
                heights.append(float(row["ratio"]) if row else np.nan)
                labels.append(stars(row["sign_p"]) if row else "")
            bars = ax.bar(positions + (offset - (series - 1) / 2) * width, heights, width,
                          color=CALLER_COLORS[caller],
                          alpha=1.0 if feature == "ctcf" else 0.55,
                          edgecolor="white", linewidth=0.6,
                          label=(f"{caller}, {FEATURE_LABEL[feature]}"
                                 if len(callers) > 1 else FEATURE_LABEL[feature]))
            drawn += [h for h in heights if np.isfinite(h)]
            for bar, text in zip(bars, labels):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        text, ha="center", va="bottom", fontsize=STAR_SIZE)
            offset += 1
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in contrasts])
    ax.set_ylabel("merge / split boundary density,\nover the comparison set")
    ax.set_ylim(*headroom(drawn, 0.8, 1.78, len(callers) == len(CALLERS)))
    title = "Merge and split are not a separate class"
    ax.set_title(f"{letter}  {title}" if letter else title,
                 loc="left", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))
    significance_legend(ax, x=0.66,
                        y=0.80 if len(callers) == len(CALLERS) else 0.88)


def panel_cross_caller(ax, cross):
    if not cross:
        ax.set_visible(False)
        return
    cross = sorted(cross, key=lambda r: int(r["slop_bp"]))
    positions = np.arange(len(cross))
    width = 0.36
    observed = [int(r["matched"]) for r in cross]
    null = [float(r["null_mean"]) for r in cross]
    ax.bar(positions - width / 2, observed, width, color="#55a868", label="observed")
    ax.bar(positions + width / 2, null, width, color="0.75", label="loci slid at random")
    for x, row in zip(positions, cross):
        ax.text(x, max(int(row["matched"]), float(row["null_mean"])) + 1.0,
                f"{float(row['fold']):.2f}x\np={float(row['p_greater']):.2f}",
                ha="center", va="bottom", fontsize=6)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(r['slop_bp']) / 1e6:g} Mb" for r in cross])
    ax.set_xlabel("distance allowed between two callers' loci")
    ax.set_ylabel("OnTAD merge / split loci\nmatching a RobusTAD locus")
    ax.set_ylim(0, max(max(observed), max(null)) * 1.45)
    ax.set_title("d  The callers do not agree on where the merges\n"
                 "     and splits are", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-panels", action="store_true",
                        help="also write the per-configuration and cross-caller panels")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    table = classes_table()
    turnover = read_tsv(RESULTS / "turnover" / "boundary_turnover.tsv")
    cross = read_tsv(RESULTS / "turnover" / "cross_caller.tsv")
    if not table:
        print("No results/turnover tables. Run boundary_turnover.py first.")
        return

    out_dir = Path(args.out_dir or FIGURES)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=(5.4, 4.0))
    panel_classes(ax, table)
    figure.tight_layout()
    first = out_dir / "ctcf_boundaries.png"
    figure.savefig(first, dpi=300)
    print(f"wrote {first}")

    figure, ax = plt.subplots(figsize=(5.0, 4.0))
    panel_merge_split(ax, table)
    figure.tight_layout()
    second = out_dir / "merge_split.png"
    figure.savefig(second, dpi=300)
    print(f"wrote {second}")

    if args.all_panels:
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
        panel_per_configuration(axes[0], turnover)
        panel_cross_caller(axes[1], cross)
        figure.tight_layout()
        extra = out_dir / "boundary_supporting.png"
        figure.savefig(extra, dpi=300)
        print(f"wrote {extra}")


if __name__ == "__main__":
    main()
