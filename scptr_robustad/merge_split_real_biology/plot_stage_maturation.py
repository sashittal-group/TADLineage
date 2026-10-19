#!/usr/bin/env python3
"""The stage trajectory of boundary CTCF and gene density, as one four-panel figure.

a. CTCF density at the boundaries of each stage median, against relocated boundaries of
   the same median, one line per bin size. The published prediction is a rise across the
   chain, and the panel is drawn so that a flat line is the null result.
b. the same for gene density, which is an independent annotation of the same boundaries.
c. the per-chromosome difference between the 64-cell and 2-cell folds, which is where the
   test lives: chromosomes are the replicates and the claim is that the points sit above
   zero.
d. the monotone trend from the 2-cell stage on, under each of the three relocation pools,
   which is the check that the rise is not an artefact of where the caller looked.

Usage:
    python plot_stage_maturation.py
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
from stage_boundary_maturation import STAGE_ORDER, TREND_STAGES  # noqa: E402

FEATURE_LABEL = {"ctcf": "CTCF", "genes": "gene"}

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESOLUTIONS = ["250kb", "500kb", "1000kb"]
# bin size is drawn in one hue at three depths, deliberately not blue or orange,
# because those two carry the caller in the trajectory panels of the same figure
COLORS = {"250kb": "#4a3b76", "500kb": "#7d6bab", "1000kb": "#b3a6d4"}
SHORT = {s: s.replace("-cell", "") for s in STAGE_ORDER}

# The trajectory panels carry two callers at three bin sizes. Colour is the caller, and it
# means the same thing here as in the boundary-class panels so that the two can sit in one
# figure; the bin size is the line style. The absolute level differs between callers
# because each places its boundaries differently -- the claim is the shape, not the height.
CALLER_COLORS = {"RobusTAD": "#4c72b0", "OnTAD": "#dd8452"}
CALLER_TABLES = {"RobusTAD": "stage", "OnTAD": "stage_ontad"}
STYLES = {"250kb": "-", "500kb": "--", "1000kb": ":"}
MARKERS = {"250kb": "o", "500kb": "s", "1000kb": "^"}
TITLE_SIZE = 10
LEGEND_SIZE = 8


def per_chromosome(rows, feature, pool):
    """``{(resolution, chrom): {stage: fold}}`` with the tolerances averaged."""
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        if row["feature"] != feature or row["pool"] != pool or row["fold"] == "":
            continue
        buckets[(row["resolution"], row["chrom"])][row["stage"]].append(float(row["fold"]))
    return {key: {stage: float(np.mean(v)) for stage, v in stages.items()}
            for key, stages in buckets.items()}


def panel_trajectory(ax, rows_by_caller, feature, letter, title, legend=True):
    """The stage trajectory of one feature, both callers, three bin sizes each."""
    positions = np.arange(len(STAGE_ORDER))
    for caller, rows in rows_by_caller.items():
        if not rows:
            continue
        cells = per_chromosome(rows, feature, "union")
        for resolution in RESOLUTIONS:
            subset = [c for (res, _), c in cells.items() if res == resolution]
            if not subset:
                continue
            means, errors = [], []
            for stage in STAGE_ORDER:
                values = [c[stage] for c in subset if stage in c]
                means.append(np.mean(values) if values else np.nan)
                errors.append(np.std(values, ddof=1) / np.sqrt(len(values))
                              if len(values) > 1 else 0.0)
            ax.errorbar(positions, means, yerr=errors, marker=MARKERS[resolution],
                        markersize=3.5, capsize=1.5, linewidth=1.2, elinewidth=0.8,
                        linestyle=STYLES[resolution], color=CALLER_COLORS[caller])
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT[s] for s in STAGE_ORDER])
    ax.set_xlabel("stage")
    ax.set_ylabel(f"{FEATURE_LABEL.get(feature, feature)} density at boundaries\n"
                  "(fold over relocated boundaries)")
    ax.set_title(f"{letter}  {title}" if letter else title, loc="left",
                 fontsize=TITLE_SIZE)
    if legend:
        trajectory_legend(ax, rows_by_caller)


def trajectory_legend(ax, rows_by_caller):
    """Colour is the caller, line style is the bin size, so the key has two halves."""
    from matplotlib.lines import Line2D

    drawn = [caller for caller in rows_by_caller if rows_by_caller[caller]]
    sizes = [Line2D([], [], color="0.35", linewidth=1.2, linestyle=STYLES[r],
                    marker=MARKERS[r], markersize=3.5, label=r)
             for r in RESOLUTIONS]
    if len(drawn) < 2:
        # one caller means colour distinguishes nothing, so naming it in the key only
        # invites the height to be read as a result; the bin sizes are the whole legend
        ax.legend(handles=sizes, fontsize=LEGEND_SIZE, frameon=False,
                  loc="lower right", handlelength=1.8)
        return
    callers = [Line2D([], [], color=CALLER_COLORS[caller], linewidth=1.6, label=caller)
               for caller in drawn]
    # matplotlib fills a multi-column legend column by column, so pad the caller column
    # to the height of the bin-size column; otherwise a bin size lands under a caller
    blank = Line2D([], [], color="none", label=" ")
    while len(callers) < len(sizes):
        callers.append(blank)
    ax.legend(handles=callers + sizes, fontsize=LEGEND_SIZE, frameon=False, ncol=2,
              loc="lower right", handlelength=1.8, columnspacing=1.2)


def panel_deltas(ax, rows):
    cells = per_chromosome(rows, "ctcf", "union")
    rng = np.random.default_rng(0)
    counts = []
    for offset, resolution in enumerate(RESOLUTIONS):
        deltas = [c["64-cell"] - c["2-cell"]
                  for (res, _), c in cells.items()
                  if res == resolution and "64-cell" in c and "2-cell" in c]
        jitter = rng.uniform(-0.12, 0.12, len(deltas))
        ax.scatter(np.full(len(deltas), offset) + jitter, deltas, s=14,
                   color=COLORS[resolution], alpha=0.75, edgecolors="none")
        ax.hlines(np.mean(deltas), offset - 0.25, offset + 0.25, color="black",
                  linewidth=1.6)
        counts.append((offset, sum(1 for d in deltas if d > 0), len(deltas)))
    # the label row is placed once the axis has seen every point, so the three
    # annotations sit at one height instead of climbing with the autoscale
    top = ax.get_ylim()[1]
    ax.set_ylim(top=top + 0.06 * (top - ax.get_ylim()[0]))
    for offset, above, total in counts:
        ax.text(offset, top, f"{above}/{total} above zero", ha="center", va="bottom",
                fontsize=7)
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(RESOLUTIONS)))
    ax.set_xticklabels(RESOLUTIONS)
    ax.set_ylabel("CTCF fold, 64-cell minus 2-cell\n(one point per chromosome)")
    ax.set_title("c  The rise holds on almost every chromosome", loc="left", fontsize=10)


def panel_pools(ax, tests):
    pools = ["union", "own", "common"]
    width = 0.26
    positions = np.arange(len(pools))
    for offset, resolution in enumerate(RESOLUTIONS):
        values = []
        for pool in pools:
            match = [r for r in tests
                     if r["resolution"] == resolution and r["pool"] == pool
                     and r["feature"] == "ctcf"
                     and r["statistic"] == "spearman rho over 2->64"]
            values.append(float(match[0]["mean"]) if match else 0.0)
        bars = ax.bar(positions + (offset - 1) * width, values, width,
                      color=COLORS[resolution], label=resolution)
        for pool, bar in zip(pools, bars):
            match = [r for r in tests
                     if r["resolution"] == resolution and r["pool"] == pool
                     and r["feature"] == "ctcf"
                     and r["statistic"] == "spearman rho over 2->64"]
            if match and float(match[0]["wilcoxon_p"]) <= 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        "*", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{p}\npool" for p in pools])
    ax.set_ylabel("mean rho over " + "→".join(SHORT[s] for s in TREND_STAGES))
    ax.set_title("d  Not an artefact of where the caller looked", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(RESULTS / "figures" / "stage_maturation.png"))
    args = parser.parse_args()

    rows_by_caller = {
        caller: read_tsv(RESULTS / table / "stage_boundary_enrichment.tsv")
        for caller, table in CALLER_TABLES.items()}
    rows = rows_by_caller["RobusTAD"]
    tests = read_tsv(RESULTS / "stage" / "stage_trend_tests.tsv")
    if not rows:
        print("No results/stage tables. Run stage_boundary_maturation.py first.")
        return

    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    panel_trajectory(axes[0][0], rows_by_caller, "ctcf", "a",
                     "Boundaries gain CTCF after the 2-cell stage")
    panel_trajectory(axes[0][1], rows_by_caller, "genes", "b",
                     "Gene density follows the same path")
    panel_deltas(axes[1][0], rows)
    panel_pools(axes[1][1], tests)
    figure.tight_layout()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
