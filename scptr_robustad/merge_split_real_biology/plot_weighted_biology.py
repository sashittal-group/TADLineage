#!/usr/bin/env python3
"""The alpha sweep and the boundary-turnover asymmetry, in one figure.

Top row, one panel per readout: what happens to the biological readout as the ranking of
merge/split calls is slid from our own held-out evidence (alpha = 0) to the CTCF/cohesin
anchor (alpha = 1). The grey band is the permutation null at 95 per cent and the dashed
lines are the 5th and 95th percentiles of the curve rebuilt with the structural ranking
shuffled inside each chromosome, which is what the arithmetic alone produces. A result
would be a curve leaving both bands somewhere strictly inside the interval.

Bottom row: log2 of the feature density at the boundaries a transition gains over the
density at the boundaries it loses, one point per (caller, bin size), with the sign test
over chromosomes behind each point. Gene density is the row to read first -- every active
track moves with it -- and H3K9me3 is the control that stays flat.

Usage:
    python plot_weighted_biology.py
    python plot_weighted_biology.py --caller OnTAD --resolution 500kb --top-k 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEIGHTED = HERE / "results" / "weighted"
TERRITORY = HERE / "results" / "territory" / "turnover_asymmetry.tsv"
FIGURES = HERE / "results" / "figures"

# categorical slots 1 and 2 of the reference palette, which validate all-pairs in both
# modes; every series is direct labelled as well, so identity is never colour alone
CALLER_COLOUR = {"OnTAD": "#2a78d6", "RobusTAD": "#eb6834"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedddA"

READOUT_TITLE = {
    "replication": "replication in held-out cells",
    "dynamism": "transcriptional dynamism at the boundary",
    "crossboundary": "divergence across the boundary",
}
READOUT_Y = {
    "replication": "mean held-out $z$ of the selected set",
    "dynamism": "matched $-$ other dynamism percentile",
    "crossboundary": "matched $-$ other divergence change",
}
TRANSITION_LABEL = ["1$\\rightarrow$2", "2$\\rightarrow$4", "4$\\rightarrow$8",
                    "8$\\rightarrow$64"]
TRACKS = [("gene_density", "gene density"), ("ctcf", "CTCF"), ("h3k9me3", "H3K9me3")]


def read(path):
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def alpha_panel(ax, curve, controls, readout, title_index):
    rows = sorted(curve, key=lambda r: number(r["alpha"]))
    alpha = np.array([number(r["alpha"]) for r in rows])
    observed = np.array([number(r["contrast"]) for r in rows])
    centre = np.array([number(r["null_mean"]) for r in rows])
    spread = np.array([number(r["null_sd"]) for r in rows])

    ax.fill_between(alpha, centre - 1.96 * spread, centre + 1.96 * spread,
                    color=GRID, alpha=0.85, linewidth=0, zorder=1,
                    label="permutation null, 95%")
    shuffled = [r for r in controls if r["control"] == "shuffled-S"]
    if shuffled:
        shuffled.sort(key=lambda r: number(r["alpha"]))
        low = np.array([number(r["null_p05"]) for r in shuffled])
        high = np.array([number(r["null_p95"]) for r in shuffled])
        ax.plot(alpha, low, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2,
                label="shuffled ranking, 5-95%")
        ax.plot(alpha, high, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.plot(alpha, observed, color=INK, linewidth=2.0, zorder=3, label="observed")
    ax.plot(alpha, observed, "o", color=INK, markersize=4.5, zorder=4,
            markeredgecolor="white", markeredgewidth=1.0)

    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("$\\alpha$   (0 = our evidence,  1 = CTCF + cohesin)", fontsize=8,
                  color=MUTED)
    ax.set_ylabel(READOUT_Y[readout], fontsize=8, color=MUTED)
    ax.set_title(f"{'abc'[title_index]}  {READOUT_TITLE[readout]}", fontsize=9.5,
                 color=INK, loc="left", pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=MUTED, length=3)


def territory_panel(ax, rows, track, label, title_index, resolutions):
    transitions = sorted({r["transition"] for r in rows},
                         key=lambda t: ["1-cell", "2-cell", "4-cell", "8-cell"].index(
                             t.split("->")[0]))
    offsets = np.linspace(-0.18, 0.18, len(resolutions))
    ax.axhline(0.0, color=MUTED, linewidth=1.0, zorder=1)
    seen = set()
    for caller, colour in CALLER_COLOUR.items():
        for k, resolution in enumerate(resolutions):
            xs, ys, filled = [], [], []
            for i, transition in enumerate(transitions):
                match = [r for r in rows if r["caller"] == caller
                         and r["resolution"] == resolution
                         and r["transition"] == transition and r["track"] == track]
                if not match:
                    continue
                shift = -0.09 if caller == "OnTAD" else 0.09
                xs.append(i + shift + offsets[k] * 0.5)
                ys.append(number(match[0]["mean_log2_ratio"]))
                filled.append(number(match[0]["sign_p"]) < 0.05)
            if not xs:
                continue
            xs, ys = np.array(xs), np.array(ys)
            filled = np.array(filled)
            ax.scatter(xs[filled], ys[filled], s=34, color=colour, zorder=3,
                       edgecolors="white", linewidths=1.0,
                       label=caller if caller not in seen else None)
            ax.scatter(xs[~filled], ys[~filled], s=34, facecolors="white", zorder=3,
                       edgecolors=colour, linewidths=1.4)
            seen.add(caller)

    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(TRANSITION_LABEL[:len(transitions)], fontsize=8)
    ax.set_xlim(-0.5, len(transitions) - 0.5)
    ax.set_ylabel("$\\log_2$(gained / lost)", fontsize=8, color=MUTED)
    ax.set_xlabel("transition (cells)", fontsize=8, color=MUTED)
    ax.set_title(f"{'def'[title_index]}  {label}", fontsize=9.5, color=INK,
                 loc="left", pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=MUTED, length=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caller", default="OnTAD")
    parser.add_argument("--resolution", default="500kb")
    parser.add_argument("--event-type", default="merge+split")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out")
    args = parser.parse_args()

    curve_rows = read(WEIGHTED / args.caller / "alpha_curve.tsv")
    control_rows = read(WEIGHTED / args.caller / "alpha_controls.tsv")
    territory_rows = read(TERRITORY)
    territory_rows = [r for r in territory_rows if r["classes"] == "all"]
    resolutions = ["250kb", "500kb", "1000kb"]

    def pick(rows, readout):
        return [r for r in rows if r["readout"] == readout
                and r["resolution"] == args.resolution
                and r["event_type"] == args.event_type
                and int(r["top_k"]) == args.top_k]

    readouts = [r for r in ("replication", "dynamism", "crossboundary")
                if pick(curve_rows, r)]
    if not readouts:
        raise SystemExit(f"no curve for {args.caller} {args.resolution} "
                         f"{args.event_type} K={args.top_k}")

    figure, axes = plt.subplots(2, 3, figsize=(11.4, 6.9))
    figure.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")

    for i, readout in enumerate(readouts):
        alpha_panel(axes[0, i], pick(curve_rows, readout), pick(control_rows, readout),
                    readout, i)
    for i in range(len(readouts), 3):
        axes[0, i].axis("off")
    for i, (track, label) in enumerate(TRACKS):
        territory_panel(axes[1, i], territory_rows, track, label, i, resolutions)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, fontsize=7.5, frameon=False, loc="best",
                      labelcolor=MUTED)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend(handles, labels, fontsize=7.5, frameon=False, loc="best",
                      title="filled = sign test $p<0.05$", title_fontsize=7,
                      labelcolor=MUTED)

    figure.text(0.012, 0.972,
                f"Weighting merge/split calls toward the biology  \u2014  {args.caller}, "
                f"{args.resolution}, {args.event_type}, top {args.top_k}",
                fontsize=11, color=INK, ha="left", va="top")
    figure.text(0.012, 0.455,
                "Boundary turnover: the territory each transition trades  \u2014  "
                "both callers, three bin sizes",
                fontsize=11, color=INK, ha="left", va="top")
    figure.subplots_adjust(left=0.065, right=0.985, top=0.895, bottom=0.075,
                           hspace=0.95, wspace=0.30)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else FIGURES / "weighted_biology.png"
    figure.savefig(out, dpi=200, facecolor=figure.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
