#!/usr/bin/env python3
"""
plot_genome_wide_best_tolerance.py

Column chart of the quartet consistency of the genome-wide tree (genome_tree.py,
one tree for all chromosomes) at its best tolerance, per resolution. The tolerance
(or tolerances, when tied) that reaches it is printed, not drawn.

Reads <json-dir>/<weighting>_<res>_<tol>*.json, one resolution per run. Every
resolution uses all TADs except 100 kb, where all TADs do not fit in memory and the
runs keep each cell's top 25% of OnTAD calls per chromosome (TOP_FRACTION).

Usage (from hemato_cells/large_parsimony):
    python plot_genome_wide_best_tolerance.py

The paper figure (with the quartet schematic as panel a) is drawn by
TAD_Reorganization_latex/make_hemato_large_parsimony_genome_wide.py, which reuses load().
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES_ORDER = ["2mb", "1mb", "500kb", "250kb", "100kb"]
RES_LABEL = {"2mb": "2 Mb", "1mb": "1 Mb", "500kb": "500 kb", "250kb": "250 kb", "100kb": "100 kb"}
TOP_FRACTION = {"100kb": 0.25}   # all other resolutions: 1 (all TADs)
TOLS = range(6)

SERIES = "#2a78d6"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def load(json_dir, weighting):
    best = {}
    for path in glob.glob(os.path.join(json_dir, f"{weighting}_*.json")):
        with open(path) as fh:
            d = json.load(fh)
        if d["resolution"] not in RES_ORDER or d.get("top_fraction", 1.0) != TOP_FRACTION.get(d["resolution"], 1.0):
            continue
        best.setdefault(d["resolution"], {})[d["tolerance"]] = d["quartet_consistency"]
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", default="genome_wide/jsons")
    ap.add_argument("--weighting", default="ontad_score")
    ap.add_argument("--out", default="genome_wide/genome_wide_best_tolerance")
    ap.add_argument("--figsize", type=float, nargs=2, default=(5.8, 3.6), metavar=("W", "H"),
                    help="figure size in inches (the paper uses its printed size)")
    ap.add_argument("--font-size", type=float, default=10)
    args = ap.parse_args()

    per_res = load(args.json_dir, args.weighting)
    rows = []
    for res in RES_ORDER:
        vals = per_res.get(res, {})
        missing = [t for t in TOLS if t not in vals]
        if missing:
            raise SystemExit(f"{res}: missing tolerances {missing} in {args.json_dir}")
        top = max(vals.values())
        rows.append((res, top, [t for t, v in vals.items() if abs(v - top) < 1e-9]))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": args.font_size, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=200)
    fig.patch.set_facecolor("white")
    width = 0.42
    for i, (res, val, tols) in enumerate(rows):
        ax.bar(i, val, width=width, color=SERIES, linewidth=0, zorder=2)
        ax.text(i, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=args.font_size, color=INK, fontweight="bold")

    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(rows)), [RES_LABEL[r] for r, _, _ in rows])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Resolution", color=INK_2)
    ax.set_ylabel("Quartet consistency", color=INK_2)
    ax.set_title("Genome-wide tree", color=INK, fontsize=args.font_size + 1, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, length=0)
    for lab in ax.get_xticklabels():
        lab.set_color(INK_2)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", facecolor="white")
    print("resolution  best  tolerance(s)")
    for res, val, tols in rows:
        print(f"{res:>10}  {val:.3f}  {tols}")
    print(f"Saved {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
