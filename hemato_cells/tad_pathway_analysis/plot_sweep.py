#!/usr/bin/env python3
"""Does the per-edge biology survive changing the resolution, the tolerance and the caller?

One cell per edge and configuration, shaded by how many dispersed haematopoietic gene sets
that edge carries there -- the same count ``per_edge_biology.py`` prints, split by
configuration instead of pooled.  A row that is shaded across the whole sweep is a result;
a row with one dark cell is a configuration that happened to work.

Usage:
    python plot_sweep.py
    python plot_sweep.py --q-max 0.05 --out figures/sweep_strict.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_dotplot import in_focus, tandem  # noqa: E402
from per_edge_biology import EDGES  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CALLER_NAME = {"ontad": "OnTAD", "robustad": "RobusTAD"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=str(RESULTS / "enrichment_grid.tsv"))
    ap.add_argument("--q-max", type=float, default=0.10)
    ap.add_argument("--min-overlap", type=int, default=10)
    ap.add_argument("--min-fold", type=float, default=1.5)
    ap.add_argument("--out")
    args = ap.parse_args()

    with open(args.grid) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    counts: dict = defaultdict(set)
    configs: list = []
    for r in rows:
        config = (r["caller"], r["resolution"], int(r["tolerance"]))
        if config not in configs:
            configs.append(config)
        if (float(r["q"]) <= args.q_max and int(r["overlap"]) >= args.min_overlap
                and float(r["fold"]) >= args.min_fold
                and in_focus(r["gene_set"]) and not tandem(r)):
            counts[(r["edge"], config)].add(r["gene_set"])

    configs.sort(key=lambda c: (c[0], c[1], c[2]))
    grid = np.array([[len(counts.get((edge, c), ())) for c in configs] for edge in EDGES],
                    float)
    # an edge the caller's tree does not have is absent, not zero
    for i, edge in enumerate(EDGES):
        for j, (caller, _res, _tol) in enumerate(configs):
            missing = (caller == "robustad" and edge in ("LT->ST", "ST->MPP")) or \
                      (caller == "ontad" and edge == "LT->MPP")
            if missing:
                grid[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(0.46 * len(configs) + 3.4, 0.42 * len(EDGES) + 2.6))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("0.94")
    image = ax.imshow(grid, cmap=cmap, vmin=0, aspect="auto")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([f"{c[1]} t{c[2]}" for c in configs], rotation=90, fontsize=7)
    ax.set_yticks(range(len(EDGES)))
    ax.set_yticklabels([e.replace("->", " → ") for e in EDGES], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(configs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(EDGES), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)

    for i in range(len(EDGES)):
        for j in range(len(configs)):
            value = grid[i, j]
            if np.isnan(value) or value == 0:
                continue
            ax.text(j, i, int(value), ha="center", va="center", fontsize=6.5,
                    color="white" if value > np.nanmax(grid) * 0.55 else "0.15")

    boundary = sum(1 for c in configs if c[0] == "ontad")
    ax.axvline(boundary - 0.5, color="0.25", lw=1.6)
    ax.text((boundary - 1) / 2, -0.85, CALLER_NAME["ontad"], ha="center", fontsize=10)
    ax.text((boundary + len(configs) - 1) / 2, -0.85, CALLER_NAME["robustad"],
            ha="center", fontsize=10)

    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("dispersed haematopoietic gene sets", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    ax.set_title("On-lineage enrichment per edge across the configuration grid",
                 fontsize=11, pad=34)
    fig.text(0.5, -0.02, "Grey cells are edges that caller's tree does not have "
             "(RobusTAD has no ST call).", ha="center", fontsize=7.5, color="0.35")

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else FIGURES / "sweep_on_lineage_by_edge.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
