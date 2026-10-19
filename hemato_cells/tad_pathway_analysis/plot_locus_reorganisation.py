#!/usr/bin/env python3
"""Which landmark gene clusters sit in reorganised TADs, edge by edge.

The pathway test reaches most of its smallest p-values through a handful of mouse tandem
arrays -- the chr13 Hist1 histone cluster, the H2 (MHC) locus, the alpha-defensins, the
keratins. Rather than discount those rows, this asks the question they are really answering,
directly and without a gene set in the way:

    for each landmark cluster and each edge, is the cluster covered by reorganised TADs more
    often than that edge's own genome-wide foreground rate would predict?

Per configuration the test is the same Fisher's exact test the pathway analysis uses, with
the cluster's genes standing in for a gene set. The figure reports the mean fold enrichment
over the configuration grid, and marks cells where most configurations reach p < 0.05.

The point of the figure is that the answer is edge-specific. A cluster that simply sits in
fragile chromatin would light up everywhere; the ones that do not are saying something about
the transition.

Usage:
    python plot_locus_reorganisation.py
    python plot_locus_reorganisation.py --max-domain-bp 1000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hypergeom

HERE = Path(__file__).resolve().parent
SCPTR = HERE.parent.parent / "scptr_robustad" / "merge_split_real_biology"
for path in (str(HERE), str(SCPTR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from annotate_events import GeneIndex  # noqa: E402
from gene_sets import load_gene_table  # noqa: E402
from pathway_enrichment import collect, genes_in  # noqa: E402
from reorganized_tads import edges_for  # noqa: E402

FIGURES = HERE / "figures"

# mm10 coordinates of the tandem arrays that drive the pathway table, plus the globin loci
# as lineage-specific controls.
LOCI = [
    ("Hist1 histone cluster",      "chr13",  21_700_000,  24_000_000),
    ("H2 / MHC",                   "chr17",  33_600_000,  38_400_000),
    ("Defa α-defensins",      "chr8",   19_900_000,  22_500_000),
    ("Gzm granzymes",              "chr14",  56_000_000,  57_200_000),
    ("Ifna interferons",           "chr4",   88_000_000,  88_900_000),
    ("Krt type II keratins",       "chr15", 101_200_000, 102_200_000),
    ("Krt type I keratins",        "chr11",  99_000_000, 100_600_000),
    ("Hbb β-globin",          "chr7",  103_700_000, 104_000_000),
    ("Hba α-globin",          "chr11",  32_100_000,  32_400_000),
]
EDGES = ["LT->ST", "LT->MPP", "ST->MPP", "MPP->CMP", "MPP->CLP", "CMP->MEP",
         "CMP->GMP", "MEP->MKP", "MKP->MK", "GMP->G"]
CALLERS = ("ontad", "robustad")
RESOLUTIONS = ("100kb", "250kb")
TOLERANCES = (1, 2, 3, 5, 8, 10)
MIN_GENES = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-domain-bp", type=int, default=2_000_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out")
    args = ap.parse_args()

    index = GeneIndex(load_gene_table())
    members = {name: set(index.in_window(chrom, lo, hi))
               for name, chrom, lo, hi in LOCI}

    folds: dict = {(locus, edge): [] for locus, *_ in LOCI for edge in EDGES}
    hits: dict = {key: 0 for key in folds}
    tried: dict = {key: 0 for key in folds}

    for caller in CALLERS:
        for resolution in RESOLUTIONS:
            for tolerance in TOLERANCES:
                changed, spans = collect(caller, resolution, tolerance,
                                         args.max_domain_bp)
                if not changed:
                    continue
                universe = genes_in(index, spans)
                n_universe = len(universe)
                for parent, child in edges_for(caller):
                    edge = f"{parent}->{child}"
                    foreground = genes_in(index, changed.get(edge, ()))
                    n_foreground = len(foreground)
                    if not n_foreground:
                        continue
                    rate = n_foreground / n_universe
                    for locus, *_ in LOCI:
                        genes = members[locus] & universe
                        if len(genes) < MIN_GENES:
                            continue
                        overlap = len(genes & foreground)
                        p = hypergeom.sf(overlap - 1, n_universe, len(genes),
                                         n_foreground)
                        folds[(locus, edge)].append((overlap / len(genes)) / rate)
                        tried[(locus, edge)] += 1
                        hits[(locus, edge)] += p < args.alpha

    names = [locus for locus, *_ in LOCI]
    grid = np.full((len(names), len(EDGES)), np.nan)
    marks = np.zeros_like(grid, bool)
    for i, locus in enumerate(names):
        for j, edge in enumerate(EDGES):
            values = folds[(locus, edge)]
            if not values:
                continue
            grid[i, j] = float(np.mean(values))
            marks[i, j] = tried[(locus, edge)] and \
                hits[(locus, edge)] / tried[(locus, edge)] >= 0.5

    fig, ax = plt.subplots(figsize=(9.6, 0.52 * len(names) + 3.0))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("0.94")
    image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=np.nanmax(grid), aspect="auto")

    ax.set_xticks(range(len(EDGES)))
    ax.set_xticklabels([e.replace("->", " → ") for e in EDGES],
                       rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}  ({len(members[n])} genes)" for n in names], fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(EDGES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="both", length=0)

    high = np.nanmax(grid) * 0.55
    for i in range(len(names)):
        for j in range(len(EDGES)):
            value = grid[i, j]
            if np.isnan(value):
                continue
            ink = "white" if value > high else "0.2"
            text = f"{value:.1f}" + ("*" if marks[i, j] else "")
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=ink)

    bar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    bar.set_label("fold coverage vs. the edge's own foreground rate", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    ax.set_title("Landmark gene clusters inside reorganised TADs, per lineage edge",
                 fontsize=11.5, pad=12)
    fig.text(0.5, 0.005,
             "Mean over 24 configurations (2 callers × 2 resolutions × 6 "
             "tolerances).  * = Fisher p < 0.05 in most configurations.  "
             "1.0 means the cluster is covered exactly as often as the genome at large.",
             ha="center", fontsize=7.5, color="0.35")

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else FIGURES / "locus_reorganisation_by_edge.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")

    print(f"\n{'locus':<26}" + "".join(f"{e.replace('->','>'):>11}" for e in EDGES))
    for i, locus in enumerate(names):
        print(f"{locus:<26}" + "".join(
            "          -" if np.isnan(grid[i, j]) else
            f"{grid[i, j]:>10.2f}{'*' if marks[i, j] else ' '}"
            for j in range(len(EDGES))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
