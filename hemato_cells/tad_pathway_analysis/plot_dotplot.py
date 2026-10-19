#!/usr/bin/env python3
"""Dotplot of the pathways over-represented among genes in reorganised TADs, per lineage edge.

One panel. Rows are gene sets, columns are the edges of the **OnTAD** tree, and each dot
combines the evidence from both TAD callers:

    dot area    overlap ratio -- the share of the set's genes inside reorganised TADs
    dot colour  Benjamini-Hochberg adjusted p, one hue light to dark, dark being significant
    dark ring   both callers reach q <= --q-max on that edge (no legend entry: this is
                described in the text rather than on the figure)

The colour and area are taken from whichever caller gives the smaller adjusted p, so a dot is
the best evidence for that set on that edge and the ring says whether the other caller agrees.

**Columns are OnTAD's nine edges.** RobusTAD has no ST node, so its stem is a single LT->MPP
edge with no OnTAD counterpart; it is dropped from the figure rather than folded into LT->ST
or ST->MPP, neither of which it corresponds to. LT->ST and ST->MPP are therefore OnTAD-only
columns and can never carry a ring. `per_edge_biology.py` still reports the RobusTAD stem.

Rows are grouped by the edge each set is most significant on, so edge-specific biology reads
down the diagonal. **Every set that clears the thresholds is shown; nothing is demoted.** A
dagger marks a set whose overlap is more than half inside three 2 Mb windows or one gene
family -- the leading-edge composition, reported the way a leading-edge gene list normally
is, not a judgement of the row. The MHC sets carry it because the MHC is one locus.

Size legend
-----------
The legend is **relative**, the convention the single-cell figure uses: the smallest dot is
labelled 0.01 and every other tick is scaled from it, so the legend reads 0.01, 0.02, 0.03 ...
whatever the raw overlap ratios happen to be. The raw ratios here are 0.35-1.00, because
reorganised TADs cover a large share of the genome; the single-cell run's were far smaller.
Labelling relative to the smallest dot puts the two figures on the same footing. The run
prints the raw ratio behind every tick. `--size-breaks` sets the relative labels directly.

Which rows are eligible
-----------------------
`--focus blood` (the default) keeps sets whose names match the haematopoietic and cell-cycle
vocabulary in `FOCUS_TOKENS`. GO BP alone holds over seven thousand sets and ranking them by
p without a stated hypothesis space fills the figure with tertiary alcohol metabolism. This
is a display choice: Benjamini-Hochberg still runs across every set of the collection inside
`pathway_enrichment.py`, so no p-value here was helped by it. `--focus all` drops it.

Usage:
    python plot_dotplot.py                                   # 100kb, tolerance 10
    python plot_dotplot.py --resolution 250kb --tolerance 5
    python plot_dotplot.py --size-breaks 0.4,0.6,0.8,1.0 --top 6
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# the columns: OnTAD's tree, down the stem and then each branch to its terminal cell
EDGES = ["LT->ST", "ST->MPP", "MPP->CMP", "MPP->CLP", "CMP->MEP",
         "CMP->GMP", "MEP->MKP", "MKP->MK", "GMP->G"]
# edges RobusTAD also has, so a ring is possible there
SHARED = {"MPP->CMP", "MPP->CLP", "CMP->MEP", "CMP->GMP", "MEP->MKP", "MKP->MK", "GMP->G"}
CALLERS = ("ontad", "robustad")
DEFAULT_COLLECTIONS = ["mh.all", "m2.cp.reactome", "m5.go.bp"]
SOURCE_TAG = {"mh.all": "H", "m2.cp.reactome": "R", "m5.go.bp": "GO", "m8.all": "C8"}

# The haematopoietic and cell-cycle vocabulary the figure is allowed to show.  Written from
# what the edges are -- stem maintenance, lineage commitment, terminal maturation -- not from
# the enrichment table.  Matched against whole underscore-separated tokens or a token's
# leading run: bare substring matching quietly accepts ENVIRONMENT for "IRON" and lets
# "DIFFERENTIATION" drag in keratinocytes.
FOCUS_TOKENS = {
    "LYMPHOCYTE", "LYMPHOID", "LYMPHOCYTES", "B", "T", "NK", "KILLER", "IMMUNOGLOBULIN",
    "ANTIGEN", "MHC", "MYELOID", "GRANULOCYTE", "NEUTROPHIL", "MONOCYTE", "MACROPHAGE",
    "MAST", "DEFENSIN", "DEFENSINS", "ERYTHROCYTE", "ERYTHROID", "HEMOGLOBIN", "HEME",
    "IRON", "PORPHYRIN", "MEGAKARYOCYTE", "PLATELET", "PLATELETS", "COAGULATION",
    "HEMOSTASIS", "LEUKOCYTE", "IMMUNE", "IMMUNITY", "CYTOKINE", "CHEMOKINE", "NFKB",
    "JAK", "STAT5", "STAT3", "NOTCH", "WNT", "BMP", "HEDGEHOG", "RUNX1", "GATA1", "IKAROS",
    "CHECKPOINT", "CHECKPOINTS", "MITOTIC", "APOPTOTIC", "QUIESCENCE", "HEMATOPOIETIC",
    "PROSTAGLANDIN", "METALLOPROTEINASES", "PROSTANOID", "THROMBOPOIETIN",
}
FOCUS_PREFIXES = ("INTERLEUKIN", "INTERFERON", "PHAGOCYT", "ANTIMICROB", "THROMB",
                  "INFLAMMATOR", "DEGRANULAT", "TOLL", "SELF")
FOCUS_PHRASES = ("CELL_CYCLE", "G2_M", "G1_S", "S_PHASE", "DNA_REPLICATION",
                 "STEM_CELL", "SELF_RENEWAL", "TGF_BETA", "DNA_DAMAGE")


def in_focus(name: str) -> bool:
    if any(phrase in name for phrase in FOCUS_PHRASES):
        return True
    return any(token in FOCUS_TOKENS or token.startswith(FOCUS_PREFIXES)
               for token in name.split("_"))


CMAP = "Blues_r"           # single hue, dark = small q
MAX_AREA = 230.0
MIN_AREA = 20.0
# The size legend is RELATIVE, the convention the single-cell figure uses: the smallest dot
# in the figure is labelled LEGEND_FLOOR and every other tick is scaled from it, so the
# legend reads 0.01, 0.02, 0.03 ... regardless of where the raw overlap ratios happen to sit.
# The raw ratios here are 0.35-1.00 because reorganised TADs cover a large share of the
# genome; the single-cell run's were far smaller. Labelling them relative to the smallest dot
# puts both figures on the same footing and keeps the legend to a few readable ticks.
LEGEND_FLOOR = 0.01
LEGEND_STEP = 0.01
CLUSTERED = 0.5            # the dagger threshold: an annotation, never a filter
DAGGER = "†"
RING = "0.25"

PRETTY = [(r"^HALLMARK_", ""), (r"^REACTOME_", ""), (r"^GOBP_", ""),
          (r"^TABULA_MURIS_SENIS_", ""), (r"_AGEING$", " (ageing)")]
SHORTEN = [
    ("Antigen Processing And Presentation", "Ag processing/presentation"),
    ("Peptide Antigen Assembly With", "Peptide Ag assembly,"),
    ("Runx1 Regulates Genes Involved In Megakaryocyte Differentiation And Platelet Function",
     "RUNX1 targets: MK differentiation"),
    ("Factors Involved In Megakaryocyte Development And Platelet Production",
     "MK development & platelet production"),
    ("Transcriptional Regulation By Runx1", "Transcriptional regulation by RUNX1"),
    ("Regulation Of", "reg. of"), ("Positive ", "+"), ("Negative ", "−"),
    ("Metabolic Process", "metabolism"), ("Involved In Immune Response", "(immune)"),
]


def prettify(name: str, collection: str, dagger: bool) -> str:
    label = name
    for pattern, replacement in PRETTY:
        label = re.sub(pattern, replacement, label)
    label = label.replace("_", " ").title()
    for word, fixed in (("Dna", "DNA"), ("Rna", "RNA"), ("Mhc", "MHC"), ("Bmp", "BMP"),
                        ("Tgfb", "TGF-beta"), ("G2 M", "G2/M"), ("Nk ", "NK "),
                        ("Il ", "IL-"), ("Class Ii", "class II"), ("Class Ib", "class Ib"),
                        ("Class I ", "class I "), ("Runx1", "RUNX1"), ("Mk ", "MK ")):
        label = label.replace(word, fixed)
    for long, short in SHORTEN:
        label = label.replace(long, short)
    label = re.sub(r"\s+", " ", label).strip()
    if len(label) > 48:
        label = label[:46].rstrip() + "…"
    return f"{label}{DAGGER if dagger else ''}  [{SOURCE_TAG.get(collection, '?')}]"


def read_grid(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def tandem(row: dict) -> bool:
    """Leading-edge annotation: is most of the overlap a few tandem arrays?"""
    return (float(row["cluster_frac"]) > CLUSTERED
            or float(row["locus_frac"]) > CLUSTERED
            or float(row["family_frac"]) > CLUSTERED)


def combine(rows, args):
    """``{(gene_set, collection): {edge: (best row, n callers supporting)}}``.

    Both callers are asked the same question on the same edge; the dot takes the smaller
    adjusted p and the ring records whether the other caller agreed.
    """
    keep = [r for r in rows
            if r["resolution"] == args.resolution
            and int(r["tolerance"]) == args.tolerance
            and r["edge"] in EDGES                      # drops RobusTAD's LT->MPP stem
            and r["collection"] in args.collection
            and float(r["q"]) <= args.q_max
            and int(r["overlap"]) >= args.min_overlap
            and float(r["fold"]) >= args.min_fold
            and (args.focus == "all" or in_focus(r["gene_set"]))]

    per_caller: dict = defaultdict(dict)
    for r in keep:
        key = (r["gene_set"], r["collection"], r["edge"])
        prior = per_caller[key].get(r["caller"])
        if prior is None or float(r["q"]) < float(prior["q"]):
            per_caller[key][r["caller"]] = r

    table: dict = defaultdict(dict)
    for (gene_set, collection, edge), by_caller in per_caller.items():
        best = min(by_caller.values(), key=lambda r: float(r["q"]))
        table[(gene_set, collection)][edge] = (best, len(by_caller))
    return table


def choose(table, args):
    by_edge = defaultdict(list)
    for key, per_edge in table.items():
        for edge, (row, _n) in per_edge.items():
            by_edge[edge].append((float(row["q"]), key))
    chosen = set()
    for edge in EDGES:
        for _q, key in sorted(by_edge.get(edge, []))[:args.top]:
            chosen.add(key)
    return chosen


def order_rows(chosen, table):
    """Group each set under the edge it is most significant on, so the figure reads diagonally."""
    placed = []
    for key in chosen:
        q, edge, _ = min((float(row["q"]), EDGES.index(edge), edge)
                         for edge, (row, _n) in table[key].items())
        placed.append((edge, q, key))
    # drawn bottom-up, so reverse: the first edge ends at the top of the panel
    return [key for _e, _q, key in sorted(placed, reverse=True)]


def area(ratio, lo, hi):
    """Dot area, stretched across the range plotted, as the single-cell figure does."""
    span = (hi - lo) or 1.0
    return MIN_AREA + (np.clip(ratio, lo, hi) - lo) / span * (MAX_AREA - MIN_AREA)


def legend_ticks(lo, hi, floor=LEGEND_FLOOR, step=LEGEND_STEP):
    """Raw ratios to draw, paired with their relative labels anchored at ``floor``.

    The smallest dot is called ``floor``; a dot twice its overlap ratio is called twice that.
    Ticks step by ``step`` up to whatever the largest dot works out to.
    """
    top = floor * hi / lo if lo else floor
    labels = [floor + i * step for i in range(int(round((top - floor) / step)) + 1)]
    ticks = [(label * lo / floor, label) for label in labels]
    # the top tick is the largest dot actually drawn, never a ratio nothing reaches
    if ticks[-1][0] < hi:
        ticks.append((hi, round(top, 2)))
    elif ticks[-1][0] > hi:
        ticks[-1] = (hi, round(top, 2))
    return ticks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=str(RESULTS / "enrichment_grid.tsv"))
    ap.add_argument("--resolution", default="100kb")
    ap.add_argument("--tolerance", type=int, default=10)
    ap.add_argument("--collection", action="append")
    ap.add_argument("--focus", default="blood", choices=["blood", "all"])
    ap.add_argument("--q-max", type=float, default=0.05)
    ap.add_argument("--min-overlap", type=int, default=10)
    ap.add_argument("--min-fold", type=float, default=1.5)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--plain", action="store_true",
                    help="drop the cross-caller marks (column shading, ring) and the "
                         "dagger; the figure then shows only set, edge, size and q")
    ap.add_argument("--size-breaks", default=None,
                    help="comma-separated overlap-ratio legend ticks, e.g. 0.4,0.6,0.8,1.0")
    ap.add_argument("--out")
    args = ap.parse_args()
    args.collection = args.collection or DEFAULT_COLLECTIONS

    table = combine(read_grid(Path(args.grid)), args)
    chosen = choose(table, args)
    if not chosen:
        print("Nothing to plot at this configuration.")
        return 1
    rows = order_rows(chosen, table)

    ratios = [float(row["overlap_ratio"]) for key in chosen
              for row, _n in table[key].values()]
    lo, hi = min(ratios), max(ratios)
    ticks = ([(float(t) * lo / LEGEND_FLOOR, float(t))
              for t in args.size_breaks.split(",")] if args.size_breaks
             else legend_ticks(lo, hi))
    daggered = {key for key in chosen
                if any(tandem(row) for row, _n in table[key].values())}
    both = sum(1 for key in chosen for _r, n in table[key].values() if n == 2)
    print(f"{len(rows)} rows, {len(daggered)} daggered; "
          f"{both} dots confirmed by both callers; raw overlap ratio {lo:.3f}-{hi:.3f}")
    print(f"legend is relative: {LEGEND_FLOOR} = the smallest dot "
          f"(raw overlap ratio {lo:.3f}); ticks "
          + ", ".join(f"{label:.2f}→{raw:.3f}" for raw, label in ticks))

    norm = Normalize(vmin=0.0, vmax=args.q_max)
    height = 0.30 * len(rows) + 2.3
    fig, ax = plt.subplots(figsize=(9.4, height))

    ax.set_xlim(-0.6, len(EDGES) - 0.4)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xticks(range(len(EDGES)))
    ax.set_xticklabels([e.replace("->", " → ") for e in EDGES],
                       rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([prettify(*key, key in daggered and not args.plain)
                        for key in rows], fontsize=8)
    for y in range(len(rows)):
        ax.axhline(y, color="0.94", lw=0.6, zorder=0)
    # the two columns RobusTAD has no counterpart for
    if not args.plain:
        for x, edge in enumerate(EDGES):
            if edge not in SHARED:
                ax.axvspan(x - 0.5, x + 0.5, color="0.965", zorder=-1)

    plain = {"ec": "white", "lw": 0.7}
    groups = ({2: plain, 1: plain} if args.plain
              else {2: {"ec": RING, "lw": 1.5}, 1: {"ec": "white", "lw": 0.7}})
    for n, style in groups.items():
        xs, ys, sizes, colors = [], [], [], []
        for y, key in enumerate(rows):
            for x, edge in enumerate(EDGES):
                cell = table[key].get(edge)
                if cell is None or cell[1] != n:
                    continue
                row, _ = cell
                xs.append(x)
                ys.append(y)
                sizes.append(area(float(row["overlap_ratio"]), lo, hi))
                colors.append(float(row["q"]))
        if xs:
            dots = ax.scatter(xs, ys, s=sizes, c=colors, cmap=CMAP, norm=norm,
                              edgecolors=style["ec"], linewidths=style["lw"], zorder=3)
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.75")

    bar = fig.colorbar(dots, ax=ax, fraction=0.028, pad=0.02)
    bar.set_label("adjusted p (BH)", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    handles = [Line2D([], [], marker="o", ls="", markerfacecolor="0.55",
                      markeredgecolor="white", markersize=np.sqrt(area(raw, lo, hi)),
                      label=f"{label:.2f}") for raw, label in ticks]
    ax.legend(handles=handles, title="overlap ratio", loc="upper left",
              bbox_to_anchor=(0.0, -1.15 / height), ncol=len(handles),
              frameon=False, fontsize=8, title_fontsize=8.5,
              handletextpad=0.4, columnspacing=1.5, borderpad=0.0)

    ax.set_title("Pathways over-represented among genes in reorganised TADs, per lineage "
                 f"edge\n{args.resolution}, "
                 f"Fisher's exact test + Benjamini–Hochberg, q ≤ {args.q_max}",
                 fontsize=11, pad=12)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else (
        FIGURES / f"pathway_dotplot_{args.resolution}_tol{args.tolerance}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
