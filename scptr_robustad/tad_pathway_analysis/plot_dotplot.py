#!/usr/bin/env python3
"""The pathway figure: one configuration, one column per stage transition.

One column per transition, one row per gene set, in the dot-plot style of the reference
figure.  The area of a dot is the overlap ratio -- the fraction of the gene set lying in
reorganised structure -- and its colour is the BH-adjusted p-value of Fisher's exact test.

A boundary that moves and a domain that is gained, lost, split or merged are two views of
the same reorganisation, so the foreground merges them: a gene counts if its start site
falls either in a bin carrying a gained or lost boundary, or inside the body of a
reorganised domain.  That is the `--feature both` table.

Gene sets are contributed by each column, and a set is skipped if most of its overlapping
genes are already on the plot under another name -- Reactome names the mouse histone
cluster eight ways, and without that step eight rows of the figure are one locus.  This is
presentation, not filtering; the table keeps everything.

Usage:
    python plot_dotplot.py                       # the configuration chosen in the README
    python plot_dotplot.py --caller RobusTAD --resolution 1000kb --tolerance 3
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

STAGE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
TRANSITIONS = [f"{a}->{b}" for a, b in zip(STAGE_ORDER, STAGE_ORDER[1:])]
# the column label: "1-cell->2-cell" is too long to sit under a column
SHORT = {f"{a}->{b}": f"{a.split('-')[0]}-{b.split('-')[0]}"
         for a, b in zip(STAGE_ORDER, STAGE_ORDER[1:])}

# a set is dropped from the figure if this much of its overlap is already plotted
REDUNDANCY = 0.5
CMAP = "rainbow_r"
# the number the smallest dot on the plot is labelled with; the rest follow in
# proportion, so the legend reads as relative size rather than as an absolute ratio
LEGEND_FLOOR = 0.01

ABBREVIATIONS = {
    "Dna": "DNA", "Rna": "RNA", "Mhc": "MHC", "Hdacs": "HDACs", "Hats": "HATs",
    "Prc2": "PRC2", "Atp": "ATP", "Er": "ER", "Mrna": "mRNA", "Il": "IL",
    "Jak": "JAK", "Stat": "STAT", "P53": "p53", "Uv": "UV", "Cenpa": "CENP-A",
    "Ifna": "IFN-alpha", "Ifnb": "IFN-beta", "Adme": "ADME", "Runx1": "RUNX1",
    "Dap12": "DAP12", "Tcr": "TCR", "Cd3": "CD3", "Pdgf": "PDGF", "Cbl": "Cbl",
    "Ub": "Ub", "Esr": "ESR", "Hdr": "HDR", "Hrr": "HRR", "Pd": "PD",
}
# names Reactome spells with a number that Title-casing splits off from its acronym
REJOIN = {"PD 1": "PD-1", "HRR": "(HRR)", "Interleukin 3": "IL-3", "Interleukin 5": "IL-5"}


# Reactome spells a handful of sets at a length that no wrapping fits into a panel
# column; these are the same sets under the name a reader would write them by hand
RENAME = {
    "REACTOME_DNA_DAMAGE_TELOMERE_STRESS_INDUCED_SENESCENCE":
        "DNA Damage / Telomere Stress Senescence",
    "REACTOME_IMMUNOREGULATORY_INTERACTIONS_BETWEEN_A_LYMPHOID_AND_A_NON_LYMPHOID_CELL":
        "Lymphoid / Non-Lymphoid Immunoregulation",
    "REACTOME_G2_M_CHECKPOINTS": "G2/M Checkpoints",
    "REACTOME_PHOSPHORYLATION_OF_CD3_AND_TCR_ZETA_CHAINS": "CD3 And TCR Zeta Phosphorylation",
    "REACTOME_ACTIVATION_OF_MATRIX_METALLOPROTEINASES": "Matrix Metalloproteinase Activation",
    "REACTOME_FORMYL_PEPTIDE_RECEPTORS_BIND_FORMYL_PEPTIDES_AND_MANY_OTHER_LIGANDS":
        "Formyl Peptide Receptor Binding",
    "REACTOME_E3_UBIQUITIN_LIGASES_UBIQUITINATE_TARGET_PROTEINS":
        "E3 Ubiquitin Ligase Targets",
    "REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION":
        "Class I MHC Antigen Presentation",
    "REACTOME_COSTIMULATION_BY_THE_CD28_FAMILY": "CD28 Family Costimulation",
    "REACTOME_REGULATION_OF_SIGNALING_BY_CBL": "Regulation Of Signaling By Cbl",
}


def prettify(name):
    """``REACTOME_MITOTIC_PROPHASE`` -> ``Mitotic Prophase``."""
    if name in RENAME:
        return RENAME[name]
    for prefix in ("REACTOME_", "GOBP_", "HALLMARK_", "GOMF_", "GOCC_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    words = name.replace("_", " ").title().split()
    text = " ".join(ABBREVIATIONS.get(w, w) for w in words)
    for old, new in REJOIN.items():
        text = text.replace(old, new)
    return text


def wrap(name, width=40, lines=2):
    """Long gene-set names over a few lines, so the axis does not clip them."""
    return "\n".join(textwrap.wrap(prettify(name), width)[:lines])


def read_enrichment(caller, resolution, tolerance, collection, feature="both"):
    """``{transition: [row]}`` for one configuration, best p-value first."""
    per_column = {t: [] for t in TRANSITIONS}
    path = RESULTS / f"enrichment_{feature}.tsv"
    if not path.exists():
        print(f"missing {path}; run pathway_enrichment.py --feature {feature}",
              file=sys.stderr)
        return per_column
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if (row["caller"] != caller or row["resolution"] != resolution
                    or int(row["tolerance"]) != tolerance
                    or row["collection"] != collection):
                continue
            if row["transition"] not in per_column:
                continue
            row["q"] = float(row["q"])
            row["overlap_ratio"] = float(row["overlap_ratio"])
            row["gene_list"] = set(row["genes"].split(";")) if row["genes"] else set()
            per_column[row["transition"]].append(row)
    for rows in per_column.values():
        rows.sort(key=lambda r: r["q"])
    return per_column


def choose(per_column, top, q_max):
    """The gene sets to draw: the best of each column, minus near-duplicate gene lists.

    Redundancy is judged against every set already chosen, across all columns, so a locus
    picked up in one transition does not reappear under a second name in the next.
    """
    chosen, kept_lists = [], []
    for transition in TRANSITIONS:
        n = 0
        for row in per_column[transition]:
            if n >= top or row["q"] > q_max:
                break
            if row["gene_set"] in chosen:
                continue
            genes = row["gene_list"]
            if genes and any(len(genes & other) / len(genes) > REDUNDANCY
                             for other in kept_lists):
                continue
            chosen.append(row["gene_set"])
            kept_lists.append(genes)
            n += 1
    return chosen


def build(per_column, labels, q_max):
    """``{(gene set, transition): row}`` for every chosen set, in every column it reaches."""
    wanted = set(labels)
    cells = {}
    for transition, rows in per_column.items():
        for row in rows:
            if row["gene_set"] in wanted and row["q"] <= q_max:
                cells[(row["gene_set"], transition)] = row
    return cells


def draw(cells, labels, title, out_path, compact=False, width=2.9, max_height=None):
    """The dot plot.

    ``compact`` draws the panel at the size it is printed at, so that a figure that sets
    it beside two other panels scales it by about one rather than by one half and keeps
    its text legible.
    """
    n_rows, n_cols = len(labels), len(TRANSITIONS)
    ratios = [c["overlap_ratio"] for c in cells.values()]
    qs = [c["q"] for c in cells.values()]
    if not ratios:
        print("nothing to draw", file=sys.stderr)
        return
    norm = Normalize(vmin=min(qs), vmax=max(qs))
    lo, hi = min(ratios), max(ratios)
    span = (hi - lo) or 1.0
    # what a gene set would score by chance: the foreground's share of the universe
    baseline = sum(int(c["n_foreground"]) / int(c["n_universe"])
                   for c in cells.values()) / len(cells)

    def area(ratio):
        return 20 + 250 * (ratio - lo) / span

    cax = None
    squeeze = 1.0
    if compact:
        # the keys sit under the panel rather than beside it, because a key beside it
        # would take a third of a width the four columns already need in full
        # everything scales with the width the panel is printed at, so the text is
        # never scaled away and never has to be scaled up on the page
        k = width / 2.15
        panel = 0.34 * k * n_rows
        height = panel + 1.20 * k
        # a full grid of gene sets is taller than the column the paper gives it, so the
        # rows are compressed to fit and the row labels shrink with them
        if max_height and height > max_height:
            squeeze = (max_height - 1.20 * k) / panel
            panel, height = panel * squeeze, max_height
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(3, 1, height_ratios=[panel, 0.22 * k, 0.06 * k],
                                left=0.46, right=0.98,
                                top=1 - 0.50 * k / height, bottom=0.24 * k / height,
                                hspace=0.35)
        ax = fig.add_subplot(grid[0])
        key = fig.add_subplot(grid[1])
        key.axis("off")
        cax = fig.add_subplot(grid[2])
    else:
        height = 1.9 + 0.34 * n_rows
        fig, ax = plt.subplots(figsize=(8.6, height + 0.8))
    if compact:
        label_size = max(4.8, 5.6 * (width / 2.15) * min(1.0, squeeze ** 0.5))
        tick_size = 6.0 * (width / 2.15)
    else:
        label_size, tick_size = 8, 10

    for (name, transition), row in cells.items():
        size = area(row["overlap_ratio"])
        ax.scatter(TRANSITIONS.index(transition), labels.index(name),
                   s=size * 0.26 * (width / 2.15) ** 2 if compact else size,
                   c=[row["q"]], cmap=CMAP, norm=norm,
                   edgecolors="none", zorder=3)

    ax.set_yticks(range(n_rows))
    lines = 2 if squeeze < 0.8 else 3
    ax.set_yticklabels([wrap(n, 24, lines) if compact else wrap(n) for n in labels],
                       fontsize=label_size)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([SHORT[t] for t in TRANSITIONS], fontsize=tick_size)
    ax.tick_params(axis="x", length=0, pad=3 if compact else 6)
    ax.set_xlabel("stage transition", fontsize=label_size + 0.6,
                  labelpad=4 if compact else 8)
    if compact:
        head, _, sub = title.partition("\n")
        fig.suptitle("\n".join(textwrap.wrap(head, 30) + ([sub] if sub else [])),
                     fontsize=label_size + 0.4, y=1 - 0.05 / height, va="top")
    else:
        ax.set_title(title, fontsize=11)

    # Five steps across the range the plot actually uses, so the legend spans the dot
    # sizes on the page rather than only their two ends.  The numbers beside them are
    # *relative*: the smallest dot is called LEGEND_FLOOR and every other is in
    # proportion to it, so a reader sees "this dot is three times that one" at a glance
    # instead of having to weigh an absolute ratio against the 19% baseline.  The
    # mapping back to the true overlap ratio is printed below, for the supplementary.
    steps = 2 if compact else 4
    ticks = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    scale = 0.44 * (width / 2.15) if compact else 1.0
    handles = [Line2D([], [], marker="o", linestyle="none", color="0.45",
                      markersize=((area(v) / 12) ** 0.5 + 2) * scale,
                      label=f"{LEGEND_FLOOR * v / lo:.3f}")
               for v in ticks]
    if compact:
        key.legend(handles=handles, title="Overlap ratio", loc="upper center",
                   bbox_to_anchor=(0.5, 1.35), frameon=False, ncol=len(handles),
                   fontsize=label_size - 1.0, title_fontsize=label_size - 1.0,
                   handletextpad=0.25, columnspacing=1.0, borderpad=0.0)
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                           orientation="horizontal")
        bar.set_label("adjusted $p$", fontsize=label_size - 1.0, labelpad=1)
        bar.ax.tick_params(labelsize=label_size - 1.8, pad=1)
        bar.locator = MaxNLocator(nbins=4)
        bar.update_ticks()
    else:
        leg = ax.legend(handles=handles, title="Overlap Ratio", loc="upper left",
                        bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8,
                        labelspacing=1.1, title_fontsize=8)
        ax.add_artist(leg)
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax,
                           fraction=0.03, pad=0.24, aspect=12)
        bar.set_label("Adj P-value", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        fig.subplots_adjust(left=0.36, right=0.78, top=0.90, bottom=0.10)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"{out_path}  ({n_rows} gene sets, {len(cells)} dots)")
    print(f"legend scale: relative = {LEGEND_FLOOR} x overlap_ratio / {lo:.4f}"
          f"   (smallest dot {lo:.4f}, largest {hi:.4f})")
    for v in ticks:
        print(f"   {LEGEND_FLOOR * v / lo:>7.3f}  =  overlap ratio {v:.3f}"
              f"  =  {v / baseline:.1f}x over a {baseline:.0%} background")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default="m2.cp.reactome")
    parser.add_argument("--caller", default="OnTAD")
    parser.add_argument("--resolution", default="250kb")
    parser.add_argument("--tolerance", type=int, default=5)
    parser.add_argument("--feature", default="both",
                        choices=("both", "boundary", "domain"))
    parser.add_argument("--top", type=int, default=5,
                        help="gene sets contributed by each column")
    parser.add_argument("--q-max", type=float, default=0.05)
    parser.add_argument("--out")
    parser.add_argument("--panel-width", type=float, default=2.9,
                        help="inches the compact panel is printed at")
    parser.add_argument("--panel-height", type=float,
                        help="inches the compact panel may not exceed; the rows are "
                             "compressed to fit it")
    parser.add_argument("--compact", action="store_true",
                        help="draw the panel at the size a three-panel paper figure "
                             "prints it at, so its text is not scaled away")
    args = parser.parse_args()

    per_column = read_enrichment(args.caller, args.resolution, args.tolerance,
                                 args.collection, args.feature)
    labels = choose(per_column, args.top, args.q_max)
    if not labels:
        print("no gene set passes the cut for this configuration", file=sys.stderr)
        return 1
    cells = build(per_column, labels, args.q_max)
    title = ("Pathways in reorganised TADs at each stage transition\n"
             f"{args.resolution}, {args.collection}")
    out = args.out or (FIGURES / f"pathway_dotplot_{args.caller}_{args.resolution}_"
                                 f"tol{args.tolerance}.png")
    draw(cells, labels, title, out, compact=args.compact, width=args.panel_width,
         max_height=args.panel_height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
