#!/usr/bin/env python3
"""Panel d: which early-development genes fall inside a merge/split window.

**This panel is exploratory and the caption must say so.** It is an annotation of the
calls, not a test of them. Every quantitative claim in this folder is elsewhere: the
enrichment tests over these same windows are null (see `results/CANDIDATE_LOCI_ONTAD.md`
for the numbers and for what argues against reading loci off this panel). What the panel
shows is that the windows are populated by genes that belong to the process the embryo is
going through, rather than by an arbitrary slice of the genome.

Two conventions keep it honest:

* **Gene clusters are drawn as one row.** HoxA, HoxD and Zscan4 are single loci; counting
  them by gene would let one locus supply ten entries and is the trap recorded in the
  folder's notes.
* **A window is at most 1 Mb.** Wider focal windows exist and are dropped here, because a
  4 Mb window containing a famous gene says nothing about placement.

Usage:
    python plot_locus_annotation.py                 # standalone panel
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (str(REPO_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from annotate_events import GeneIndex  # noqa: E402
from coordinates import resolution_to_bp  # noqa: E402
from gene_sets import load_gene_table  # noqa: E402

RESULTS = HERE / "results"
TRANSITIONS = ["1-cell->2-cell", "2-cell->4-cell", "4-cell->8-cell", "8-cell->64-cell"]
TRANSITION_LABEL = {t: t.replace("-cell", "") for t in TRANSITIONS}
MAX_WINDOW_BP = 1_000_000

# the curated sets this panel draws, in the order they are stacked
SET_FILES = [
    ("lineage", "lineage_first_segregation"),
    ("pluripotency", "pluripotency"),
    ("Hox", "hox_clusters"),
]
SET_COLORS = {"lineage": "#c44e52", "pluripotency": "#dd8452", "Hox": "#8172b3"}
# The legend names each group for what its genes do. "lineage" is not used as a label,
# because the epiblast is a lineage too and Nanog and Sox2 regulate it -- the contrast the
# panel draws is between the genes that push cells towards the supporting tissues and the
# genes that hold cells undecided, not between lineage and pluripotency.
SET_LABELS = {"lineage": "trophectoderm / endoderm", "pluripotency": "pluripotency",
              "Hox": "Hox"}
# Nanog, Sox2 and Pou5f1 belong to both curated sets. They are drawn as pluripotency,
# because the contrast the panel shows at 4-cell -> 8-cell is the pluripotency core against
# the lineage specifiers, and the lineage set lists them only for its epiblast arm.
SET_PRIORITY = ["pluripotency", "lineage", "Hox"]
# genes that are one locus; drawn as a single row under the given label
CLUSTERS = {
    "HoxA cluster": lambda g: g.startswith("Hoxa"),
    "HoxD cluster": lambda g: g.startswith("Hoxd"),
    "Zscan4 cluster": lambda g: g.startswith("Zscan4"),
}
MARKERS = {"split": "^", "merge": "v"}
TITLE_SIZE = 10
LEGEND_SIZE = 8
NOTE_SIZE = 7.5


def read_set(name):
    path = HERE / "gene_sets" / f"{name}.txt"
    return {line.split()[0] for line in open(path)
            if line.strip() and not line.startswith("#")}


def collect(events_dir=None):
    """``{row label: {transition: {event types}}}`` plus the row's set, OnTAD calls."""
    by_label = {label: read_set(filename) for label, filename in SET_FILES}
    membership = {}
    for label in SET_PRIORITY:
        for gene in by_label[label]:
            membership.setdefault(gene, label)

    index = GeneIndex(load_gene_table())
    rows = collections.defaultdict(lambda: collections.defaultdict(set))
    row_set = {}
    pattern = str(Path(events_dir or RESULTS / "events_ontad") / "*" / "*" / "tol_*_events.tsv")
    for path in glob.glob(pattern):
        for record in csv.DictReader(open(path), delimiter="\t"):
            if record["event_type"] not in MARKERS or not record["focal_start"]:
                continue
            start, end = int(record["focal_start"]), int(record["focal_end"])
            if end - start > MAX_WINDOW_BP:
                continue
            end += resolution_to_bp(record["resolution"])
            chrom = record["chrom"]
            for position, symbol in zip(index.positions.get(chrom, []),
                                        index.symbols.get(chrom, [])):
                if not start <= position <= end or symbol not in membership:
                    continue
                label = next((name for name, test in CLUSTERS.items() if test(symbol)),
                             symbol)
                rows[label][record["transition"]].add(record["event_type"])
                row_set[label] = membership[symbol]
    return rows, row_set


def panel_locus_annotation(ax, rows, row_set, letter="", title=None, legend_cols=3):
    """Gene rows against the four transitions, split up and merge down."""
    order = []
    for label, _ in SET_FILES:
        members = sorted(name for name in rows if row_set[name] == label)
        # the transition a row is first seen at, so the panel reads left to right
        members.sort(key=lambda name: min(TRANSITIONS.index(t) for t in rows[name]))
        order.extend((name, label) for name in members)

    for y, (name, label) in enumerate(order):
        ax.axhline(y, color="0.92", linewidth=0.6, zorder=0)
        for transition, kinds in rows[name].items():
            x = TRANSITIONS.index(transition)
            for offset, kind in zip((-0.13, 0.13) if len(kinds) > 1 else (0.0,),
                                    sorted(kinds)):
                ax.scatter(x + offset, y, marker=MARKERS[kind], s=34,
                           color=SET_COLORS[label], edgecolors="none", zorder=3)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([name for name, _ in order], fontsize=NOTE_SIZE)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xticks(range(len(TRANSITIONS)))
    ax.set_xticklabels([TRANSITION_LABEL[t] for t in TRANSITIONS], fontsize=NOTE_SIZE)
    ax.set_xlim(-0.5, len(TRANSITIONS) - 0.5)
    ax.set_xlabel("transition", fontsize=TITLE_SIZE - 1, labelpad=2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([], [], linestyle="none", marker=MARKERS[k], color="0.35",
                      markersize=5, label=k) for k in ("split", "merge")]
    handles += [Line2D([], [], linestyle="none", marker="s", color=SET_COLORS[label],
                       markersize=5, label=SET_LABELS[label]) for label, _ in SET_FILES]
    # one row under the axis: the panel is dense enough that any in-panel placement sits
    # on a gene, and a single row costs less height than a stacked block
    # three columns, so the two marker entries share a column and the longest group name
    # does not push the row past the panel
    ax.legend(handles=handles, fontsize=LEGEND_SIZE, frameon=False, ncol=legend_cols,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), handletextpad=0.3,
              columnspacing=1.4, labelspacing=0.3, borderaxespad=0.0)
    heading = title or "Genes in merge / split windows"
    ax.set_title(f"{letter}  {heading}" if letter else heading,
                 loc="left", fontsize=TITLE_SIZE)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(RESULTS / "figures" / "locus_annotation.png"))
    parser.add_argument("--events-dir", default=None)
    args = parser.parse_args()

    rows, row_set = collect(args.events_dir)
    if not rows:
        print("No OnTAD events found; run the pipeline first.")
        return
    figure, ax = plt.subplots(figsize=(4.4, 4.0))
    panel_locus_annotation(ax, rows, row_set, letter="d")
    figure.tight_layout()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
