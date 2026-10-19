#!/usr/bin/env python3
"""The one figure the paper carries for the embryo boundary analysis, as a 2x2 panel.

a, b  the stage trajectory: CTCF and gene density at the boundaries of each stage median,
      scored against relocations of the same boundaries, at three bin sizes. The claim is
      the shape -- a minimum at the 2-cell stage and a rise after it -- and not the
      height, which is why every test in c and d is a ratio taken inside one caller.
c     the boundary classes: what separates a boundary that moves from one that stays.
d     an **exploratory** annotation of the merge/split windows with early-development
      genes. It is not a test and the caption must say so -- the enrichment tests over
      these same windows are null; see ``results/CANDIDATE_LOCI_ONTAD.md``. The panel it
      replaced ("merge and split are not a separate class") is still drawn by
      ``plot_boundary_results.panel_merge_split`` and lives in ``merge_split.png``.

**The figure draws OnTAD alone.** RobusTAD is not deleted, only unselected: pass
``--callers OnTAD RobusTAD`` for the two-caller version, which is what every panel showed
before. The cross-caller agreement it carried has not gone anywhere -- section 3b of the
report quotes RobusTAD reproducing the 2-cell dip at the same magnitude on 15 of 19
chromosomes, and section 4 is built on scoring each caller against its own held-out cells.
Anything claimed from this figure alone is a single-caller claim and should say so.

Colour means the caller in all four panels, so the top and bottom rows can be read
together, and it keeps its meaning whichever callers are drawn. The panels themselves come
from ``plot_stage_maturation`` and ``plot_boundary_results``; this only composes them, so
the standalone figures and this one can never disagree.

Usage:
    python plot_main_figure.py
    python plot_main_figure.py --callers OnTAD RobusTAD
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_boundary_results import classes_table, panel_classes  # noqa: E402
from plot_locus_annotation import collect, panel_locus_annotation  # noqa: E402
from plot_stage_maturation import CALLER_TABLES, panel_trajectory  # noqa: E402
from summarise_results import read_tsv  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


# the paper sets these two panels in a left column beside a taller third panel, so they
# are drawn at the width and the heights that column gives them and are printed at scale
PANEL_WIDTH = 3.35
CLASSES_HEIGHT = 2.25
LOCI_HEIGHT = 3.04


def write_separate_panels(output, table, callers):
    """Panels c and d as two files, each drawn at the size it is printed at."""
    import plot_boundary_results as classes_module
    import plot_locus_annotation as loci_module

    stem = output.parent / output.stem
    output.parent.mkdir(parents=True, exist_ok=True)

    for module, sizes in ((classes_module, dict(TITLE_SIZE=7.4, LEGEND_SIZE=7.0,
                                                STAR_SIZE=6.5, NOTE_SIZE=6.0)),
                          (loci_module, dict(TITLE_SIZE=7.4, LEGEND_SIZE=5.9,
                                             NOTE_SIZE=5.8))):
        for name, value in sizes.items():
            if hasattr(module, name):
                setattr(module, name, value)

    figure, ax = plt.subplots(figsize=(PANEL_WIDTH, CLASSES_HEIGHT))
    panel_classes(ax, table, callers=callers, drop_stable=True,
                  title="Reorganised boundaries are CTCF and gene dense")
    ax.tick_params(labelsize=7.0)
    ax.yaxis.label.set_size(6.6)
    figure.tight_layout(pad=0.3)
    figure.subplots_adjust(right=0.995)
    figure.savefig(f"{stem}_classes.png", dpi=300)
    plt.close(figure)
    print(f"wrote {stem}_classes.png")

    rows, row_set = collect()
    figure, ax = plt.subplots(figsize=(PANEL_WIDTH, LOCI_HEIGHT))
    panel_locus_annotation(ax, rows, row_set, legend_cols=3,
                           title="Genes in merge / split windows")
    figure.tight_layout(pad=0.3)
    figure.subplots_adjust(right=0.995)
    figure.savefig(f"{stem}_loci.png", dpi=300)
    plt.close(figure)
    print(f"wrote {stem}_loci.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",
                        default=str(RESULTS / "figures" / "embryo_boundaries.png"))
    parser.add_argument("--callers", nargs="+", default=["OnTAD"],
                        choices=sorted(CALLER_TABLES),
                        help="callers to draw; default OnTAD alone")
    parser.add_argument("--separate-panels", action="store_true",
                        help="write panels c and d as two standalone files, unlettered "
                             "and without the stage trajectory, for a paper figure that "
                             "sets them beside a third panel of its own")
    args = parser.parse_args()

    rows_by_caller = {
        caller: read_tsv(RESULTS / CALLER_TABLES[caller] / "stage_boundary_enrichment.tsv")
        for caller in args.callers}
    table = classes_table()
    if not any(rows_by_caller.values()) or not table:
        print("Missing results/stage or results/turnover tables; run the pipeline first.")
        return

    if args.separate_panels:
        write_separate_panels(Path(args.output), table, args.callers)
        return

    # \textwidth is 469.8pt = 6.52in, so a figure drawn much wider than 8in loses too
    # much of its text to the scaling; at this size the page costs only a fifth
    figure, axes = plt.subplots(2, 2, figsize=(8.1, 6.8))
    panel_trajectory(axes[0][0], rows_by_caller, "ctcf", "a",
                     "Boundaries gain CTCF after the 2-cell stage")
    panel_trajectory(axes[0][1], rows_by_caller, "genes", "b",
                     "Gene density follows the same path")
    panel_classes(axes[1][0], table, letter="c", callers=args.callers)
    rows, row_set = collect()
    panel_locus_annotation(axes[1][1], rows, row_set, letter="d")
    figure.tight_layout(h_pad=2.0, w_pad=2.0)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
