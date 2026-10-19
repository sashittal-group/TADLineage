#!/usr/bin/env python3
"""Preview of the paper's Fig. 3 (embryo clustering) with the stage-blind contact pipeline.

Same layout and drawing code as TAD_Reorganization_latex/make_embryo_figure.py (imported,
not edited); only the TADLineage results change. All TADLineage fits use the paper's distance
(domains + start and end boundaries, tolerance 0) and the exact Nussinov median, and are read
from ../multires/ari_summary.tsv (250 kb and 500 kb from ../multires, 1 Mb from
../nussinov_median):

  (b) genome-wide     one fit over all chromosomes per bin size
  (c) one chromosome  chromosome-only fits (contacts, graph and hierarchies of that chromosome)
  (d-g) Sankeys       the chromosome and bin size where TADLineage has its highest
                      chromosome-only ARI, and the other three methods on the same chromosome
                      and bin size; scHiCPTR draws the best of its four clusterings, as in the
                      paper's figure

A bin size without TADLineage results is marked "not run". Writes figure3_preview.{pdf,png}
next to this script; nothing in the paper folder is written.

    /Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python make_figure3_preview.py
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO_ROOT = EXPERIMENT.parents[1]
LATEX_DIR = REPO_ROOT / "TAD_Reorganization_latex"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LATEX_DIR))

import make_embryo_figure as fig3  # noqa: E402  (sets the paper's rcParams)
import matplotlib.pyplot as plt  # noqa: E402
from cluster_sankey import diagonal_fraction, draw_alluvial  # noqa: E402
from make_embryo_trajectory import draw_trajectory  # noqa: E402

SUMMARY = EXPERIMENT / "multires/ari_summary.tsv"
SCHICPTR_CLUSTERINGS = ["kmeans", "louvain", "louvain5", "spectral"]
HIGASHI_RESOLUTION = {"250kb": "250000", "500kb": "500000", "1000kb": "1000000"}
LABEL = {"250kb": "250 kb", "500kb": "500 kb", "1000kb": "1 Mb"}


def stage_labels():
    ids = json.loads((EXPERIMENT / "data/cells.json").read_text())
    with open(REPO_ROOT / "scptr/GSE129029_metadata.csv") as handle:
        stage = {row["GSM"]: row["CELL_STAGE"] for row in csv.DictReader(handle)}
    return [stage[cell] for cell in ids]


def tadlineage_results():
    """{resolution: genome-wide ARI}, {resolution: [chromosome ARIs]}, and every row."""
    rows = list(csv.DictReader(open(SUMMARY), delimiter="\t"))
    genome, per_chrom = {}, defaultdict(list)
    for row in rows:
        row["ari"] = float(row["ari"])
        if row["chromosome"] == "all":
            genome[row["resolution"]] = row["ari"]
        else:
            per_chrom[row["resolution"]].append(row["ari"])
    return genome, dict(per_chrom), rows


def baseline_runs(chrom, resolution):
    """The other three methods on one chromosome and bin size; scHiCPTR's best clustering."""
    def load(folder, **kwargs):
        module = fig3.load_module(f"_{folder}", REPO_ROOT / folder / "plot_cluster_sankey.py")
        return module.load_assignments(**kwargs)
    schicptr = max((load("schicptr_scptr", resolution=resolution, chrom=chrom, method=m)
                    for m in SCHICPTR_CLUSTERINGS), key=lambda run: run["ari"])
    return [("scHiCluster", load("schiccluster_scptr", resolution=resolution, chrom=chrom)),
            ("Higashi", load("higashi_scptr", resolution=HIGASHI_RESOLUTION[resolution], chrom=chrom)),
            ("scHiCPTR", schicptr)]


def draw_sankey(ax, true_labels, predicted, method, ari, nmi, where):
    draw_alluvial(ax, true_labels, predicted, title=method, show_counts=False,
                  label_fontsize=fig3.SMALL, show_cluster_labels=False)
    ax.set_xlim(-0.42, 1.04)
    ax.set_title(method, fontsize=fig3.FONT, pad=3)
    matched = diagonal_fraction(true_labels, predicted)
    ax.text(0.5, -0.04, f"ARI = {ari:.2f}   NMI = {nmi:.2f}\naccuracy = {matched:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=fig3.SMALL, linespacing=1.3)
    print(f"{method} {where}: ARI {ari:.3f}, NMI {nmi:.3f}, accuracy {matched:.3f}")


def main():
    ari = fig3.load_module("_ari", REPO_ROOT / "scptr_robustad" / "plot_ari_comparison.py")
    data = {name: ari.read(path, kind) for name, path, kind, _ in ari.METHODS}
    truth = stage_labels()
    genome, per_chrom, rows = tadlineage_results()
    data["TADLineage"] = (genome, per_chrom)
    for resolution in ari.RESOLUTIONS:
        values = per_chrom.get(resolution, [])
        print(f"TADLineage {LABEL[resolution]}: genome-wide {genome.get(resolution, float('nan')):.3f}, "
              f"chromosome median {np.median(values) if values else float('nan'):.3f} over {len(values)}")
    best = max((r for r in rows if r["chromosome"] != "all"), key=lambda r: r["ari"])
    chrom, resolution = best["chromosome"], best["resolution"]
    where = f"{chrom}, {LABEL[resolution]}"

    fig = plt.figure(figsize=(fig3.FIG_WIDTH_IN, fig3.FIG_HEIGHT_IN))
    columns = [(0.0, 0.30), (0.335, 0.665), (0.67, 1.0)]
    bottom_columns = [(i / 4 + 0.005, (i + 1) / 4 - 0.02) for i in range(4)]
    top_row, bottom_row = (0.57, 0.94), (0.075, 0.43)
    axis_inset = 0.065

    def cell(column, row, inset=0.0, right=0.0):
        x0, x1 = columns[column]
        y0, y1 = row
        return [x0 + inset, y0, x1 - x0 - inset - right, y1 - y0]

    ax_traj = fig.add_axes([0.012, 0.72, 0.288, 0.20])
    draw_trajectory(ax_traj, font=0.26, line=0.25)
    for image in ax_traj.get_images():
        image.set_interpolation("none")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
               for _, _, _, color in ari.METHODS]
    fig.legend(handles, [name for name, _, _, _ in ari.METHODS], loc="center",
               bbox_to_anchor=(0.15, 0.63), ncol=2, frameon=False, handlelength=1.0,
               columnspacing=1.2)

    ax_all = fig.add_axes(cell(1, top_row, axis_inset, 0.015))
    ax_chrom = fig.add_axes(cell(2, top_row, axis_inset, 0.015))
    fig3.draw_genome_wide(ax_all, ari, data)
    fig3.draw_per_chromosome(ax_chrom, ari, data)
    # mark bin sizes without TADLineage results, so an absent bar is not read as zero
    width = 0.8 / len(ari.METHODS)
    centre = (len(ari.METHODS) - 1) / 2
    for ax, have in ((ax_all, genome), (ax_chrom, per_chrom)):
        for position, res in enumerate(ari.RESOLUTIONS):
            if res not in have:
                ax.text(position - centre * width + 0.07, 0.02, "not run", rotation=90, ha="center",
                        va="bottom", fontsize=fig3.SMALL - 1.5, color="#4c72b0")

    y0, y1 = bottom_row
    axes = [fig.add_axes([x0, y0, x1 - x0, y1 - y0]) for x0, x1 in bottom_columns]
    predicted = np.load(best["path"])["labels"].tolist()
    draw_sankey(axes[0], truth, predicted, "TADLineage", adjusted_rand_score(truth, predicted),
                normalized_mutual_info_score(truth, predicted), where)
    for ax, (method, run) in zip(axes[1:], baseline_runs(chrom, resolution)):
        draw_sankey(ax, run["true_labels"], run["predicted"], method, run["ari"], run["nmi"], where)
        assert Counter(run["true_labels"]) == Counter(truth), f"{method} clusters other cells"

    for i, letter in enumerate("abcdefg"):
        x = max((columns[i] if i < 3 else bottom_columns[i - 3])[0], 0.005)
        y = (top_row if i < 3 else bottom_row)[1] + 0.02
        fig.text(x, y, f"({letter})", fontsize=fig3.FONT + 1, fontweight="bold",
                 ha="left", va="bottom")

    out = HERE / "figure3_preview.pdf"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"bottom row: {where}")
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
