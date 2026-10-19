#!/usr/bin/env python3
"""Alluvial ("Sankey") plot of the scHiCluster clustering: true stage -> predicted cluster.

Reads the per-cell assignment that ``run_schiccluster_scptr.py`` writes and draws the
flow of cells from their known developmental stage into the clusters that scHiCluster
returned. See ``cluster_sankey.py`` at the repository root for the drawing itself.

Usage:
    python plot_cluster_sankey.py --resolution 250kb
    python plot_cluster_sankey.py --resolution 250kb --chrom chr1
    python plot_cluster_sankey.py --all      # every resolution and chromosome present
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cluster_sankey import figure_for_single_method, print_table, summary_line  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_assignments(resolution="250kb", chrom="all"):
    """Per-cell (stage, cluster) plus the scores of that run."""
    results_dir = HERE / "results" / resolution / chrom
    table_path = results_dir / "schiccluster_clusters.tsv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"No scHiCluster assignment at {table_path}. "
            "Run run_schiccluster_scptr.py for this resolution first."
        )

    true_labels, predicted = [], []
    with open(table_path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            true_labels.append(row["cell_type"])
            predicted.append(int(row["predicted_cluster"]))

    scores = {}
    scores_path = results_dir / "schiccluster_results.txt"
    if scores_path.exists():
        for line in open(scores_path):
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 2 and fields[0] in ("ARI", "NMI", "Silhouette"):
                scores[fields[0]] = float(fields[1])

    return {
        "method": "scHiCluster",
        "true_labels": true_labels,
        "predicted": predicted,
        "ari": scores.get("ARI"),
        "nmi": scores.get("NMI"),
        "resolution": resolution,
        "results_dir": results_dir,
    }


def panel_title(resolution, chrom):
    where = "" if chrom == "all" else f", {chrom}"
    return f"scHiCluster, {resolution}{where}"


def render(run, output, resolution, chrom, quiet=False):
    fig, stats = figure_for_single_method(
        run["true_labels"], run["predicted"], title=panel_title(resolution, chrom)
    )
    # the scores line quotes the diagonal that drawing measured, so it is added afterwards
    fig.text(0.5, 0.015, summary_line(stats, ari=run["ari"], nmi=run["nmi"]),
             ha="center", fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if not quiet:
        print_table(stats)
    print(f"wrote {output}")
    return stats


def run_all():
    """Draw every assignment present under results/: each resolution and chromosome."""
    paths = sorted((HERE / "results").glob("*/*/schiccluster_clusters.tsv"))
    if not paths:
        raise FileNotFoundError(f"No assignments under {HERE / 'results'}")
    for path in paths:
        chrom, resolution = path.parent.name, path.parent.parent.name
        run = load_assignments(resolution, chrom)
        render(run, path.parent / "schiccluster_sankey.png", resolution, chrom, quiet=True)
    print(f"drew {len(paths)} alluvial plots under {HERE / 'results'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="250kb", help="e.g. 250kb, 500kb, 1000kb")
    parser.add_argument("--chrom", default="all")
    parser.add_argument("--output", default=None,
                        help="default: schiccluster_sankey.png beside the results")
    parser.add_argument("--all", action="store_true",
                        help="draw every resolution and chromosome present")
    args = parser.parse_args()

    if args.all:
        run_all()
        return

    run = load_assignments(args.resolution, args.chrom)
    out = args.output or run["results_dir"] / "schiccluster_sankey.png"
    render(run, out, args.resolution, args.chrom)


if __name__ == "__main__":
    main()
