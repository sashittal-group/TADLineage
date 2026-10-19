#!/usr/bin/env python3
"""Alluvial ("Sankey") plot of the scHiCPTR clustering: true stage -> predicted cluster.

Reads the per-cell assignment that ``run_schicptr_scptr.py`` writes and draws the flow of
cells from their known developmental stage into the clusters scHiCPTR returned. The drawing
itself lives in ``cluster_sankey.py`` at the repository root and is shared with
``schiccluster_scptr/``, ``higashi_scptr/`` and ``scptr_robustad/``.

Usage:
    python plot_cluster_sankey.py --resolution 250kb
    python plot_cluster_sankey.py --resolution 250kb --chrom chr1 --method kmeans
    python plot_cluster_sankey.py --all      # every resolution, chromosome and clustering
"""

import argparse
import csv
import json
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


def load_assignments(resolution="250kb", chrom="all", method="louvain"):
    """Per-cell (stage, cluster) plus the scores of that run."""
    results_dir = HERE / "results" / resolution / chrom / method
    table_path = results_dir / "schicptr_clusters.tsv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"No scHiCPTR assignment at {table_path}. "
            "Run run_schicptr_scptr.py for this resolution first."
        )

    true_labels, predicted = [], []
    with open(table_path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            true_labels.append(row["cell_type"])
            predicted.append(int(row["predicted_cluster"]))

    scores = json.loads((results_dir / "schicptr_results.json").read_text())
    return {
        "method": "scHiCPTR",
        "true_labels": true_labels,
        "predicted": predicted,
        "ari": scores.get("ARI"),
        "nmi": scores.get("NMI"),
        "resolution": resolution,
        "results_dir": results_dir,
    }


def panel_title(resolution, chrom, method):
    where = "" if chrom == "all" else f", {chrom}"
    return f"scHiCPTR ({method}), {resolution}{where}"


def render(run, output, resolution, chrom, method, quiet=False):
    fig, stats = figure_for_single_method(
        run["true_labels"], run["predicted"], title=panel_title(resolution, chrom, method)
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
    """Draw every assignment present under results/."""
    paths = sorted((HERE / "results").glob("*/*/*/schicptr_clusters.tsv"))
    if not paths:
        raise FileNotFoundError(f"No assignments under {HERE / 'results'}")
    for path in paths:
        method = path.parent.name
        chrom = path.parent.parent.name
        resolution = path.parent.parent.parent.name
        run = load_assignments(resolution, chrom, method)
        render(run, path.parent / "schicptr_sankey.png", resolution, chrom, method, quiet=True)
    print(f"drew {len(paths)} alluvial plots under {HERE / 'results'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="250kb", help="e.g. 250kb, 500kb, 1000kb")
    parser.add_argument("--chrom", default="all")
    parser.add_argument("--method", default="louvain", choices=["louvain", "louvain5", "kmeans", "spectral"])
    parser.add_argument("--output", default=None,
                        help="default: schicptr_sankey.png beside the results")
    parser.add_argument("--all", action="store_true",
                        help="draw every resolution, chromosome and clustering present")
    args = parser.parse_args()

    if args.all:
        run_all()
        return

    run = load_assignments(args.resolution, args.chrom, args.method)
    out = args.output or run["results_dir"] / "schicptr_sankey.png"
    render(run, out, args.resolution, args.chrom, args.method)


if __name__ == "__main__":
    main()
