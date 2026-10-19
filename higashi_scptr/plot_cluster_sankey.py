#!/usr/bin/env python3
"""Alluvial ("Sankey") plot of the Higashi clustering: true stage -> predicted cluster.

Reads the per-cell assignment that ``cluster_all_embeddings.py`` writes into
``results/<resolution>/all/cluster_results.txt`` and draws the flow of cells from their
known developmental stage into the clusters that Higashi returned. See
``cluster_sankey.py`` at the repository root for the drawing itself.

Usage:
    python plot_cluster_sankey.py --resolution 1000000
    python plot_cluster_sankey.py --resolution 1000000 --chrom chr1
    python plot_cluster_sankey.py --all      # every resolution and chromosome present
"""

import argparse
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

SCORE_KEYS = ("ARI", "NMI", "Silhouette")


def load_assignments(resolution="1000000", chrom="all"):
    """Per-cell (stage, cluster) plus the scores of that run.

    ``cluster_results.txt`` comes in two layouts, and this reads either. The genome-wide
    runs of ``cluster_all_embeddings.py`` write the configuration and the scores, a
    per-cluster summary, then a per-cell table headed
    ``cell_id predicted_cluster cell_type total_coverage``. The per-chromosome runs of
    ``cluster_embeddings.py`` write the scores followed by a per-cell table headed
    ``predicted_cluster cell_type``. Both are found by looking for the header that names
    the two columns this needs, so neither column order is assumed.
    """
    results_dir = HERE / "results" / str(resolution) / chrom
    results_path = results_dir / "cluster_results.txt"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No Higashi result at {results_path}. "
            "Run cluster_all_embeddings.py for this resolution first."
        )

    scores = {}
    true_labels, predicted = [], []
    cluster_column = stage_column = None
    for line in open(results_path):
        fields = line.rstrip("\n").split("\t")
        if "predicted_cluster" in fields and "cell_type" in fields:
            cluster_column = fields.index("predicted_cluster")
            stage_column = fields.index("cell_type")
            continue
        if cluster_column is not None:
            if len(fields) <= max(cluster_column, stage_column):
                continue
            predicted.append(int(fields[cluster_column]))
            true_labels.append(fields[stage_column])
        elif len(fields) == 2 and fields[0] in SCORE_KEYS:
            scores[fields[0]] = float(fields[1])

    if not true_labels:
        raise ValueError(f"No per-cell rows found in {results_path}")

    return {
        "method": "Higashi",
        "true_labels": true_labels,
        "predicted": predicted,
        "ari": scores.get("ARI"),
        "nmi": scores.get("NMI"),
        "resolution": str(resolution),
        "results_dir": results_dir,
    }


def resolution_label(resolution):
    """1000000 -> '1 Mb', 250000 -> '250 kb', matching how the paper names bin sizes."""
    resolution = int(resolution)
    if resolution % 1_000_000 == 0:
        return f"{resolution // 1_000_000} Mb"
    return f"{resolution // 1000} kb"


def panel_title(resolution, chrom):
    where = "" if chrom in ("all", "chrall", "all_higashi") else f", {chrom}"
    return f"Higashi, {resolution_label(resolution)}{where}"


def render(run, output, resolution, chrom, quiet=False):
    fig, stats = figure_for_single_method(
        run["true_labels"], run["predicted"], title=panel_title(resolution, chrom)
    )
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
    """Draw every result present under results/: each resolution and chromosome."""
    paths = sorted((HERE / "results").glob("*/*/cluster_results.txt"))
    if not paths:
        raise FileNotFoundError(f"No results under {HERE / 'results'}")
    drawn = 0
    for path in paths:
        chrom, resolution = path.parent.name, path.parent.parent.name
        try:
            run = load_assignments(resolution, chrom)
        except ValueError as error:
            # a run whose file carries no per-cell table, e.g. an interrupted job
            print(f"skipping {path}: {error}")
            continue
        render(run, path.parent / "cluster_sankey.png", resolution, chrom, quiet=True)
        drawn += 1
    print(f"drew {drawn} alluvial plots under {HERE / 'results'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="1000000",
                        help="bin size in bp, e.g. 250000, 500000, 1000000")
    parser.add_argument("--chrom", default="all")
    parser.add_argument("--output", default=None,
                        help="default: cluster_sankey.png beside the results")
    parser.add_argument("--all", action="store_true",
                        help="draw every resolution and chromosome present")
    args = parser.parse_args()

    if args.all:
        run_all()
        return

    run = load_assignments(args.resolution, args.chrom)
    out = args.output or run["results_dir"] / "cluster_sankey.png"
    render(run, out, args.resolution, args.chrom)


if __name__ == "__main__":
    main()
