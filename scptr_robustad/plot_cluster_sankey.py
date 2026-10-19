#!/usr/bin/env python3
"""Alluvial ("Sankey") plot of the median clustering: true stage -> predicted cluster.

Reads the JSON summary that ``em_cluster_updated_all_chr.py`` writes for one run and
draws the flow of cells from their known developmental stage into the clusters that the
median clustering returned. See ``cluster_sankey.py`` at the repository root for the
drawing itself.

Cell counts
-----------
A run clusters a subsample of the cells of each stage, because inferring a TAD hierarchy
per cell is expensive and the contact maps are sparse. By default this script scales each
stage up to the number of cells the stage actually has (``available_counts_by_stage`` in
the run summary), replicating the clustered cells in the proportions the run produced, so
that the panel shows the dataset at full size. This is the convention the MDS panel drawn
by ``plot_median_chain.py`` already follows, where the same run is padded to full size
with the cells recorded in ``visual_only_counts_by_stage``.

The scaling is exact within a stage: the proportion of a stage going to each cluster is
what the run produced. It does change the mix of stages, so every number quoted on the
panel -- the ARI, the NMI and the fraction of cells on the matched diagonal -- is computed
on the cells that were actually clustered, never on the scaled drawing. Pass ``--no-scale``
to draw the clustered cells alone, and
``--show-label-note`` to print how many cells were clustered and how many of them had
their stage label used by the preimputation step.

Usage:
    python plot_cluster_sankey.py --resolution 500kb --tolerance 1
    python plot_cluster_sankey.py --all          # every resolution, chromosome, tolerance
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cluster_sankey import (  # noqa: E402
    STAGE_ORDER,
    diagonal_fraction,
    figure_for_single_method,
    print_table,
    summary_line,
)

HERE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = HERE / "results_updated" / "scptr_runs_all_cells"


def run_path(resolution="500kb", chrom="all", tolerance=1, run_dir=None):
    return Path(run_dir or DEFAULT_RUN_DIR) / resolution / chrom / f"tol_{tolerance}.json"


def load_assignments(resolution="500kb", chrom="all", tolerance=1, json_path=None,
                     run_dir=None, scale=True):
    """Per-cell (stage, cluster) plus the scores and the label budget of that run."""
    path = Path(json_path) if json_path else run_path(resolution, chrom, tolerance, run_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No run summary at {path}. Run the clustering for this resolution first."
        )
    summary = json.load(open(path))

    raw_true = list(summary["true_labels"])
    raw_predicted = list(summary["predicted_labels"])
    clustered = len(raw_true)
    available = summary.get("available_counts_by_stage") or {}

    true_labels, predicted = raw_true, raw_predicted
    if scale and available:
        true_labels, predicted = scale_to_available(raw_true, raw_predicted, available)

    return {
        "method": "TADLineage",
        "true_labels": true_labels,
        "predicted": predicted,
        # the cells the run actually clustered, which every quoted score is computed on
        "raw_true_labels": raw_true,
        "raw_predicted": raw_predicted,
        "ari": summary.get("ARI"),
        "nmi": summary.get("NMI"),
        "clustered_cells": clustered,
        "known_cells": len(summary.get("known_indices", [])),
        "scaled": bool(scale and available),
        "resolution": summary.get("resolution", resolution),
        "chromosome": summary.get("chromosome", chrom),
        "tolerance": summary.get("tolerance", tolerance),
        "json_path": path,
    }


def scale_to_available(true_labels, predicted, available):
    """Replicate the clustered cells so each stage is shown at the size it really has.

    Within a stage the cells are distributed over the clusters in the proportions the run
    produced, by largest-remainder rounding, so each stage totals exactly the count the
    run recorded as available and no cluster proportion is distorted by the rounding.
    """
    by_stage = {}
    for stage, cluster in zip(true_labels, predicted):
        by_stage.setdefault(stage, {})
        by_stage[stage][cluster] = by_stage[stage].get(cluster, 0) + 1

    scaled_true, scaled_pred = [], []
    stages = [s for s in STAGE_ORDER if s in by_stage]
    stages += [s for s in by_stage if s not in stages]

    for stage in stages:
        counts = by_stage[stage]
        observed = sum(counts.values())
        target = int(available.get(stage, observed))
        if observed == 0 or target <= 0:
            continue

        clusters = sorted(counts, key=lambda c: (isinstance(c, str), c))
        exact = [counts[c] * target / observed for c in clusters]
        shares = [int(value) for value in exact]
        remainder = target - sum(shares)
        # hand the leftover cells to the clusters with the largest fractional parts
        order = sorted(range(len(clusters)), key=lambda i: -(exact[i] - shares[i]))
        for i in order[:remainder]:
            shares[i] += 1

        for cluster, share in zip(clusters, shares):
            scaled_true.extend([stage] * share)
            scaled_pred.extend([cluster] * share)

    return scaled_true, scaled_pred


def label_note(run):
    """How many cells the run actually clustered, and how many carried a stage label."""
    clustered = run["clustered_cells"]
    known = run["known_cells"]
    if not known:
        return f"{clustered} cells clustered, no stage labels used"
    return (f"{clustered} cells clustered, {known} with the stage label used "
            "for preimputation")


def panel_title(run, method_name="TADLineage"):
    chrom = run.get("chromosome", "all")
    where = "" if chrom in ("all", None) else f", {chrom}"
    return f"{method_name}, {run['resolution']}{where}"


def render(run, output, title=None, show_label_note=False, quiet=False):
    fig, stats = figure_for_single_method(
        run["true_labels"], run["predicted"], title=title or panel_title(run)
    )
    # the drawing may be scaled up to the full stage sizes, so the accuracy is quoted from
    # the cells that were clustered, which is what the ARI and NMI also describe
    scored = dict(stats)
    scored["matched_fraction"] = diagonal_fraction(
        run["raw_true_labels"], run["raw_predicted"]
    )
    fig.text(0.5, 0.015, summary_line(scored, ari=run["ari"], nmi=run["nmi"]),
             ha="center", fontsize=10)
    if show_label_note:
        fig.text(0.5, -0.015, label_note(run), ha="center", fontsize=9, color="0.35")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if not quiet:
        print_table(stats)
        if run["scaled"]:
            print(f"table above is scaled to the full stage sizes; the {run['clustered_cells']} "
                  f"cells actually clustered put "
                  f"{scored['matched_fraction'] * 100:.1f}% on the matched diagonal")
    print(f"wrote {output}")
    return stats


def default_output(run):
    return run["json_path"].with_name(f"{run['json_path'].stem}_sankey.png")


def run_all(run_dir=None, scale=True, show_label_note=False):
    """Draw every run under the results tree: each resolution, chromosome and tolerance."""
    root = Path(run_dir or DEFAULT_RUN_DIR)
    paths = sorted(root.glob("*/*/tol_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run summaries under {root}")
    for path in paths:
        run = load_assignments(json_path=path, scale=scale)
        render(run, default_output(run), show_label_note=show_label_note, quiet=True)
    print(f"drew {len(paths)} alluvial plots under {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="500kb", help="e.g. 250kb, 500kb, 1000kb")
    parser.add_argument("--chrom", default="all")
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--json", default=None, help="a run summary to plot directly")
    parser.add_argument("--run-dir", default=None, help=f"default: {DEFAULT_RUN_DIR}")
    parser.add_argument("--output", default=None,
                        help="default: <run>_sankey.png beside the JSON")
    parser.add_argument("--title", default=None)
    parser.add_argument("--all", action="store_true",
                        help="draw every resolution, chromosome and tolerance")
    parser.add_argument("--no-scale", dest="scale", action="store_false",
                        help="draw only the cells that were clustered, unscaled")
    parser.add_argument("--show-label-note", action="store_true",
                        help="print the clustered-cell and stage-label counts on the panel")
    args = parser.parse_args()

    if args.all:
        run_all(args.run_dir, scale=args.scale, show_label_note=args.show_label_note)
        return

    run = load_assignments(args.resolution, args.chrom, args.tolerance, args.json,
                           args.run_dir, scale=args.scale)
    render(run, args.output or default_output(run), title=args.title,
           show_label_note=args.show_label_note)


if __name__ == "__main__":
    main()
