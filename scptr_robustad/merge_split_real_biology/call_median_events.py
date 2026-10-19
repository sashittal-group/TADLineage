#!/usr/bin/env python3
"""Merge and split events between the cluster medians of consecutive embryonic stages.

Each cluster of a clustering run has a median TAD set, the consensus structure of the
cells assigned to it, and each cluster is matched to the developmental stage most of its
cells come from. Walking the medians along the chain

    1-cell -> 2-cell -> 4-cell -> 8-cell -> 64-cell

and transforming one median into the next with ``tad_sequence.transform_sequence`` gives
the domain rearrangements the clustering implies for that transition: which domains split
in two, which pairs merged into one, and which appeared or disappeared outright. This is
the single-cell counterpart of ``hemato_cells/analyze_edge_merge_split_events.py``, which
does the same along the edges of the hematopoietic tree, and it reuses the same
tolerance-aware transformation so that the event definitions match.

Two tables come out of a run, both in mm10 coordinates:

``events.tsv``
    one row per split, merge, domain gain or domain loss, with the domain it happened to
    and the focal boundary that appeared or disappeared inside it.

``boundaries.tsv``
    one row per boundary created or destroyed, which is the feature the downstream gene,
    expression and CTCF analyses are anchored on.

Clusters are matched to stages the way the paper's figure matches them, by the assignment
that puts as many cells as possible on the diagonal, so the stage a median is called by
here is the stage its ribbon flows to in the alluvial plot. A run whose clusters do not
cover all five stages yields only the transitions whose two stages were both matched, and
says so.

Usage:
    python call_median_events.py --resolution 500kb --chrom chr1 --tolerance 1
    python call_median_events.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cluster_sankey import STAGE_ORDER, contingency_table, match_clusters_to_stages
from tad_sequence import (
    _match_boundary_values_with_tolerance,
    transform_sequence,
)
from tad_utilities import parse_interval

from coordinates import interval_to_bp, resolution_to_bp

DEFAULT_RUN_DIR = HERE.parent / "results_updated" / "scptr_runs_all_cells"
DEFAULT_OUT_DIR = HERE / "results" / "events"

EVENT_FIELDS = [
    "resolution", "chrom", "tolerance", "transition", "from_stage", "to_stage",
    "event_type", "region_start", "region_end", "region_bins",
    "focal_start", "focal_end", "focal_bins", "gap_bins", "partner_bins", "cost",
]

CALLABLE_FIELDS = ["resolution", "chrom", "tolerance", "start", "end", "bins"]

BOUNDARY_FIELDS = [
    "resolution", "chrom", "tolerance", "transition", "from_stage", "to_stage",
    "event_type", "change", "side", "bin", "start", "end", "gap_bins", "region_bins",
]


def stage_medians(summary):
    """``{stage: [(a, b), ...]}`` for the clusters matched to a stage, plus the mapping.

    The match is the one the alluvial figure draws: the assignment of clusters to stages
    that maximises the cells on the diagonal.
    """
    medians = summary["medians"]
    table, stages, clusters = contingency_table(
        summary["true_labels"], summary["predicted_labels"]
    )
    _, matched_stage = match_clusters_to_stages(table)

    by_stage, cluster_of_stage = {}, {}
    for column, row in matched_stage.items():
        stage, cluster = stages[row], clusters[column]
        if cluster >= len(medians):
            continue
        by_stage[stage] = [parse_interval(text) for text in medians[cluster]]
        cluster_of_stage[stage] = int(cluster)
    return by_stage, cluster_of_stage


def edge_bins(left, right):
    """The two bins carrying the edges that face each other in ``[a,x]``, ``[y,b]``.

    A domain ``[a, b]`` occupies bins ``a`` through ``b``, so the right edge of the left
    domain sits at the far side of bin ``x`` and the left edge of the right domain at the
    near side of bin ``y``. When the two are adjacent, ``y == x + 1``, those two bins
    straddle a single boundary; when they are not, the split left a gap between the
    daughters and each of its two edges is a boundary in its own right.
    """
    _, x = left
    y, _ = right
    return int(x), int(y)


def gap_between(left, right):
    """Bins lying strictly between two domains: 0 when they are adjacent."""
    _, x = left
    y, _ = right
    return max(0, int(y) - int(x) - 1)


def callable_bins(by_stage):
    """The bins any median of the run covers, merged into runs of consecutive bins.

    This is the stretch of the chromosome where an event could have been called at all:
    RobusTAD calls are not spread evenly along a chromosome, so a random relocation of an
    event has to stay inside this region to be a fair comparison. The permutation nulls of
    ``enrichment.py`` draw from exactly these bins.
    """
    covered = set()
    for intervals in by_stage.values():
        for a, b in intervals:
            covered.update(range(int(a), int(b) + 1))
    if not covered:
        return []

    spans, start = [], None
    previous = None
    for value in sorted(covered):
        if start is None:
            start = value
        elif value != previous + 1:
            spans.append((start, previous))
            start = value
        previous = value
    spans.append((start, previous))
    return spans


def events_for_transition(source, target, tolerance, context):
    """Split, merge, gain and loss records for one median-to-median transition.

    Every event yields one row in the event table and one row per boundary it creates or
    destroys in the boundary table. A boundary row is always a single bin, which is the
    finest position the data supports, so the positional tests downstream never have to
    reason about windows of different widths.
    """
    operations, _, _ = transform_sequence(
        list(source), list(target), Ct=1.0, Cb=1.0, tolerance=tolerance
    )
    resolution_bp = context["resolution_bp"]

    def span(interval):
        return interval_to_bp(interval, resolution_bp)

    def bins_text(*intervals):
        return ";".join(f"{a}-{b}" for a, b in intervals)

    rows, boundary_rows = [], []
    for operation in operations:
        kind = operation["op"]
        row = dict(context["labels"])
        row["cost"] = operation.get("cost", "")

        if kind in ("split", "merge"):
            if kind == "split":
                region = operation["source"]
                left, right = sorted(operation["targets"])
                change = "gained"
            else:
                region = operation["target"]
                left, right = sorted(operation["sources"])
                change = "lost"

            first, second = edge_bins(left, right)
            gap = gap_between(left, right)
            # the stretch the boundary change happened over: from the far edge of the left
            # domain to the near edge of the right one
            focal_start = span((first, first))[1]
            focal_end = span((second, second))[0]
            row.update(event_type=kind, region_bins=bins_text(region),
                       partner_bins=bins_text(left, right),
                       focal_bins=f"{first};{second}", gap_bins=gap)
            row["region_start"], row["region_end"] = span(region)
            row["focal_start"] = min(focal_start, focal_end)
            row["focal_end"] = max(focal_start, focal_end)
            # when the two domains touch in the same bin there is one boundary, not two
            edges = ((("shared_edge", first),) if first == second
                     else (("left_edge", first), ("right_edge", second)))
            for side, position in edges:
                boundary_rows.append(_boundary_row(context, kind, change, side, position,
                                                   region, gap, resolution_bp))

        elif kind in ("insert", "delete"):
            region = operation["interval"]
            event_type = "domain_gain" if kind == "insert" else "domain_loss"
            change = "gained" if kind == "insert" else "lost"
            row.update(event_type=event_type, region_bins=bins_text(region),
                       partner_bins="", focal_bins="", gap_bins="")
            row["region_start"], row["region_end"] = span(region)
            row["focal_start"], row["focal_end"] = "", ""
            # only the edges transform_sequence charged for, that is the ones no other
            # domain of the same median already carries
            for name, position in operation.get("boundary_ops", []):
                side = "left_edge" if name.endswith("start") else "right_edge"
                boundary_rows.append(_boundary_row(context, event_type, change, side,
                                                   position, region, "", resolution_bp))
        else:
            continue
        rows.append(row)

    return rows, boundary_rows


def stable_boundary_rows(source, target, tolerance, context, resolution_bp):
    """Boundaries the two medians share, which is the control the changed ones need.

    A boundary that survives a transition is the natural comparison for one that appears or
    disappears: it comes from the same medians, the same caller and the same chromosome, so
    a difference between the two sets cannot be explained by where TADs were called. Edges
    are matched between the medians with the same tolerance the transformation used.
    """
    source_edges = {a for a, _ in source} | {b for _, b in source}
    target_edges = {a for a, _ in target} | {b for _, b in target}
    matched_source, _ = _match_boundary_values_with_tolerance(
        source_edges, target_edges, tolerance
    )
    rows = []
    for position in sorted(matched_source):
        rows.append(_boundary_row(context, "stable", "stable", "edge", position,
                                  (position, position), "", resolution_bp))
    return rows


def _boundary_row(context, event_type, change, side, position, region, gap, resolution_bp):
    """One boundary that appeared or disappeared, as the single bin that carries it."""
    row = dict(context["labels"])
    position = max(1, int(position))
    row.update(event_type=event_type, change=change, side=side, bin=position,
               gap_bins=gap)
    row["start"], row["end"] = interval_to_bp((position, position), resolution_bp)
    row["region_bins"] = f"{region[0]}-{region[1]}"
    return row


def call_run(json_path, out_dir=None, quiet=False):
    """Every transition of one run, written as an events table and a boundary table."""
    summary = json.load(open(json_path))
    resolution = summary.get("resolution")
    chrom = summary.get("chromosome")
    tolerance = summary.get("tolerance")
    if chrom in (None, "all"):
        raise ValueError(
            f"{json_path} is a genome-wide run. Its bins are laid out with mm9 chromosome "
            "sizes over mm10 data, so genomic coordinates near a chromosome end are not "
            "reliable; call events on the per-chromosome runs instead."
        )
    resolution_bp = resolution_to_bp(resolution)

    by_stage, cluster_of_stage = stage_medians(summary)
    present = [stage for stage in STAGE_ORDER if stage in by_stage]
    missing = [stage for stage in STAGE_ORDER if stage not in by_stage]

    events, boundaries = [], []
    transitions = []
    for from_stage, to_stage in zip(STAGE_ORDER, STAGE_ORDER[1:]):
        if from_stage not in by_stage or to_stage not in by_stage:
            continue
        context = {
            "resolution_bp": resolution_bp,
            "labels": {
                "resolution": resolution, "chrom": chrom, "tolerance": tolerance,
                "transition": f"{from_stage}->{to_stage}",
                "from_stage": from_stage, "to_stage": to_stage,
            },
        }
        rows, boundary_rows = events_for_transition(
            by_stage[from_stage], by_stage[to_stage], tolerance, context
        )
        boundary_rows.extend(stable_boundary_rows(
            by_stage[from_stage], by_stage[to_stage], tolerance, context, resolution_bp
        ))
        events.extend(rows)
        boundaries.extend(boundary_rows)
        transitions.append(f"{from_stage}->{to_stage}")

    out_dir = Path(out_dir or DEFAULT_OUT_DIR) / str(resolution) / str(chrom)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tol_{tolerance}"
    write_table(out_dir / f"{stem}_events.tsv", EVENT_FIELDS, events)
    write_table(out_dir / f"{stem}_boundaries.tsv", BOUNDARY_FIELDS, boundaries)

    spans = callable_bins(by_stage)
    write_table(out_dir / f"{stem}_callable.tsv", CALLABLE_FIELDS, [
        {"resolution": resolution, "chrom": chrom, "tolerance": tolerance,
         "start": interval_to_bp((lo, hi), resolution_bp)[0],
         "end": interval_to_bp((lo, hi), resolution_bp)[1], "bins": f"{lo}-{hi}"}
        for lo, hi in spans
    ])

    manifest = {
        "run": str(json_path),
        "resolution": resolution,
        "chromosome": chrom,
        "tolerance": tolerance,
        "ARI": summary.get("ARI"),
        "stages_matched": present,
        "stages_unmatched": missing,
        "cluster_of_stage": cluster_of_stage,
        "median_sizes": {stage: len(by_stage[stage]) for stage in present},
        "transitions": transitions,
        "callable_bins": sum(hi - lo + 1 for lo, hi in spans),
        "callable_spans": len(spans),
        "counts": count_by_type(events),
    }
    (out_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not quiet:
        print(f"{chrom} {resolution} tol {tolerance}: "
              f"{len(transitions)}/4 transitions, {len(events)} events "
              f"({format_counts(manifest['counts'])})")
        if missing:
            print(f"  no cluster matched: {', '.join(missing)}")
    return manifest


def count_by_type(events):
    counts = {}
    for row in events:
        counts[row["event_type"]] = counts.get(row["event_type"], 0) + 1
    return counts


def format_counts(counts):
    return ", ".join(f"{count} {name}" for name, count in sorted(counts.items())) or "none"


def write_table(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_all(run_dir=None, out_dir=None, resolutions=None, tolerances=None):
    root = Path(run_dir or DEFAULT_RUN_DIR)
    paths = sorted(path for path in root.glob("*/*/tol_*.json")
                   if path.parent.name != "all")
    if resolutions:
        paths = [p for p in paths if p.parent.parent.name in set(resolutions)]
    if tolerances:
        keep = {f"tol_{t}.json" for t in tolerances}
        paths = [p for p in paths if p.name in keep]
    if not paths:
        raise FileNotFoundError(f"No per-chromosome run summaries under {root}")

    total_events, failed = 0, []
    for path in paths:
        try:
            manifest = call_run(path, out_dir, quiet=True)
        except Exception as error:  # a run that never finished, or has no medians
            failed.append((path, error))
            continue
        total_events += sum(manifest["counts"].values())
    print(f"called events for {len(paths) - len(failed)} runs, {total_events} events total")
    for path, error in failed:
        print(f"  skipped {path}: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default="500kb", help="e.g. 250kb, 500kb, 1000kb")
    parser.add_argument("--chrom", default="chr1")
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--json", default=None, help="a run summary to read directly")
    parser.add_argument("--run-dir", default=None, help=f"default: {DEFAULT_RUN_DIR}")
    parser.add_argument("--out-dir", default=None, help=f"default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--all", action="store_true",
                        help="every per-chromosome run, resolution and tolerance")
    parser.add_argument("--resolutions", nargs="*", default=None,
                        help="with --all, restrict to these resolutions")
    parser.add_argument("--tolerances", nargs="*", type=int, default=None,
                        help="with --all, restrict to these tolerances")
    args = parser.parse_args()

    if args.all:
        run_all(args.run_dir, args.out_dir, args.resolutions, args.tolerances)
        return

    path = Path(args.json) if args.json else (
        Path(args.run_dir or DEFAULT_RUN_DIR) / args.resolution / args.chrom
        / f"tol_{args.tolerance}.json"
    )
    call_run(path, args.out_dir)


if __name__ == "__main__":
    main()
