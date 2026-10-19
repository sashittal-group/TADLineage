#!/usr/bin/env python3
"""
Whole-TAD agreement between our edge merge/split calls and diffDomain.

This is the "updated" version of the diffDomain comparison. The original
analysis only looked at the source TADs that *our* pipeline flags as
merge/split and asked whether diffDomain agrees on those. This version instead
looks at **every** source TAD on each tree edge:

  * our label for a source TAD is one of {split, merge, none},
  * diffDomain's label for the same TAD is one of {split, merge, none},

and reports how often the two labels agree over all source TADs. Because every
TAD is considered, this now captures the case where our pipeline says a TAD is
NOT part of a merge/split but diffDomain says it is (and vice versa). The
headline number is a plain accuracy: agreeing TADs / total source TADs.

diffDomain is run honestly: the two-sample test is applied to every source TAD
region, BH adjustment is done across all of them, and the classifier sees the
full source and target TAD lists. No event-specific p-value biasing or target
weighting is applied.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
HEMATO_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = HEMATO_DIR.parent
DIFFDOMAIN_DIR = PROJECT_ROOT / "diffDomain" / "diffdomain-py3"

sys.path.append(str(PROJECT_ROOT))

from tad_sequence import transform_sequence
from tad_utilities import parse_interval, read_tad_intervals


VARIANT_TOPOLOGY = {
    "inferred": "internal",
    "inferred_given_root": "given_root",
    "inferred_given_root_weighted": "given_root",
    "inferred_given_root_weighted_updated": "given_root",
}

LABELS = ("none", "split", "merge")


def resolve_hemato_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    alt = HEMATO_DIR / candidate
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"Missing hemato file: {path}")


def resolution_to_bin_size(resolution: str) -> int:
    text = resolution.strip().lower()
    if text.endswith("kb"):
        return int(text[:-2]) * 1000
    if text.endswith("mb"):
        return int(text[:-2]) * 1_000_000
    return int(text)


def load_tads(path: str | Path) -> set[tuple[int, int]]:
    intervals = read_tad_intervals(str(path))
    return {parse_interval(interval) for interval in intervals}


def inferred_tad_path(
    outdir: str | Path,
    node: str,
    leaf1: str,
    leaf2: str,
    chrom: str,
    res: str,
    tolerance: int,
    tad_kind: str,
) -> Path:
    suffix = "" if tad_kind == "all" else f"_{tad_kind}"
    return HEMATO_DIR / outdir / node / f"chr{chrom}" / res / f"inferred_{leaf1}_{leaf2}_{tolerance}{suffix}.tad"


def edge_input_paths(
    variant: str,
    leaf1: str,
    leaf2: str,
    chrom: str,
    res: str,
    tolerance: int,
    tad_kind: str,
    outdir: str | Path,
) -> dict[str, dict]:
    clp_path = resolve_hemato_path(HEMATO_DIR / "CLP" / f"chr{chrom}" / res / "tads.tad")
    leaf1_path = resolve_hemato_path(HEMATO_DIR / leaf1 / f"chr{chrom}" / res / "tads.tad")
    leaf2_path = resolve_hemato_path(HEMATO_DIR / leaf2 / f"chr{chrom}" / res / "tads.tad")
    mpp_inferred = inferred_tad_path(outdir, "MPP", leaf1, leaf2, chrom, res, tolerance, tad_kind)
    cmp_inferred = inferred_tad_path(outdir, "CMP", leaf1, leaf2, chrom, res, tolerance, tad_kind)

    edges = {
        "MPP-CLP": {"source_node": "MPP", "target_node": "CLP", "source_tads": mpp_inferred, "target_tads": clp_path},
        "MPP-CMP": {"source_node": "MPP", "target_node": "CMP", "source_tads": mpp_inferred, "target_tads": cmp_inferred},
        f"CMP-{leaf1}": {"source_node": "CMP", "target_node": leaf1, "source_tads": cmp_inferred, "target_tads": leaf1_path},
        f"CMP-{leaf2}": {"source_node": "CMP", "target_node": leaf2, "source_tads": cmp_inferred, "target_tads": leaf2_path},
    }

    if variant != "inferred":
        lt_path = resolve_hemato_path(HEMATO_DIR / "LT" / f"chr{chrom}" / res / "tads.tad")
        edges = {
            "LT-MPP": {"source_node": "LT", "target_node": "MPP", "source_tads": lt_path, "target_tads": mpp_inferred},
            **edges,
        }
    return edges


def matrix_path(node: str, chrom: str, res: str) -> Path:
    cleaned = HEMATO_DIR / node / f"chr{chrom}" / res / "hic_matrix.cleaned.txt"
    raw = HEMATO_DIR / node / f"chr{chrom}" / res / "hic_matrix.txt"
    if cleaned.exists():
        return cleaned
    if raw.exists():
        return raw
    raise FileNotFoundError(f"Missing Hi-C matrix for {node} chr{chrom} {res}: {cleaned}")


def interval_to_json(interval: tuple[int, int]) -> dict[str, int]:
    return {"start": int(interval[0]), "end": int(interval[1])}


def event_to_json(edge_name: str, op: dict, event_id: int) -> dict:
    kind = op["op"]
    if kind == "split":
        source_tads = [op["source"]]
        target_tads = list(op["targets"])
        anchor = op["source"]
    elif kind == "merge":
        source_tads = list(op["sources"])
        target_tads = [op["target"]]
        anchor = op["target"]
    else:
        raise ValueError(f"Unsupported event op: {kind}")

    return {
        "event_id": f"{edge_name}:{kind}:{event_id}",
        "edge": edge_name,
        "event_type": kind,
        "source_tads": [interval_to_json(iv) for iv in source_tads],
        "target_tads": [interval_to_json(iv) for iv in target_tads],
        "anchor_tad": interval_to_json(anchor),
        "cost": float(op.get("cost", 0.0)),
        "boundary_ops": op.get("boundary_ops", []),
    }


def summarize_edge(source_tads: set[tuple[int, int]], target_tads: set[tuple[int, int]], tolerance: int, edge_name: str) -> dict:
    operations, total_cost, final_intervals = transform_sequence(
        list(source_tads),
        list(target_tads),
        Ct=1.0,
        Cb=1.0,
        tolerance=tolerance,
    )

    events = []
    counts = {"split": 0, "merge": 0, "delete": 0, "insert": 0}
    for op in operations:
        kind = op.get("op")
        if kind in counts:
            counts[kind] += 1
        if kind in {"split", "merge"}:
            events.append(event_to_json(edge_name, op, len(events) + 1))

    return {
        "source_tads": len(source_tads),
        "target_tads": len(target_tads),
        "split_events": counts["split"],
        "merge_events": counts["merge"],
        "delete_events": counts["delete"],
        "insert_events": counts["insert"],
        "total_events": counts["split"] + counts["merge"],
        "operation_count": len(operations),
        "sequence_cost": total_cost,
        "final_tads": len(final_intervals),
        "events": events,
    }


def our_source_tad_labels(
    source_tads: set[tuple[int, int]],
    events: list[dict],
) -> dict[tuple[int, int], str]:
    """Label every source TAD with our call: split, merge, or none.

    A source TAD is labelled 'split' if it is the source of a split event, and
    'merge' if it is one of the sources of a merge event. Everything else that
    our pipeline leaves untouched (matched, deleted, ...) is 'none'.
    """
    labels: dict[tuple[int, int], str] = {tad: "none" for tad in source_tads}
    for event in events:
        event_type = event["event_type"]
        for tad in event["source_tads"]:
            interval = (int(tad["start"]), int(tad["end"]))
            if interval in labels:
                labels[interval] = event_type
    return labels


def read_dense_matrix(path: Path) -> list[list[float]]:
    matrix = []
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                matrix.append([float(value) for value in stripped.split()])
    return matrix


def write_sparse_region(dense: list[list[float]], interval: tuple[int, int], output_path: Path) -> None:
    start, end = interval
    n = len(dense)
    row_start = max(1, start)
    row_end = min(end, n)
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["bin1", "bin2", "count"])
        for i_bin in range(row_start, row_end + 1):
            row = dense[i_bin - 1]
            col_end = min(end, len(row))
            for j_bin in range(max(i_bin, start), col_end + 1):
                value = row[j_bin - 1]
                if math.isfinite(value) and value != 0:
                    writer.writerow([i_bin, j_bin, value])


def bh_adjust(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    adjusted = [math.nan] * n
    valid = [(idx, p) for idx, p in enumerate(pvalues) if p is not None and math.isfinite(p)]
    if not valid:
        return adjusted
    valid.sort(key=lambda item: item[1], reverse=True)
    running = 1.0
    m = len(valid)
    for rank_from_end, (idx, pvalue) in enumerate(valid, start=1):
        rank = m - rank_from_end + 1
        running = min(running, pvalue * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def write_tad_table(
    tads: Iterable[tuple[int, int]],
    chrom: str,
    output_path: Path,
) -> int:
    row_count = 0
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["chr", "start", "end"])
        for start, end in sorted(tads):
            writer.writerow([f"chr{chrom}", start, end])
            row_count += 1
    return row_count


def run_diffdomain_on_source_tads(
    source_tads: set[tuple[int, int]],
    source_matrix: Path,
    target_matrix: Path,
    target_tads: set[tuple[int, int]],
    chrom: str,
    res: str,
    workdir: Path,
    min_nbin: int,
    missing_filter: float,
    diffdomain_alpha: float,
    classifier_limit: int,
    remote: int,
) -> dict:
    """Run diffDomain's two-sample test + classifier over ALL source TADs."""
    if not source_tads:
        return {
            "status": "skipped_no_source_tads",
            "source_tads_tested": 0,
            "significant_tads": 0,
            "classification_file": None,
            "diffdomain_results": [],
            "classification": [],
        }

    utils = load_diffdomain_utils()
    source_dense = read_dense_matrix(source_matrix)
    target_dense = read_dense_matrix(target_matrix)
    tested_tads = sorted(source_tads)
    pvalues = []
    raw_results = []

    with tempfile.TemporaryDirectory(prefix="diffdomain_regions_", dir=str(workdir)) as temp_name:
        temp_dir = Path(temp_name)
        for start, end in tested_tads:
            source_sparse = temp_dir / f"source_{start}_{end}.tsv"
            target_sparse = temp_dir / f"target_{start}_{end}.tsv"
            write_sparse_region(source_dense, (start, end), source_sparse)
            write_sparse_region(target_dense, (start, end), target_sparse)
            result = utils.comp2domins_by_twtest(
                chrn=f"chr{chrom}",
                start=int(start),
                end=int(end),
                reso=1,
                hicnorm="NONE",
                fhic0=str(source_sparse),
                fhic1=str(target_sparse),
                min_nbin=int(min_nbin),
                f=float(missing_filter),
            )
            pvalue = float(result[5]) if result[5] == result[5] else math.nan
            pvalues.append(pvalue)
            raw_results.append(
                {
                    "chr": result[0],
                    "start": int(result[1]),
                    "end": int(result[2]),
                    "region": result[3],
                    "stat": None if result[4] != result[4] else float(result[4]),
                    "pvalue": None if result[5] != result[5] else float(result[5]),
                    "bins": None if result[6] != result[6] else float(result[6]),
                }
            )

    adjusted = bh_adjust(pvalues)
    for row, adj in zip(raw_results, adjusted):
        row["adj_pvalue"] = None if adj != adj else float(adj)

    result_file = workdir / "diffdomain_source_tads.tsv"
    with result_file.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            delimiter="\t",
            fieldnames=["chr", "start", "end", "region", "stat", "pvalue", "bins", "adj_pvalue"],
        )
        writer.writeheader()
        writer.writerows(raw_results)

    # diffDomain's classification.py hard-codes significance at adj_pvalue < 0.05.
    # To honour a custom alpha we remap each source TAD's adjusted p-value so that
    # "significant at our alpha" lands just below 0.05 and everything else just
    # above. This is applied uniformly to every source TAD (no per-event bias).
    classifier_result_file = workdir / f"diffdomain_source_tads_classifier_alpha_{diffdomain_alpha:g}.tsv"
    classifier_rows = []
    for row in raw_results:
        classifier_row = dict(row)
        adj_pvalue = classifier_row["adj_pvalue"]
        classifier_row["original_adj_pvalue"] = adj_pvalue
        if adj_pvalue is not None and math.isfinite(adj_pvalue) and adj_pvalue < diffdomain_alpha:
            classifier_row["adj_pvalue"] = 0.049
        elif adj_pvalue is not None:
            classifier_row["adj_pvalue"] = max(0.051, adj_pvalue)
        classifier_rows.append(classifier_row)
    with classifier_result_file.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            delimiter="\t",
            fieldnames=["chr", "start", "end", "region", "stat", "pvalue", "bins", "original_adj_pvalue", "adj_pvalue"],
        )
        writer.writeheader()
        writer.writerows(classifier_rows)

    target_tad_file = workdir / "target_tads_for_classification.tsv"
    target_tad_rows = write_tad_table(target_tads, chrom, target_tad_file)
    classification_file = workdir / "diffdomain_classification.tsv"
    classifier = DIFFDOMAIN_DIR / "classification.py"
    cmd = [
        sys.executable,
        str(classifier),
        "--diff",
        str(classifier_result_file),
        "--tad",
        str(target_tad_file),
        "--out",
        str(classification_file),
        "--limit",
        str(classifier_limit),
        # These coordinates are bin indices (the two-sample test above is run with
        # reso=1), so --remote must also be given in bins. diffDomain's default of
        # 1_000_000 is a base-pair cap that is never reached in bin space, which
        # lets the region-growth loop in classification.py merge an entire
        # chromosome arm and spin indefinitely on large chromosomes.
        "--remote",
        str(remote),
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "diffDomain classification.py failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    class_rows = read_classification(classification_file)
    return {
        "status": "ok",
        "diffdomain_alpha": float(diffdomain_alpha),
        "source_tads_tested": len(tested_tads),
        "significant_tads": sum(
            1 for row in raw_results if row["adj_pvalue"] is not None and row["adj_pvalue"] < diffdomain_alpha
        ),
        "classifier_limit": int(classifier_limit),
        "target_tads_for_classification": len(target_tads),
        "target_tad_rows_for_classification": target_tad_rows,
        "diffdomain_result_file": str(result_file),
        "classifier_result_file": str(classifier_result_file),
        "target_tad_file": str(target_tad_file),
        "classification_file": str(classification_file),
        "diffdomain_results": raw_results,
        "classification": class_rows,
    }


def load_diffdomain_utils():
    utils_path = DIFFDOMAIN_DIR / "utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"diffDomain utils.py not found: {utils_path}")
    sys.path.insert(0, str(DIFFDOMAIN_DIR))
    spec = importlib.util.spec_from_file_location("diffdomain_original_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load diffDomain utilities from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_classification(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            normalized = {key: value for key, value in row.items()}
            for key in ("start", "end", "significant"):
                if key in normalized and normalized[key] not in {"", None}:
                    value = float(normalized[key])
                    normalized[key] = int(value) if math.isfinite(value) else None
            rows.append(normalized)
    return rows


def diffdomain_source_tad_labels(
    source_tads: set[tuple[int, int]],
    classification: list[dict],
) -> dict[tuple[int, int], str]:
    """Label every source TAD with diffDomain's call: split, merge, or none.

    Only significant condition-1 (source-side) classifications count, matching
    the original analysis. Any TAD diffDomain does not call split or merge
    (single, complex, loss, or not significant / not present) is 'none'.
    """
    by_interval: dict[tuple[int, int], set[str]] = {}
    for row in classification:
        if row.get("origin") != "condition1":
            continue
        if row.get("start") is None or row.get("end") is None:
            continue
        if int(row.get("significant") or 0) != 1:
            continue
        key = (int(row["start"]), int(row["end"]))
        by_interval.setdefault(key, set()).add(str(row.get("type", "")).lower())

    labels: dict[tuple[int, int], str] = {}
    for tad in source_tads:
        types = by_interval.get(tad, set())
        if "split" in types:
            labels[tad] = "split"
        elif "merge" in types:
            labels[tad] = "merge"
        else:
            labels[tad] = "none"
    return labels


def compare_all_source_tads(
    source_tads: set[tuple[int, int]],
    our_labels: dict[tuple[int, int], str],
    diffdomain_labels: dict[tuple[int, int], str],
) -> dict:
    """Build the whole-TAD confusion matrix and accuracy for one edge."""
    confusion = {ours: {theirs: 0 for theirs in LABELS} for ours in LABELS}
    per_tad = []
    for tad in sorted(source_tads):
        ours = our_labels.get(tad, "none")
        theirs = diffdomain_labels.get(tad, "none")
        confusion[ours][theirs] += 1
        per_tad.append(
            {
                "start": int(tad[0]),
                "end": int(tad[1]),
                "our_label": ours,
                "diffdomain_label": theirs,
                "agree": ours == theirs,
            }
        )

    total = len(source_tads)
    agree = sum(confusion[label][label] for label in LABELS)
    our_counts = {label: sum(confusion[label][t] for t in LABELS) for label in LABELS}
    diffdomain_counts = {label: sum(confusion[o][label] for o in LABELS) for label in LABELS}

    return {
        "total_source_tads": total,
        "agree": agree,
        "disagree": total - agree,
        "accuracy_percent": (agree / total * 100.0) if total else None,
        "confusion": confusion,
        "our_counts": our_counts,
        "diffdomain_counts": diffdomain_counts,
        # Highlight the two directions of disagreement the updated analysis adds:
        "we_none_diffdomain_event": confusion["none"]["split"] + confusion["none"]["merge"],
        "we_event_diffdomain_none": confusion["split"]["none"] + confusion["merge"]["none"],
        "per_tad": per_tad,
    }


def empty_aggregate() -> dict:
    return {
        "total_source_tads": 0,
        "agree": 0,
        "confusion": {ours: {theirs: 0 for theirs in LABELS} for ours in LABELS},
    }


def add_to_aggregate(aggregate: dict, edge_comparison: dict) -> None:
    aggregate["total_source_tads"] += edge_comparison["total_source_tads"]
    aggregate["agree"] += edge_comparison["agree"]
    for ours in LABELS:
        for theirs in LABELS:
            aggregate["confusion"][ours][theirs] += edge_comparison["confusion"][ours][theirs]


def finalize_aggregate(aggregate: dict) -> dict:
    total = aggregate["total_source_tads"]
    confusion = aggregate["confusion"]
    our_counts = {label: sum(confusion[label][t] for t in LABELS) for label in LABELS}
    diffdomain_counts = {label: sum(confusion[o][label] for o in LABELS) for label in LABELS}
    aggregate["disagree"] = total - aggregate["agree"]
    aggregate["accuracy_percent"] = (aggregate["agree"] / total * 100.0) if total else None
    aggregate["our_counts"] = our_counts
    aggregate["diffdomain_counts"] = diffdomain_counts
    aggregate["we_none_diffdomain_event"] = confusion["none"]["split"] + confusion["none"]["merge"]
    aggregate["we_event_diffdomain_none"] = confusion["split"]["none"] + confusion["merge"]["none"]
    return aggregate


def run_analysis(args: argparse.Namespace) -> dict:
    if args.variant not in VARIANT_TOPOLOGY:
        raise ValueError(f"Unknown variant: {args.variant}")

    edge_map = edge_input_paths(
        args.variant,
        args.leaf1,
        args.leaf2,
        args.chrom,
        args.resolution,
        args.tolerance,
        args.tad_kind,
        args.outdir,
    )
    work_root = Path(args.workdir) if args.workdir else HEMATO_DIR / args.variant / "diffdomainanalysis_updated"
    run_name = f"chr{args.chrom}_{args.resolution}_{args.leaf1}_{args.leaf2}_{args.tolerance}"
    work_root = work_root / run_name
    work_root.mkdir(parents=True, exist_ok=True)

    # classification.py compares region span (here in bins, since reso=1) against
    # --remote. Translate diffDomain's own 1 Mb default into bins for this
    # resolution unless the caller overrides it. A value in the hundreds of bins
    # lets the classifier's region-merge loop run unbounded on TAD-dense
    # chromosomes; the 1 Mb-equivalent (10 bins at 100kb) is both faithful to
    # diffDomain and fast.
    bin_size = resolution_to_bin_size(args.resolution)
    remote_bins = args.remote if args.remote is not None else max(1, round(1_000_000 / bin_size))

    summary_edges = {}
    aggregate = empty_aggregate()

    for edge_name, edge_info in edge_map.items():
        source_path = Path(edge_info["source_tads"])
        target_path = Path(edge_info["target_tads"])
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source TAD file for {edge_name}: {source_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target TAD file for {edge_name}: {target_path}")

        source_tads = load_tads(source_path)
        target_tads = load_tads(target_path)
        # Event detection uses the full TAD lists (unchanged).
        edge_summary = summarize_edge(source_tads, target_tads, args.tolerance, edge_name)
        edge_workdir = work_root / edge_name.replace("-", "_")
        edge_workdir.mkdir(parents=True, exist_ok=True)

        # diffDomain is designed to be run on individual TADs, not on nested
        # whole-arm/whole-chromosome regions. Those oversized regions are what make
        # the two-sample test crawl and, worse, make classification.py's
        # region-growth loop merge an entire arm and hang. Drop anything wider than
        # --max-tad-bins from both the tested source TADs and the target TADs used
        # for classification, and score agreement over the same filtered set.
        max_bins = args.max_tad_bins
        tested_source_tads = {tad for tad in source_tads if (tad[1] - tad[0]) <= max_bins}
        tested_target_tads = {tad for tad in target_tads if (tad[1] - tad[0]) <= max_bins}
        dropped_source = len(source_tads) - len(tested_source_tads)
        dropped_target = len(target_tads) - len(tested_target_tads)

        diffdomain = run_diffdomain_on_source_tads(
            source_tads=tested_source_tads,
            source_matrix=matrix_path(edge_info["source_node"], args.chrom, args.resolution),
            target_matrix=matrix_path(edge_info["target_node"], args.chrom, args.resolution),
            target_tads=tested_target_tads,
            chrom=args.chrom,
            res=args.resolution,
            workdir=edge_workdir,
            min_nbin=args.min_nbin,
            missing_filter=args.missing_filter,
            diffdomain_alpha=args.diffdomain_alpha,
            classifier_limit=args.classifier_limit,
            remote=remote_bins,
        )
        diffdomain["remote_bins"] = int(remote_bins)
        diffdomain["max_tad_bins"] = int(max_bins)
        diffdomain["dropped_oversized_source_tads"] = int(dropped_source)
        diffdomain["dropped_oversized_target_tads"] = int(dropped_target)

        our_labels = our_source_tad_labels(tested_source_tads, edge_summary["events"])
        diffdomain_labels = diffdomain_source_tad_labels(tested_source_tads, diffdomain.get("classification", []))
        comparison = compare_all_source_tads(tested_source_tads, our_labels, diffdomain_labels)
        add_to_aggregate(aggregate, comparison)

        edge_summary.update(
            {
                "source_node": edge_info["source_node"],
                "target_node": edge_info["target_node"],
                "source_file": str(source_path),
                "target_file": str(target_path),
                "source_matrix": str(matrix_path(edge_info["source_node"], args.chrom, args.resolution)),
                "target_matrix": str(matrix_path(edge_info["target_node"], args.chrom, args.resolution)),
                "diffdomain": diffdomain,
                "tad_agreement": comparison,
            }
        )
        summary_edges[edge_name] = edge_summary

    finalize_aggregate(aggregate)

    return {
        "variant": args.variant,
        "topology": VARIANT_TOPOLOGY[args.variant],
        "chrom": args.chrom,
        "resolution": args.resolution,
        "resolution_bin_size": resolution_to_bin_size(args.resolution),
        "leaf1": args.leaf1,
        "leaf2": args.leaf2,
        "tolerance": args.tolerance,
        "tad_kind": args.tad_kind,
        "outdir": args.outdir,
        "diffdomain_package": str(DIFFDOMAIN_DIR),
        "diffdomain_alpha": float(args.diffdomain_alpha),
        "classifier_limit": int(args.classifier_limit),
        "workdir": str(work_root),
        "aggregate": aggregate,
        "edges": summary_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whole-TAD agreement between hemato merge/split calls and diffDomain."
    )
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_TOPOLOGY.keys()))
    parser.add_argument("--leaf1", required=True)
    parser.add_argument("--leaf2", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--tolerance", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tad-kind", default="all", choices=["all", "hq"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--min-nbin", type=int, default=10)
    parser.add_argument("--missing-filter", type=float, default=0.5)
    parser.add_argument(
        "--diffdomain-alpha",
        type=float,
        default=0.05,
        help="Adjusted-pvalue threshold at which a source TAD counts as differential for diffDomain.",
    )
    parser.add_argument(
        "--classifier-limit",
        type=int,
        default=3,
        help="Boundary window passed to diffDomain classification.py --limit (bin units for this pipeline).",
    )
    parser.add_argument(
        "--max-tad-bins",
        type=int,
        default=200,
        help=(
            "Skip source/target TADs wider than this many bins before running the "
            "diffDomain two-sample test and classifier. Excludes nested whole-arm / "
            "whole-chromosome regions that diffDomain is not meant to run on and that "
            "otherwise make the run crawl or hang."
        ),
    )
    parser.add_argument(
        "--remote",
        type=int,
        default=None,
        help=(
            "Region-span cap (in bins) passed to classification.py --remote. Must be "
            "in bin units because this pipeline runs diffDomain with reso=1. If not "
            "given, defaults to diffDomain's own 1 Mb cap expressed in bins for this "
            "resolution (1_000_000 / bin_size, e.g. 10 bins at 100kb). diffDomain's "
            "raw default of 1_000_000 is a base-pair value that never triggers in bin "
            "space and lets the classifier's region-merge loop run unbounded, which "
            "hangs large, TAD-dense chromosomes such as chr1."
        ),
    )
    args = parser.parse_args()
    if args.classifier_limit < 0:
        raise ValueError("--classifier-limit must be >= 0")
    if args.max_tad_bins <= 0:
        raise ValueError("--max-tad-bins must be > 0")
    if args.remote is not None and args.remote <= 0:
        raise ValueError("--remote must be > 0")

    summary = run_analysis(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved diffDomain whole-TAD agreement summary: {output_path}")


if __name__ == "__main__":
    main()
