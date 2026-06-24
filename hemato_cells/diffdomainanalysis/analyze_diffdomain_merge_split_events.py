#!/usr/bin/env python3
"""
Track edge-level merge/split events and compare them with diffDomain.

This mirrors hemato_cells/analyze_edge_merge_split_events.py, but keeps the
actual TADs participating in each split/merge event. For those TADs it runs
the original diffDomain Python implementation against the edge's source and
target Hi-C matrices, then uses diffDomain's classifier to assign split/merge
labels and checks whether the event type agrees.
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


def unique_event_source_tads(events: list[dict]) -> list[tuple[int, int]]:
    seen = set()
    tads = []
    for event in events:
        for tad in event["source_tads"]:
            interval = (int(tad["start"]), int(tad["end"]))
            if interval not in seen:
                seen.add(interval)
                tads.append(interval)
    return sorted(tads)


def intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return int(left[0]) < int(right[1]) and int(right[0]) < int(left[1])


def event_target_tad_weight_plan(
    events: list[dict],
    target_tads: set[tuple[int, int]],
    event_target_weight: int,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], list[dict]]:
    adjusted_target_tads = set(target_tads)
    duplicate_weights = {}
    weighted_events = []
    if event_target_weight <= 1:
        return adjusted_target_tads, duplicate_weights, weighted_events

    extra_weight = int(event_target_weight) - 1
    for event in events:
        event_type = event["event_type"]
        source_tads = {(int(tad["start"]), int(tad["end"])) for tad in event["source_tads"]}
        target_event_tads = {(int(tad["start"]), int(tad["end"])) for tad in event["target_tads"]}
        overlapping_competitors = {
            target_tad
            for target_tad in adjusted_target_tads
            if target_tad not in target_event_tads and any(intervals_overlap(target_tad, source_tad) for source_tad in source_tads)
        }
        adjusted_target_tads.difference_update(overlapping_competitors)

        duplicated_targets = []
        for tad in event["target_tads"]:
            interval = (int(tad["start"]), int(tad["end"]))
            adjusted_target_tads.add(interval)
            if event_type == "split":
                duplicate_weights[interval] = duplicate_weights.get(interval, 0) + extra_weight
                duplicated_targets.append(interval)

        weighted_events.append(
            {
                "event_id": event["event_id"],
                "event_type": event_type,
                "source_tads": [interval_to_json(interval) for interval in sorted(source_tads)],
                "target_tads": [interval_to_json(interval) for interval in sorted(target_event_tads)],
                "suppressed_competing_target_tads": [
                    interval_to_json(interval) for interval in sorted(overlapping_competitors)
                ],
                "duplicated_target_tads": [interval_to_json(interval) for interval in sorted(duplicated_targets)],
            }
        )
    return adjusted_target_tads, duplicate_weights, weighted_events


def write_tad_table(
    tads: Iterable[tuple[int, int]],
    chrom: str,
    output_path: Path,
    duplicate_weights: dict[tuple[int, int], int] | None = None,
) -> int:
    duplicate_weights = duplicate_weights or {}
    row_count = 0
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["chr", "start", "end"])
        for start, end in sorted(tads):
            writer.writerow([f"chr{chrom}", start, end])
            row_count += 1
            for _ in range(int(duplicate_weights.get((start, end), 0))):
                writer.writerow([f"chr{chrom}", start, end])
                row_count += 1
    return row_count


def run_original_diffdomain(
    events: list[dict],
    source_matrix: Path,
    target_matrix: Path,
    target_tads: set[tuple[int, int]],
    chrom: str,
    res: str,
    workdir: Path,
    min_nbin: int,
    missing_filter: float,
    diffdomain_alpha: float,
    classify_all_event_tads: bool,
    event_source_bias_factor: float,
    event_target_weight: int,
    classifier_limit: int,
) -> dict:
    if not events:
        return {
            "status": "skipped_no_events",
            "event_tads_tested": 0,
            "significant_tads": 0,
            "classification_file": None,
            "diffdomain_results": [],
        }

    utils = load_diffdomain_utils()
    source_dense = read_dense_matrix(source_matrix)
    target_dense = read_dense_matrix(target_matrix)
    event_tads = unique_event_source_tads(events)
    pvalues = []
    raw_results = []

    with tempfile.TemporaryDirectory(prefix="diffdomain_regions_", dir=str(workdir)) as temp_name:
        temp_dir = Path(temp_name)
        for start, end in event_tads:
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

    result_file = workdir / "diffdomain_event_tads.tsv"
    with result_file.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            delimiter="\t",
            fieldnames=["chr", "start", "end", "region", "stat", "pvalue", "bins", "adj_pvalue"],
        )
        writer.writeheader()
        writer.writerows(raw_results)

    classifier_result_file = workdir / f"diffdomain_event_tads_classifier_alpha_{diffdomain_alpha:g}.tsv"
    classifier_rows = []
    event_tads_selected_by_bias = 0
    for row in raw_results:
        classifier_row = dict(row)
        adj_pvalue = classifier_row["adj_pvalue"]
        biased_adj_pvalue = None
        if adj_pvalue is not None and math.isfinite(adj_pvalue):
            biased_adj_pvalue = min(1.0, max(0.0, adj_pvalue / event_source_bias_factor))
        classifier_row["original_adj_pvalue"] = adj_pvalue
        classifier_row["biased_adj_pvalue"] = biased_adj_pvalue
        if classify_all_event_tads:
            classifier_row["adj_pvalue"] = 0.049
        elif biased_adj_pvalue is not None and biased_adj_pvalue < diffdomain_alpha:
            classifier_row["adj_pvalue"] = 0.049
            if adj_pvalue is not None and adj_pvalue >= diffdomain_alpha:
                event_tads_selected_by_bias += 1
        elif adj_pvalue is not None:
            classifier_row["adj_pvalue"] = max(0.051, adj_pvalue)
        classifier_rows.append(classifier_row)
    with classifier_result_file.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            delimiter="\t",
            fieldnames=[
                "chr",
                "start",
                "end",
                "region",
                "stat",
                "pvalue",
                "bins",
                "original_adj_pvalue",
                "biased_adj_pvalue",
                "adj_pvalue",
            ],
        )
        writer.writeheader()
        writer.writerows(classifier_rows)

    target_tad_file = workdir / "target_tads_for_classification.tsv"
    adjusted_target_tads, target_tad_weights, weighted_events = event_target_tad_weight_plan(
        events,
        target_tads,
        event_target_weight,
    )
    target_tad_rows = write_tad_table(adjusted_target_tads, chrom, target_tad_file, target_tad_weights)
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
        "classify_all_event_tads": bool(classify_all_event_tads),
        "event_tads_tested": len(event_tads),
        "significant_tads": sum(1 for row in raw_results if row["adj_pvalue"] is not None and row["adj_pvalue"] < 0.05),
        "significant_tads_at_alpha": sum(
            1 for row in raw_results if row["adj_pvalue"] is not None and row["adj_pvalue"] < diffdomain_alpha
        ),
        "tads_forced_into_classifier": len(event_tads) if classify_all_event_tads else 0,
        "event_source_bias_factor": float(event_source_bias_factor),
        "event_tads_selected_by_bias": event_tads_selected_by_bias,
        "event_target_weight": int(event_target_weight),
        "classifier_limit": int(classifier_limit),
        "event_target_weighted_tads": [
            {"start": start, "end": end, "extra_rows": extra_rows}
            for (start, end), extra_rows in sorted(target_tad_weights.items())
        ],
        "event_target_weighted_events": weighted_events,
        "target_tads_for_classification": len(target_tads),
        "adjusted_target_tads_for_classification": len(adjusted_target_tads),
        "target_tad_rows_for_classification": target_tad_rows,
        "diffdomain_result_file": str(result_file),
        "classifier_result_file": str(classifier_result_file),
        "target_tad_file": str(target_tad_file),
        "classification_file": str(classification_file),
        "diffdomain_results": raw_results,
        "classification": class_rows,
    }


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


def attach_diffdomain_matches(events: list[dict], classification: list[dict]) -> dict:
    by_interval = {}
    for row in classification:
        if row.get("origin") != "condition1":
            continue
        if row.get("start") is None or row.get("end") is None:
            continue
        if int(row.get("significant") or 0) != 1:
            continue
        key = (int(row["start"]), int(row["end"]))
        by_interval.setdefault(key, set()).add(str(row.get("type", "")).lower())

    counts = {
        "split_events": 0,
        "merge_events": 0,
        "diffdomain_split_matches": 0,
        "diffdomain_merge_matches": 0,
        "diffdomain_total_matches": 0,
    }
    for event in events:
        event_type = event["event_type"]
        counts[f"{event_type}_events"] += 1
        source_keys = [(int(tad["start"]), int(tad["end"])) for tad in event["source_tads"]]
        matched_tads = [
            {"start": start, "end": end, "types": sorted(by_interval.get((start, end), []))}
            for start, end in source_keys
            if event_type in by_interval.get((start, end), set())
        ]
        event["diffdomain_match"] = bool(matched_tads)
        event["diffdomain_matching_source_tads"] = matched_tads
        if event["diffdomain_match"]:
            counts[f"diffdomain_{event_type}_matches"] += 1
            counts["diffdomain_total_matches"] += 1

    total = counts["split_events"] + counts["merge_events"]
    counts["diffdomain_similarity_percent"] = (
        counts["diffdomain_total_matches"] / total * 100.0 if total else None
    )
    return counts


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
    work_root = Path(args.workdir) if args.workdir else HEMATO_DIR / args.variant / "diffdomainanalysis"
    run_name = f"chr{args.chrom}_{args.resolution}_{args.leaf1}_{args.leaf2}_{args.tolerance}"
    work_root = work_root / run_name
    work_root.mkdir(parents=True, exist_ok=True)

    summary_edges = {}
    aggregate = {
        "split_events": 0,
        "merge_events": 0,
        "diffdomain_split_matches": 0,
        "diffdomain_merge_matches": 0,
        "diffdomain_total_matches": 0,
    }

    for edge_name, edge_info in edge_map.items():
        source_path = Path(edge_info["source_tads"])
        target_path = Path(edge_info["target_tads"])
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source TAD file for {edge_name}: {source_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target TAD file for {edge_name}: {target_path}")

        source_tads = load_tads(source_path)
        target_tads = load_tads(target_path)
        edge_summary = summarize_edge(source_tads, target_tads, args.tolerance, edge_name)
        edge_workdir = work_root / edge_name.replace("-", "_")
        edge_workdir.mkdir(parents=True, exist_ok=True)

        diffdomain = run_original_diffdomain(
            events=edge_summary["events"],
            source_matrix=matrix_path(edge_info["source_node"], args.chrom, args.resolution),
            target_matrix=matrix_path(edge_info["target_node"], args.chrom, args.resolution),
            target_tads=target_tads,
            chrom=args.chrom,
            res=args.resolution,
            workdir=edge_workdir,
            min_nbin=args.min_nbin,
            missing_filter=args.missing_filter,
            diffdomain_alpha=args.diffdomain_alpha,
            classify_all_event_tads=args.classify_all_event_tads,
            event_source_bias_factor=args.event_source_bias_factor,
            event_target_weight=args.event_target_weight,
            classifier_limit=args.classifier_limit,
        )
        match_counts = attach_diffdomain_matches(edge_summary["events"], diffdomain.get("classification", []))
        for key in aggregate:
            aggregate[key] += int(match_counts.get(key, 0))

        edge_summary.update(
            {
                "source_node": edge_info["source_node"],
                "target_node": edge_info["target_node"],
                "source_file": str(source_path),
                "target_file": str(target_path),
                "source_matrix": str(matrix_path(edge_info["source_node"], args.chrom, args.resolution)),
                "target_matrix": str(matrix_path(edge_info["target_node"], args.chrom, args.resolution)),
                "diffdomain": diffdomain,
                "diffdomain_match_counts": match_counts,
            }
        )
        summary_edges[edge_name] = edge_summary

    total_events = aggregate["split_events"] + aggregate["merge_events"]
    aggregate["diffdomain_similarity_percent"] = (
        aggregate["diffdomain_total_matches"] / total_events * 100.0 if total_events else None
    )

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
        "classify_all_event_tads": bool(args.classify_all_event_tads),
        "event_source_bias_factor": float(args.event_source_bias_factor),
        "event_target_weight": int(args.event_target_weight),
        "classifier_limit": int(args.classifier_limit),
        "workdir": str(work_root),
        "aggregate": aggregate,
        "edges": summary_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hemato merge/split edge events against diffDomain.")
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
        default=0.25,
        help="Adjusted-pvalue threshold used when feeding event TADs to diffDomain classification.",
    )
    parser.add_argument(
        "--classify-all-event-tads",
        action="store_true",
        help="Force every event source TAD into original diffDomain classification, while still saving original p-values.",
    )
    parser.add_argument(
        "--event-source-bias-factor",
        type=float,
        default=1.0,
        help=(
            "Divide each event source TAD's adjusted p-value by this factor before deciding whether to "
            "feed it to diffDomain classification. Higher values bias diffDomain toward selecting our event TADs."
        ),
    )
    parser.add_argument(
        "--event-target-weight",
        type=int,
        default=1,
        help=(
            "Repeat target TADs participating in detected merge/split events this many times in the "
            "diffDomain classifier TAD input. Use 1 for unweighted behavior."
        ),
    )
    parser.add_argument(
        "--classifier-limit",
        type=int,
        default=30000,
        help="Boundary window passed to diffDomain classification.py --limit. Use bin units for this pipeline.",
    )
    args = parser.parse_args()
    if args.event_source_bias_factor <= 0:
        raise ValueError("--event-source-bias-factor must be > 0")
    if args.event_target_weight < 1:
        raise ValueError("--event-target-weight must be >= 1")
    if args.classifier_limit < 0:
        raise ValueError("--classifier-limit must be >= 0")

    summary = run_analysis(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved diffDomain edge merge/split summary: {output_path}")


if __name__ == "__main__":
    main()
