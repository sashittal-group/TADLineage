#!/usr/bin/env python3
"""
Summarize merge/split events along tree edges using already inferred TAD files.

This script does not re-infer TADs. It reads the existing outputs from one of:
  - inferred
  - inferred_given_root
  - inferred_given_root_weighted
  - inferred_given_root_weighted_updated

and compares parent/child interval sets edge by edge with the tolerance-aware
transform_sequence implementation from tad_sequence.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

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
    alt = SCRIPT_DIR / candidate
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"Missing hemato TAD file: {path}")


def parse_path(path: str | Path) -> tuple[str, str, str]:
    normalized = str(path).rstrip("/")
    match = re.search(r"([^/]+)/chr([^/]+)/([^/]+)/tads\.tad$", normalized)
    if not match:
        raise ValueError(f"Path not in expected format '<NODE>/chr<CHROM>/<RES>/tads.tad': {path}")
    return match.group(1), match.group(2), match.group(3)


def load_tads(path: str | Path) -> set[tuple[int, int]]:
    intervals = read_tad_intervals(str(path))
    return {parse_interval(interval) for interval in intervals}


def infer_file_suffix(tad_kind: str) -> str:
    return "" if tad_kind == "all" else f"_{tad_kind}"


def inferred_tad_path(outdir: str | Path, node: str, leaf1: str, leaf2: str, chrom: str, res: str, tolerance: int, tad_kind: str) -> Path:
    suffix = infer_file_suffix(tad_kind)
    return Path(outdir) / node / f"chr{chrom}" / res / f"inferred_{leaf1}_{leaf2}_{tolerance}{suffix}.tad"


def build_tree_edges(variant: str, leaf1: str, leaf2: str) -> list[tuple[str, str]]:
    if variant == "inferred":
        return [
            ("MPP", "CLP"),
            ("MPP", "CMP"),
            ("CMP", leaf1),
            ("CMP", leaf2),
        ]

    if variant in {"inferred_given_root", "inferred_given_root_weighted", "inferred_given_root_weighted_updated"}:
        return [
            ("LT", "MPP"),
            ("MPP", "CLP"),
            ("MPP", "CMP"),
            ("CMP", leaf1),
            ("CMP", leaf2),
        ]

    raise ValueError(f"Unsupported variant: {variant}")


def edge_input_paths(variant: str, leaf1: str, leaf2: str, chrom: str, res: str, tolerance: int, tad_kind: str, outdir: str | Path):
    root = SCRIPT_DIR
    clp_path = resolve_hemato_path(root / "CLP" / f"chr{chrom}" / res / "tads.tad")
    leaf1_path = resolve_hemato_path(root / leaf1 / f"chr{chrom}" / res / "tads.tad")
    leaf2_path = resolve_hemato_path(root / leaf2 / f"chr{chrom}" / res / "tads.tad")

    mpp_inferred = inferred_tad_path(outdir, "MPP", leaf1, leaf2, chrom, res, tolerance, tad_kind)
    cmp_inferred = inferred_tad_path(outdir, "CMP", leaf1, leaf2, chrom, res, tolerance, tad_kind)

    if variant == "inferred":
        return {
            "MPP-CLP": (mpp_inferred, clp_path),
            "MPP-CMP": (mpp_inferred, cmp_inferred),
            f"CMP-{leaf1}": (cmp_inferred, leaf1_path),
            f"CMP-{leaf2}": (cmp_inferred, leaf2_path),
        }

    lt_path = resolve_hemato_path(root / "LT" / f"chr{chrom}" / res / "tads.tad")
    return {
        "LT-MPP": (lt_path, mpp_inferred),
        "MPP-CLP": (mpp_inferred, clp_path),
        "MPP-CMP": (mpp_inferred, cmp_inferred),
        f"CMP-{leaf1}": (cmp_inferred, leaf1_path),
        f"CMP-{leaf2}": (cmp_inferred, leaf2_path),
    }


def summarize_edge(source_tads, target_tads, tolerance: int):
    operations, total_cost, final_intervals = transform_sequence(
        list(source_tads),
        list(target_tads),
        Ct=1.0,
        Cb=1.0,
        tolerance=tolerance,
    )

    op_counts = {"split": 0, "merge": 0, "delete": 0, "insert": 0}
    for op in operations:
        kind = op.get("op")
        if kind in op_counts:
            op_counts[kind] += 1

    return {
        "source_tads": len(source_tads),
        "target_tads": len(target_tads),
        "split_events": op_counts["split"],
        "merge_events": op_counts["merge"],
        "delete_events": op_counts["delete"],
        "insert_events": op_counts["insert"],
        "total_events": op_counts["split"] + op_counts["merge"],
        "operation_count": len(operations),
        "sequence_cost": total_cost,
        "final_tads": len(final_intervals),
    }


def run_analysis(
    variant: str,
    leaf1: str,
    leaf2: str,
    chrom: str,
    res: str,
    tolerance: int,
    outdir: str,
    tad_kind: str = "all",
):
    if variant not in VARIANT_TOPOLOGY:
        raise ValueError(f"Unknown variant: {variant}")

    edge_map = edge_input_paths(variant, leaf1, leaf2, chrom, res, tolerance, tad_kind, outdir)

    summary_edges = {}
    for label, (source_path, target_path) in edge_map.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source TAD file for {label}: {source_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target TAD file for {label}: {target_path}")

        source_tads = load_tads(source_path)
        target_tads = load_tads(target_path)
        summary_edges[label] = {
            "source_file": str(source_path),
            "target_file": str(target_path),
            **summarize_edge(source_tads, target_tads, tolerance),
        }

    summary = {
        "variant": variant,
        "topology": VARIANT_TOPOLOGY[variant],
        "chrom": chrom,
        "resolution": res,
        "leaf1": leaf1,
        "leaf2": leaf2,
        "tolerance": tolerance,
        "tad_kind": tad_kind,
        "outdir": outdir,
        "edges": summary_edges,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize merge/split events on hemato tree edges.")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_TOPOLOGY.keys()))
    parser.add_argument("--leaf1", required=True, help="leaf sample name, e.g. MEP/MKP/MK")
    parser.add_argument("--leaf2", required=True, help="leaf sample name, e.g. GMP/G")
    parser.add_argument("--chrom", required=True, help="chromosome without chr prefix")
    parser.add_argument("--resolution", required=True, help="e.g. 250kb")
    parser.add_argument("--tolerance", type=int, required=True, help="matching tolerance in bins")
    parser.add_argument("--outdir", required=True, help="base inferred directory, e.g. inferred_given_root")
    parser.add_argument("--tad-kind", default="all", choices=["all", "hq"], help="use *_all or *_hq inferred files")
    parser.add_argument("--output", required=True, help="output json path")
    args = parser.parse_args()

    summary = run_analysis(
        variant=args.variant,
        leaf1=args.leaf1,
        leaf2=args.leaf2,
        chrom=args.chrom,
        res=args.resolution,
        tolerance=args.tolerance,
        outdir=args.outdir,
        tad_kind=args.tad_kind,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved edge merge/split summary: {output_path}")


if __name__ == "__main__":
    main()