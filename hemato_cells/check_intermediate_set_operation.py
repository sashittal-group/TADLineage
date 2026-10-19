#!/usr/bin/env python3
"""
Infer CMP/MPP with a deterministic child set operation and LT fixed as root.

For the hemato tree used by the given-root scripts:

    LT
    |
    MPP
    |-- CLP
    |
    CMP
    |-- leaf1
    |-- leaf2

CMP is inferred from leaf1 and leaf2. MPP is inferred from CMP and CLP.
LT is treated as the given root and is included in harmonization/summary, but
is not overwritten by the bottom-up operation.
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
sys.path.append(str(SCRIPT_DIR))

from filtering import harmonize_leaf_boundaries
from tad_utilities import (
    match_tads_within_tolerance,
    parse_interval,
    read_tad_intervals,
    write_tads_to_file,
)


VALID_OPERATIONS = {"union", "intersection"}


def resolve_hemato_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()

    candidate = SCRIPT_DIR / p
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Missing hemato TAD file: {path}")


def parse_path(path: str | Path) -> tuple[str, str, str]:
    normalized = str(path).rstrip("/")
    m = re.search(r"([^/]+)/chr([^/]+)/([^/]+)/tads\.tad$", normalized)
    if not m:
        raise ValueError(f"Path not in expected format '<NODE>/chr<CHROM>/<RES>/tads.tad': {path}")
    return m.group(1), m.group(2), m.group(3)


def filtered_path(path: Path) -> Path:
    return path.with_name(path.stem + "_filtered.tad")


def load_tads_from_file(path: str | Path) -> set[tuple[int, int]]:
    return {parse_interval(s) for s in read_tad_intervals(str(path))}


def compute_metrics(
    pred_set: set[tuple[int, int]],
    ref_set: set[tuple[int, int]],
    tolerance: int,
) -> tuple[float, float, float, list[tuple[tuple[int, int], tuple[int, int]]]]:
    matched_pairs, _, _ = match_tads_within_tolerance(pred_set, ref_set, tolerance)
    matched = len(matched_pairs)
    precision = matched / len(pred_set) if pred_set else 0.0
    recall = matched / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, matched_pairs


def combine_child_tads(
    child_sets: list[set[tuple[int, int]]],
    operation: str,
) -> set[tuple[int, int]]:
    if operation not in VALID_OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")
    if not child_sets:
        return set()

    if operation == "union":
        combined: set[tuple[int, int]] = set()
        for child_set in child_sets:
            combined.update(child_set)
        return combined

    combined = set(child_sets[0])
    for child_set in child_sets[1:]:
        combined.intersection_update(child_set)
    return combined


def infer_and_evaluate(
    leaf1_path: str,
    leaf2_path: str,
    operation: str,
    tolerance: int = 2,
    outbase: str | None = None,
    harmonize: bool = True,
):
    if operation not in VALID_OPERATIONS:
        raise ValueError(f"--operation must be one of {sorted(VALID_OPERATIONS)}")

    leaf1_resolved = resolve_hemato_path(leaf1_path)
    leaf2_resolved = resolve_hemato_path(leaf2_path)
    node1, chrom1, res1 = parse_path(leaf1_resolved)
    node2, chrom2, res2 = parse_path(leaf2_resolved)

    if chrom1 != chrom2 or res1 != res2:
        raise ValueError("Chromosome or resolution mismatch between supplied leaf files.")

    chrom = chrom1
    res = res1
    outbase_path = Path(outbase or f"inferred_{operation}")

    clp_path = resolve_hemato_path(Path("CLP") / f"chr{chrom}" / res / "tads.tad")
    lt_path = resolve_hemato_path(Path("LT") / f"chr{chrom}" / res / "tads.tad")
    true_cmp_path = resolve_hemato_path(Path("CMP") / f"chr{chrom}" / res / "tads.tad")
    true_mpp_path = resolve_hemato_path(Path("MPP") / f"chr{chrom}" / res / "tads.tad")

    if harmonize:
        harmonize_list = [str(leaf1_resolved), str(leaf2_resolved), str(clp_path), str(lt_path)]
        print("Harmonizing boundaries for:", harmonize_list, f"tolerance={tolerance}")
        harmonize_leaf_boundaries(harmonize_list, tolerance)

        leaf1_for_infer = filtered_path(leaf1_resolved)
        leaf2_for_infer = filtered_path(leaf2_resolved)
        clp_for_infer = filtered_path(clp_path)
        lt_for_summary = filtered_path(lt_path)
    else:
        leaf1_for_infer = leaf1_resolved
        leaf2_for_infer = leaf2_resolved
        clp_for_infer = clp_path
        lt_for_summary = lt_path

    for path in (leaf1_for_infer, leaf2_for_infer, clp_for_infer, lt_for_summary):
        if not path.exists():
            raise FileNotFoundError(f"Filtered/input TAD file expected but missing: {path}")

    leaf1_tads = load_tads_from_file(leaf1_for_infer)
    leaf2_tads = load_tads_from_file(leaf2_for_infer)
    clp_tads = load_tads_from_file(clp_for_infer)
    lt_root_tads = load_tads_from_file(lt_for_summary)

    tads_cmp = combine_child_tads([leaf1_tads, leaf2_tads], operation)
    tads_mpp = combine_child_tads([tads_cmp, clp_tads], operation)

    inferred_dir_cmp = outbase_path / "CMP" / f"chr{chrom}" / res
    inferred_dir_mpp = outbase_path / "MPP" / f"chr{chrom}" / res
    inferred_dir_cmp.mkdir(parents=True, exist_ok=True)
    inferred_dir_mpp.mkdir(parents=True, exist_ok=True)

    fname_cmp = inferred_dir_cmp / f"inferred_{node1}_{node2}_{tolerance}.tad"
    fname_mpp = inferred_dir_mpp / f"inferred_{node1}_{node2}_{tolerance}.tad"

    write_tads_to_file(tads_cmp, str(fname_cmp))
    write_tads_to_file(tads_mpp, str(fname_mpp))

    true_cmp = load_tads_from_file(true_cmp_path)
    true_mpp = load_tads_from_file(true_mpp_path)

    cmp_prec, cmp_rec, cmp_f1, _ = compute_metrics(tads_cmp, true_cmp, tolerance)
    mpp_prec, mpp_rec, mpp_f1, _ = compute_metrics(tads_mpp, true_mpp, tolerance)

    summary = {
        "chrom": chrom,
        "resolution": res,
        "leaf1": node1,
        "leaf2": node2,
        "given_root": "LT",
        "method": operation,
        "tolerance": tolerance,
        "harmonized_inputs": harmonize,
        "hemato_inputs": {
            "LT": str(lt_path),
            "CLP": str(clp_path),
            node1: str(leaf1_resolved),
            node2: str(leaf2_resolved),
            "CMP_truth": str(true_cmp_path),
            "MPP_truth": str(true_mpp_path),
        },
        "counts": {
            "LT_root": len(lt_root_tads),
            "CLP_leaf": len(clp_tads),
            f"{node1}_leaf": len(leaf1_tads),
            f"{node2}_leaf": len(leaf2_tads),
        },
        "CMP": {
            "inferred_file": str(fname_cmp),
            "true_file": str(true_cmp_path),
            "precision": cmp_prec,
            "recall": cmp_rec,
            "f1": cmp_f1,
            "n_inferred": len(tads_cmp),
            "n_true": len(true_cmp),
        },
        "MPP": {
            "inferred_file": str(fname_mpp),
            "true_file": str(true_mpp_path),
            "precision": mpp_prec,
            "recall": mpp_rec,
            "f1": mpp_f1,
            "n_inferred": len(tads_mpp),
            "n_true": len(true_mpp),
        },
    }

    print("\n=== Summary ===")
    print(f"Chrom: {chrom}  Res: {res}  Method: {operation}")
    print(f"Leaves used: {node1}, {node2}, CLP | given root: LT")
    print(
        "CMP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, true={})".format(
            cmp_prec, cmp_rec, cmp_f1, len(tads_cmp), len(true_cmp)
        )
    )
    print(
        "MPP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, true={})".format(
            mpp_prec, mpp_rec, mpp_f1, len(tads_mpp), len(true_mpp)
        )
    )
    print("=== End ===\n")

    return summary


def run_cli(operation: str, default_outdir: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Infer CMP/MPP by {operation} of child TADs with fixed LT root."
    )
    parser.add_argument("leaf1", help="path to first leaf .tad, e.g. MKP/chr1/500kb/tads.tad")
    parser.add_argument("leaf2", help="path to second leaf .tad, e.g. GMP/chr1/500kb/tads.tad")
    parser.add_argument("--tolerance", type=int, default=2, help="boundary/TAD tolerance in bins")
    parser.add_argument("--outdir", default=default_outdir, help="output directory base")
    parser.add_argument("--no-harmonize", action="store_true", help="skip boundary harmonization/filtering")
    args = parser.parse_args()

    summary = infer_and_evaluate(
        args.leaf1,
        args.leaf2,
        operation=operation,
        tolerance=args.tolerance,
        outbase=args.outdir,
        harmonize=not args.no_harmonize,
    )

    leaf1_name = parse_path(args.leaf1)[0]
    leaf2_name = parse_path(args.leaf2)[0]
    out_json = (
        Path(args.outdir)
        / "jsons"
        / f"summary_chr{summary['chrom']}_{summary['resolution']}_{leaf1_name}_{leaf2_name}_{args.tolerance}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved summary to", out_json)
