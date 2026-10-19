#!/usr/bin/env python3
"""Does positional tolerance rescue the boundary signal?

Single-cell Hi-C is sparse, so a boundary the caller puts at bin 10 in one stage median may
be the same boundary it puts at bin 13 in the next. If that jitter is what flattened the
gene and CTCF tests, then allowing two positions a few bins apart to count as one should
bring the signal back. This runs that experiment on all three places tolerance can enter:

    ctcf   -- widen every boundary window by +-k bins and re-run the CTCF density test
    genes  -- the same, for every curated and derived gene set
    loci   -- pad by k bp when deciding whether two configurations called the same event,
              which is what decides how many loci are recovered at more than one bin size

The gene set arm is reported with the nesting respected. Six boundary tolerances of one bin
size are re-analyses of one clustering, so they are averaged before testing, leaving three
bin sizes by four transitions, twelve cells, and a sign test on how many of them exceed 1.
Each fold is also divided by the ``all_genes`` fold of the same cell, so a window set that
is gene poor overall does not read as depleted for every set.

Read the output together with ``boundary_profile.py``. The profile says whether there is a
displaced signal for a tolerance to find; this says what happens to the numbers when the
tolerance is applied. A tolerance that widens a window always drags its fold toward 1,
because the window grows toward the genome average, so a fold that merely creeps upward is
dilution rather than recovery. Recovery would look like a fold that rises above 1, or a
count of reproducible loci that keeps climbing.

Usage:
    python tolerance_sweep.py
    python tolerance_sweep.py --flanks 0 1 2 3 5 --slops 0 500000 1000000 2000000
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

import consensus_events
import ctcf_analysis
import enrichment as enrichment_module
from annotate_events import GeneIndex
from gene_sets import load_ctcf_peaks, load_gene_table

HERE = Path(__file__).resolve().parent
EVENTS_DIR = HERE / "results" / "events"
OUT_DIR = HERE / "results" / "tolerance"

BOUNDARY_SETS = ["boundary_gained", "boundary_lost", "boundary_stable",
                 "split_boundary", "merge_boundary"]


def ctcf_sweep(flanks, permutations, max_gap_bins, seed):
    peaks = load_ctcf_peaks()
    if peaks is None:
        return []
    pairs = sorted({(path.parent.parent.name, int(path.name.split("_")[1]))
                    for path in EVENTS_DIR.glob("*/*/tol_*_events.tsv")})
    rows = []
    for flank in flanks:
        for resolution, tolerance in pairs:
            for row in ctcf_analysis.analyse(resolution, tolerance, peaks,
                                             permutations=permutations,
                                             max_gap_bins=max_gap_bins, seed=seed,
                                             flank_bins=flank):
                row["flank_bins"] = flank
                rows.append(row)
        print(f"  ctcf flank {flank}: {len(rows)} rows so far")
    return rows


def gene_sweep(flanks, permutations, max_gap_bins, seed):
    genes = load_gene_table()
    index = GeneIndex(genes)
    gene_sets = enrichment_module.load_all_gene_sets(genes)
    pairs = sorted({(path.parent.parent.name, int(path.name.split("_")[1]))
                    for path in EVENTS_DIR.glob("*/*/tol_*_events.tsv")})
    rows = []
    for flank in flanks:
        for resolution, tolerance in pairs:
            for row in enrichment_module.test_configuration(
                    resolution, tolerance, index, gene_sets, permutations=permutations,
                    max_gap_bins=max_gap_bins, seed=seed, flank_bins=flank):
                row["flank_bins"] = flank
                rows.append(row)
        print(f"  genes flank {flank}: {len(rows)} rows so far")
    return rows


def nested_gene_test(rows):
    """Sign test per (gene set, feature set, flank), tolerances averaged within bin size."""
    from scipy import stats

    folds = collections.defaultdict(list)
    density = collections.defaultdict(list)
    for row in rows:
        if not row["fold"]:
            continue
        cell = (row["flank_bins"], row["feature_set"], row["resolution"],
                row["transition"])
        if row["gene_set"] == "all_genes":
            density[cell].append(float(row["fold"]))
        folds[(row["gene_set"],) + cell].append(float(row["fold"]))

    grouped = collections.defaultdict(list)
    for (gene_set, flank, feature, resolution, transition), values in folds.items():
        cell = (flank, feature, resolution, transition)
        if cell not in density:
            continue
        base = np.mean(density[cell])
        if base == 0:
            continue
        grouped[(gene_set, feature, flank)].append(np.mean(values) / base)

    out = []
    for (gene_set, feature, flank), values in sorted(grouped.items()):
        values = np.asarray(values)
        n = len(values)
        above = int((values > 1).sum())
        sign = (stats.binomtest(above, n, 0.5, alternative="greater").pvalue
                if n else float("nan"))
        out.append({
            "gene_set": gene_set, "feature_set": feature, "flank_bins": flank,
            "cells": n, "cells_above_one": above,
            "mean_fold_density_corrected": round(float(values.mean()), 3),
            "sign_p": round(float(sign), 4),
        })
    return out


def loci_sweep(slops, max_gap_bins):
    calls = consensus_events.read_boundary_calls(max_gap_bins=max_gap_bins)
    rows = []
    for slop in slops:
        clusters = consensus_events.cluster_calls(calls, slop=slop)
        counts = {1: 0, 2: 0, 3: 0}
        for cluster in clusters:
            seen = len({m["resolution"] for m in cluster["members"]})
            counts[seen] = counts.get(seen, 0) + 1
        rows.append({"slop_bp": slop, "loci": len(clusters),
                     "one_resolution": counts.get(1, 0),
                     "two_resolutions": counts.get(2, 0),
                     "three_resolutions": counts.get(3, 0)})
        print(f"  slop {slop:>9} bp: {len(clusters)} loci, "
              f"{counts.get(2, 0)} at two bin sizes, {counts.get(3, 0)} at three")
    return rows


def resolution_gradient(ctcf_rows, flank=0):
    """CTCF fold at each bin size, which separates a biological result from a sparse one.

    The relocation null already controls for where in the genome the windows sit, so a fold
    that depends on bin size is telling us about the calls, not about the genome. Contacts
    per bin rise fourfold from 250 kb to 1 Mb, so a depletion that fades along that axis is
    a property of how reliably boundaries can be placed in sparse data.
    """
    rows = []
    resolutions = sorted({r["resolution"] for r in ctcf_rows},
                         key=lambda r: int(r.replace("kb", "")))
    for name in BOUNDARY_SETS:
        row = {"feature_set": name}
        for resolution in resolutions:
            folds = [r["fold"] for r in ctcf_rows
                     if r["feature_set"] == name and r["resolution"] == resolution
                     and r["flank_bins"] == flank and r["fold"] != ""]
            row[resolution] = round(float(np.mean(folds)), 3) if folds else ""
        rows.append(row)
    return rows, resolutions


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flanks", type=int, nargs="*", default=[0, 1, 2, 3, 5])
    parser.add_argument("--slops", type=int, nargs="*",
                        default=[0, 250000, 500000, 1000000, 2000000, 4000000])
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--skip-genes", action="store_true",
                        help="skip the gene set arm, which is the slow one")
    parser.add_argument("--max-gap-bins", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or OUT_DIR)

    print("CTCF density against boundary window flank")
    ctcf_rows = ctcf_sweep(args.flanks, args.permutations, args.max_gap_bins, args.seed)
    if ctcf_rows:
        write(out_dir / "ctcf_flank_sweep.tsv",
              ["flank_bins"] + ctcf_analysis.FIELDS, ctcf_rows)

    print()
    print("Gene set density against boundary window flank")
    gene_rows = [] if args.skip_genes else gene_sweep(
        args.flanks, args.permutations, args.max_gap_bins, args.seed)
    if gene_rows:
        write(out_dir / "enrichment_flank_sweep.tsv",
              ["flank_bins"] + enrichment_module.RESULT_FIELDS, gene_rows)
        nested = nested_gene_test(gene_rows)
        write(out_dir / "enrichment_flank_nested_test.tsv",
              ["gene_set", "feature_set", "flank_bins", "cells", "cells_above_one",
               "mean_fold_density_corrected", "sign_p"], nested)

    if ctcf_rows:
        gradient, resolutions = resolution_gradient(ctcf_rows)
        write(out_dir / "ctcf_resolution_gradient.tsv",
              ["feature_set"] + resolutions, gradient)
        print()
        print("CTCF fold by bin size at zero flank; sparsity per bin falls left to right")
        print(f"{'feature set':20}" + "".join(f"{r:>10}" for r in resolutions))
        for row in gradient:
            line = "".join(f"{row[r]:>10.3f}" if row[r] != "" else f"{'-':>10}"
                           for r in resolutions)
            print(f"{row['feature_set']:20}{line}")

    print()
    print("Reproducible loci against matching slop")
    loci_rows = loci_sweep(args.slops, args.max_gap_bins)
    write(out_dir / "consensus_slop_sweep.tsv",
          ["slop_bp", "loci", "one_resolution", "two_resolutions", "three_resolutions"],
          loci_rows)

    if not ctcf_rows:
        return
    print()
    print("Mean CTCF fold, pooled over transitions and configurations")
    header = "".join(f"{'flank ' + str(f):>11}" for f in args.flanks)
    print(f"{'feature set':20}{header}")
    for name in BOUNDARY_SETS:
        line = ""
        for flank in args.flanks:
            folds = [r["fold"] for r in ctcf_rows
                     if r["feature_set"] == name and r["flank_bins"] == flank
                     and r["fold"] != ""]
            line += f"{np.mean(folds):>11.3f}" if folds else f"{'-':>11}"
        print(f"{name:20}{line}")

    if not gene_rows:
        return
    print()
    print("Gene sets whose density-corrected fold exceeds 1 in most cells, by flank")
    print(f"{'gene set':32}{'feature':18}{'flank':>7}{'cells>1':>9}{'fold':>8}"
          f"{'sign p':>9}")
    for row in sorted(nested, key=lambda r: r["sign_p"]):
        if row["sign_p"] > 0.2 or row["gene_set"] == "all_genes":
            continue
        print(f"{row['gene_set']:32}{row['feature_set']:18}{row['flank_bins']:>7}"
              f"{str(row['cells_above_one']) + '/' + str(row['cells']):>9}"
              f"{row['mean_fold_density_corrected']:>8.2f}{row['sign_p']:>9.4f}")


if __name__ == "__main__":
    main()
