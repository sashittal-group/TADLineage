#!/usr/bin/env python3
"""Can any anchor measure make these boundaries look like CTCF anchors, and is the
2-cell dip about anchors or about territory?

`stage_boundary_maturation.py` scores a boundary by counting mESC CTCF peaks in the bin
it lands in. At 250 kb that bin holds about 27 peaks whatever you do with it, so the
measure cannot see a boundary anchor -- it sees how CTCF-rich the neighbourhood is. This
script asks whether a sharper definition of an anchor rescues it, and answers no.

Five tracks are scored at the same boundaries against the same relocation null, ordered
from generic to loop-extrusion-specific:

    genes        gene density, the thing every active track turns out to track
    ctcf         all mESC CTCF peaks, what the figure plots
    rad21        RAD21 cohesin, ten times sparser than CTCF
    ctcf_rad21   CTCF peaks that overlap a RAD21 peak -- the loop-anchor definition
    h3k9me3      repressive and gene-poor, the specificity control

plus `distance`, the kilobases from the bin centre to the nearest co-bound site, where a
fold below 1 means the boundaries sit closer to anchors than chance. Reading the table:

  * a fold near 1 across the anchor tracks means the resolution, not the caller, is what
    stops a boundary looking like an anchor. Widening the anchor definition does not help
    and the more specific tracks come out slightly worse, not better.
  * the 2-cell dip is the same size on gene density as on the co-bound sites, so it is an
    A-compartment effect and no protein should be named for it. H3K9me3 not moving is
    what stops that reading from being vacuous.

Chromosomes are the replicates and the six tolerances of a bin size are averaged first,
because they are re-analyses of one clustering rather than six independent ones.

Usage:
    python boundary_anchor_resolution.py --resolution 250kb
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotate_events import GeneIndex
from call_median_events import DEFAULT_RUN_DIR, stage_medians
from cell_calls import CHROMS
from coordinates import resolution_to_bp
from expression_analysis import build_draws
from gene_sets import load_gene_table
from stage_boundary_maturation import (STAGE_ORDER, boundary_bins, covered_bins,
                                       prefix_counts)

REFERENCE = HERE / "reference"
OUT_DIR = HERE / "results" / "anchor"
ONTAD_RUNS = (REPO_ROOT / "scptr_ontad" / "results_updated" / "scptr_runs_all_cells")
CALLERS = {"OnTAD": ONTAD_RUNS, "RobusTAD": DEFAULT_RUN_DIR}
TRACKS = ["genes", "ctcf", "rad21", "ctcf_rad21", "h3k9me3"]
MEASURES = TRACKS + ["distance"]
TOLERANCES = range(6)

FIELDS = ["caller", "resolution", "chrom", "stage", "measure", "tolerances", "fold"]
TEST_FIELDS = ["caller", "resolution", "measure", "dip", "chroms", "negative", "sign_p"]


def read_bed(name):
    """``{chrom: [(start, end), ...]}`` sorted, or ``{}`` when the file is absent."""
    path = REFERENCE / name
    if not path.exists():
        return {}
    peaks = {}
    for line in open(path):
        fields = line.split("\t")
        if len(fields) >= 3:
            peaks.setdefault(fields[0], []).append((int(fields[1]), int(fields[2])))
    return {chrom: sorted(v) for chrom, v in peaks.items()}


def co_bound(ctcf, rad21, chrom):
    """CTCF peaks on one chromosome that overlap a RAD21 peak."""
    partners = rad21.get(chrom, [])
    if not partners:
        return []
    starts = np.array([s for s, _ in partners])
    ends = np.array([e for _, e in partners])
    out = []
    for start, end in ctcf.get(chrom, []):
        cut = int(np.searchsorted(starts, end))
        if cut and ends[:cut].max() >= start:
            out.append((start, end))
    return out


def nearest_kb(midpoints, centres):
    """Distance in kb from each centre to the nearest midpoint."""
    if len(midpoints) == 0:
        return np.full(len(centres), np.nan)
    marks = np.asarray(sorted(midpoints))
    right = np.clip(np.searchsorted(marks, centres), 1, len(marks) - 1)
    return np.minimum(np.abs(centres - marks[right - 1]),
                      np.abs(marks[right] - centres)) / 1000.0


def sign_p(negative, total):
    """Two-sided sign test on how many chromosomes carry a dip."""
    if not total:
        return float("nan")
    tail = min(negative, total - negative)
    return min(1.0, 2 * sum(comb(total, i) for i in range(tail + 1)) / 2 ** total)


def chromosome(run_dir, resolution, chrom, midpoints, permutations, seed):
    """``{(stage, measure): [fold per tolerance]}`` for one chromosome."""
    resolution_bp = resolution_to_bp(resolution)
    folds = collections.defaultdict(list)
    for tolerance in TOLERANCES:
        path = Path(run_dir) / resolution / chrom / f"tol_{tolerance}.json"
        if not path.exists():
            continue
        by_stage, _ = stage_medians(json.load(open(path)))
        if len(by_stage) < len(STAGE_ORDER):
            continue
        union = set().union(*[covered_bins(v) for v in by_stage.values()])
        n_bins = max(union) + 2
        tracks = {name: np.diff(prefix_counts(marks, resolution_bp, n_bins))
                  for name, marks in midpoints.items()}
        pool = {chrom: np.array(sorted(union), dtype=np.int64)}
        rng = np.random.default_rng(seed)
        for stage in STAGE_ORDER:
            bins = sorted(boundary_bins(by_stage[stage]) & union)
            if len(bins) < 5:
                continue
            draws = build_draws([(chrom, b, 1) for b in bins], pool, permutations, rng)
            if not draws:
                continue
            (_, (starts, _widths, choices)), = draws.items()
            for name in TRACKS:
                null = float(tracks[name][choices].mean(axis=1).mean())
                if null > 0:
                    folds[(stage, name)].append(float(tracks[name][starts].mean()) / null)
            # the distance measure uses the co-bound sites, the sharpest anchor available
            observed = nearest_kb(midpoints["ctcf_rad21"], (starts - 0.5) * resolution_bp)
            null_kb = nearest_kb(midpoints["ctcf_rad21"],
                                 (choices - 0.5) * resolution_bp).mean(axis=1)
            if np.isfinite(observed).any() and null_kb.mean() > 0:
                folds[(stage, "distance")].append(float(np.nanmean(observed))
                                                  / float(null_kb.mean()))
    return folds


def build(resolution, permutations, seed, chroms=None, callers=None, quiet=False):
    """Per-chromosome folds and the dip tests, for every caller."""
    ctcf, rad21 = read_bed("ctcf_mesc_mm10.bed"), read_bed("rad21_mm10.bed")
    if not ctcf:
        print("No CTCF peaks in reference/. Run fetch_reference_data.py --with-ctcf.")
        return [], []
    h3k9me3 = read_bed("h3k9me3_mm10.bed")
    index = GeneIndex(load_gene_table())
    middle = lambda peaks: [(s + e) // 2 for s, e in peaks]

    rows, tests = [], []
    for caller in (callers or CALLERS):
        per_chrom = {}
        for chrom in (chroms or CHROMS):
            midpoints = {
                "genes": index.positions.get(chrom, []),
                "ctcf": middle(ctcf.get(chrom, [])),
                "rad21": middle(rad21.get(chrom, [])),
                "ctcf_rad21": middle(co_bound(ctcf, rad21, chrom)),
                "h3k9me3": middle(h3k9me3.get(chrom, [])),
            }
            folds = chromosome(CALLERS[caller], resolution, chrom, midpoints,
                               permutations, seed)
            if not folds:
                continue
            per_chrom[chrom] = {key: float(np.mean(v)) for key, v in folds.items()}
            for (stage, measure), values in folds.items():
                rows.append({"caller": caller, "resolution": resolution, "chrom": chrom,
                             "stage": stage, "measure": measure,
                             "tolerances": len(values),
                             "fold": round(float(np.mean(values)), 6)})

        for measure in MEASURES:
            dips = []
            for values in per_chrom.values():
                others = [values[(s, measure)] for s in STAGE_ORDER
                          if s != "2-cell" and (s, measure) in values]
                if ("2-cell", measure) in values and len(others) == 4:
                    dips.append(values[("2-cell", measure)] - float(np.mean(others)))
            if not dips:
                continue
            negative = sum(d < 0 for d in dips)
            tests.append({"caller": caller, "resolution": resolution,
                          "measure": measure, "dip": round(float(np.mean(dips)), 6),
                          "chroms": len(dips), "negative": negative,
                          "sign_p": round(sign_p(negative, len(dips)), 6)})
        if not quiet:
            report(caller, resolution, rows, tests)
    return rows, tests


def report(caller, resolution, rows, tests):
    """The table the report quotes: stage trajectory, then the dip and its sign test."""
    means = collections.defaultdict(list)
    for row in rows:
        if row["caller"] == caller:
            means[(row["measure"], row["stage"])].append(row["fold"])
    print(f"\n{caller}  {resolution}  fold over relocated boundaries, "
          f"tolerances averaged, mean over chromosomes")
    print(f"  {'measure':12}" + "".join(f"{s.split('-')[0]:>8}" for s in STAGE_ORDER)
          + f"{'2c dip':>9}{'chrom':>8}{'sign p':>9}")
    lookup = {t["measure"]: t for t in tests if t["caller"] == caller}
    for measure in MEASURES:
        line = "".join(f"{np.mean(means[(measure, s)]):8.3f}"
                       if means[(measure, s)] else f"{'-':>8}" for s in STAGE_ORDER)
        test = lookup.get(measure)
        tail = (f"{test['dip']:9.3f}{f'{test['negative']}/{test['chroms']}':>8}"
                f"{test['sign_p']:9.4f}") if test else ""
        print(f"  {measure:12}{line}{tail}")


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default="250kb")
    parser.add_argument("--caller", nargs="+", default=None, choices=sorted(CALLERS))
    parser.add_argument("--chrom", nargs="+", default=None)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    rows, tests = build(args.resolution, args.permutations, args.seed,
                        chroms=args.chrom, callers=args.caller)
    if not rows:
        return
    out = Path(args.out_dir or OUT_DIR)
    write(out / "boundary_anchor.tsv", FIELDS, rows)
    write(out / "anchor_dip_tests.tsv", TEST_FIELDS, tests)
    print(f"\nWrote {out / 'boundary_anchor.tsv'} and {out / 'anchor_dip_tests.tsv'}")


if __name__ == "__main__":
    main()
