#!/usr/bin/env python3
"""Do the boundaries of a stage median become more CTCF-marked as the embryo develops?

Every other analysis in this folder is about the *difference* between two medians: which
domains appeared, which disappeared, which pair merged. That framing throws away the
medians themselves, and the medians are the part of the clustering that is best supported
-- a median is the consensus of every cell assigned to a stage, so it averages over the
sparsity that makes any individual cell's calls unreliable.

This script asks a question about the medians directly. Mammalian TAD boundaries are marked
by CTCF and are gene dense, so the CTCF density at a median's boundaries measures how far
that median's domain structure has taken on the canonical form. Running that measure stage
by stage along

    1-cell -> 2-cell -> 4-cell -> 8-cell -> 64-cell

turns it into a developmental trajectory, which is a published prediction: Du et al. 2017
and Ke et al. 2017 both report that TADs are obscure in the zygote and early cleavage
embryo and are progressively consolidated over preimplantation development. If our medians
carry that signal, CTCF density at their boundaries should climb across the chain.

Three tracks are scored. ``ctcf`` is ENCODE CTCF peak midpoints, ``genes`` is refGene
transcription start sites, and ``expression`` is those start sites weighted by the level of
each gene in the Deng et al. 2014 atlas *at the stage being measured*, so a stage's
boundaries are scored against that stage's own transcriptome. The expression track is not
independent of the gene track: dividing the expression fold by the gene fold of the same
cell leaves no trend at 500 kb or 1 Mb, so transcription at these boundaries rises because
there are more genes on them, not because the genes on them are more active.

The null is the same relocation null the rest of the folder uses, and the design is a
within-run comparison: the five stages of one (chromosome, bin size, tolerance) run are
scored against relocations drawn from the *same* pool of callable bins, so a difference
between two stages cannot come from which chromosome, which caller or which region of the
genome is involved. Three pools are offered because the obvious objection is that the
caller looked at different places in different stages:

``union``
    bins covered by any of the five medians. The pool the rest of the folder uses.
``own``
    bins covered by that stage's own median. Controls for a stage whose domains sit in a
    systematically different part of the chromosome from the others.
``common``
    bins covered by all five medians, with the observed boundaries restricted to the same
    region. The strictest of the three: every stage is then scored on identical territory.

Chromosomes are the replicates. The six tolerances of a bin size are re-analyses of one
clustering, so they are averaged within a (bin size, chromosome) before any test, and the
three bin sizes are re-analyses of one dataset, so each is tested on its own rather than
pooled.

Usage:
    python stage_boundary_maturation.py --permutations 5000
    python stage_boundary_maturation.py --resolution 500kb --tolerance 1 --pools own
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from call_median_events import DEFAULT_RUN_DIR, stage_medians
from coordinates import resolution_to_bp
from expression_analysis import build_draws, window_sums
from gene_sets import (load_ctcf_peaks, load_expression, load_gene_table, load_mark,
                       stage_level)
from annotate_events import GeneIndex

OUT_DIR = HERE / "results" / "stage"
STAGE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
# the trend is read from the 2-cell on: the zygote is a special case, see the report
TREND_STAGES = ["2-cell", "4-cell", "8-cell", "64-cell"]
POOLS = ["union", "own", "common"]
# a run whose common region is smaller than this has too little territory left to score
MIN_COMMON_BINS = 40

FIELDS = [
    "resolution", "chrom", "tolerance", "stage", "pool", "feature", "n_domains",
    "n_boundaries", "mean_domain_bins", "observed_per_bin", "null_per_bin", "fold", "z",
    "p_greater", "p_less",
]

TREND_FIELDS = [
    "resolution", "pool", "feature", "statistic", "chroms", "positive", "mean",
    "wilcoxon_p", "sign_p",
]


def prefix_counts(positions, resolution_bp, n_bins, weights=None):
    """Prefix sums of ``positions`` per 1-based bin, counted or weighted."""
    counts = np.zeros(n_bins + 2)
    for offset, position in enumerate(positions):
        counts[min(n_bins + 1, int(position) // resolution_bp + 1)] += (
            1.0 if weights is None else weights[offset])
    return np.cumsum(counts)


def covered_bins(intervals):
    return {b for a, z in intervals for b in range(int(a), int(z) + 1)}


def boundary_bins(intervals):
    """The bins carrying a domain edge; the same feature the event tables anchor on."""
    return {int(a) for a, _ in intervals} | {int(b) for _, b in intervals}


def score(prefix, draws):
    """Observed rate, null rates: features per bin, over one relocation draw."""
    observed = observed_bins = 0.0
    null_total = null_bins = None
    for _, (starts, widths, choices) in draws.items():
        observed += float(window_sums(prefix, starts, widths).sum())
        observed_bins += float(widths.sum())
        total = window_sums(prefix, choices, widths).sum(axis=1)
        null_total = total if null_total is None else null_total + total
        null_bins = widths.sum() if null_bins is None else null_bins + widths.sum()
    if observed_bins == 0 or null_total is None:
        return None, None
    return observed / observed_bins, null_total / null_bins


def analyse_run(resolution, chrom, tolerance, tracks_for, pools, permutations, seed,
                flank_bins=0, min_boundaries=5, run_dir=None):
    """One (bin size, chromosome, tolerance) run: every stage, pool and feature."""
    resolution_bp = resolution_to_bp(resolution)
    path = Path(run_dir or DEFAULT_RUN_DIR) / resolution / chrom / f"tol_{tolerance}.json"
    if not path.exists():
        return []
    by_stage, _ = stage_medians(json.load(open(path)))
    if len(by_stage) < len(STAGE_ORDER):
        # a run whose clusters do not cover all five stages has no trajectory to read
        return []

    cover = {stage: covered_bins(intervals) for stage, intervals in by_stage.items()}
    union = set().union(*cover.values())
    common = set.intersection(*cover.values())
    if len(common) < MIN_COMMON_BINS:
        common = set()
    rng = np.random.default_rng(seed)
    rows = []
    for stage in STAGE_ORDER:
        tracks = tracks_for(chrom, resolution_bp, max(union), stage)
        intervals = by_stage[stage]
        edges = boundary_bins(intervals)
        sizes = [int(b) - int(a) + 1 for a, b in intervals]
        for pool_name in pools:
            pool_bins = {"union": union, "own": cover[stage], "common": common}[pool_name]
            if not pool_bins:
                continue
            bins = sorted(edges & pool_bins)
            if len(bins) < min_boundaries:
                continue
            pool = {chrom: np.array(sorted(pool_bins), dtype=np.int64)}
            windows = [(chrom, max(1, b - flank_bins), 1 + 2 * flank_bins) for b in bins]
            draws = build_draws(windows, pool, permutations, rng)
            if not draws:
                continue
            for feature, prefix in tracks.items():
                observed, null_rates = score(prefix, draws)
                if observed is None:
                    continue
                null_mean = float(null_rates.mean())
                null_sd = float(null_rates.std(ddof=1)) if len(null_rates) > 1 else 0.0
                rows.append({
                    "resolution": resolution, "chrom": chrom, "tolerance": tolerance,
                    "stage": stage, "pool": pool_name, "feature": feature,
                    "n_domains": len(intervals), "n_boundaries": len(bins),
                    "mean_domain_bins": round(float(np.mean(sizes)), 2),
                    "observed_per_bin": round(observed, 4),
                    "null_per_bin": round(null_mean, 4),
                    "fold": round(observed / null_mean, 4) if null_mean else "",
                    "z": round((observed - null_mean) / null_sd, 2) if null_sd else "",
                    "p_greater": float((np.sum(null_rates >= observed) + 1)
                                       / (permutations + 1)),
                    "p_less": float((np.sum(null_rates <= observed) + 1)
                                    / (permutations + 1)),
                })
    return rows


def wilcoxon_signed_rank(values):
    """Two-sided Wilcoxon signed-rank p, zeros dropped; falls back to the sign test."""
    values = np.asarray([v for v in values if v == v and v != 0.0], dtype=float)
    if len(values) < 6:
        return sign_test(values)
    try:
        from scipy import stats
    except ImportError:
        return sign_test(values)
    return float(stats.wilcoxon(values).pvalue)


def sign_test(values):
    """Two-sided exact binomial p that a value is as likely positive as negative."""
    values = [v for v in values if v == v and v != 0.0]
    n = len(values)
    if n == 0:
        return 1.0
    positive = sum(1 for v in values if v > 0)
    k = min(positive, n - positive)
    tail = sum(_choose(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def _choose(n, k):
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def spearman(values):
    """Spearman rho of ``values`` against their own order, i.e. the monotone trend."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    order = np.argsort(np.argsort(values)) + 1.0
    index = np.arange(1.0, n + 1)
    a, b = order - order.mean(), index - index.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def trend_tests(rows):
    """The stage trend of each (bin size, pool, feature), chromosomes as replicates."""
    by_cell = {}
    for row in rows:
        if row["fold"] == "":
            continue
        key = (row["resolution"], row["pool"], row["feature"], row["chrom"])
        by_cell.setdefault(key, {}).setdefault(row["stage"], []).append(float(row["fold"]))

    grouped = {}
    for (resolution, pool, feature, chrom), stages in by_cell.items():
        # the six tolerances of a bin size are one clustering re-analysed: average them
        folds = {stage: float(np.mean(values)) for stage, values in stages.items()}
        if not all(stage in folds for stage in TREND_STAGES):
            continue
        grouped.setdefault((resolution, pool, feature), []).append(folds)

    out = []
    for (resolution, pool, feature), cells in sorted(grouped.items()):
        deltas = [c["64-cell"] - c["2-cell"] for c in cells]
        rhos = [spearman([c[s] for s in TREND_STAGES]) for c in cells]
        for name, values in (("64-cell minus 2-cell", deltas),
                             ("spearman rho over 2->64", rhos)):
            out.append({
                "resolution": resolution, "pool": pool, "feature": feature,
                "statistic": name, "chroms": len(values),
                "positive": sum(1 for v in values if v > 0),
                "mean": round(float(np.mean(values)), 4),
                "wilcoxon_p": round(wilcoxon_signed_rank(values), 6),
                "sign_p": round(sign_test(values), 6),
            })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default=None, help="default: every bin size")
    parser.add_argument("--chrom", default=None)
    parser.add_argument("--tolerance", type=int, default=None)
    parser.add_argument("--pools", nargs="+", default=POOLS, choices=POOLS)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--marks", nargs="*", default=[],
                        help="extra reference/ chromatin tracks to score beside ctcf and "
                             "genes, e.g. h3k27me3; scored as peak midpoints, same null")
    parser.add_argument("--flank-bins", type=int, default=0,
                        help="widen every boundary window by this many bins on each side")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    peaks = load_ctcf_peaks()
    if peaks is None:
        print("No CTCF peaks in reference/. Run fetch_reference_data.py --with-ctcf.")
        return
    index = GeneIndex(load_gene_table())

    expression, _ = load_expression()

    # Extra chromatin tracks. These are mESC and identical at every stage, exactly like
    # ctcf, so any stage difference they show comes from where the boundaries moved.
    extra = {}
    for name in args.marks:
        intervals = load_mark(name)
        if intervals is None:
            print(f"No reference/ track for {name}; run fetch_reference_data.py "
                  f"--with-chromatin.")
            return
        extra[name] = intervals

    cache = {}

    def tracks_for(chrom, resolution_bp, n_bins, stage):
        # CTCF and gene density do not depend on the stage; the expression track does,
        # because a stage's boundaries are scored against that stage's own transcriptome
        key = (chrom, resolution_bp, n_bins, stage)
        if key not in cache:
            positions = index.positions.get(chrom, [])
            symbols = index.symbols.get(chrom, [])
            levels = [stage_level(expression, symbol, stage) or 0.0 for symbol in symbols]
            cache[key] = {
                "ctcf": prefix_counts([(s + e) // 2 for s, e in peaks.get(chrom, [])],
                                      resolution_bp, n_bins),
                "genes": prefix_counts(positions, resolution_bp, n_bins),
                "expression": prefix_counts(positions, resolution_bp, n_bins, levels),
            }
            for name, intervals in extra.items():
                cache[key][name] = prefix_counts(
                    [(s + e) // 2 for s, e in intervals.get(chrom, [])],
                    resolution_bp, n_bins)
        return cache[key]

    run_dir = Path(args.run_dir or DEFAULT_RUN_DIR)
    resolutions = ([args.resolution] if args.resolution
                   else sorted(p.name for p in run_dir.iterdir() if p.is_dir()))

    rows = []
    for resolution in resolutions:
        chroms = ([args.chrom] if args.chrom
                  else sorted(p.name for p in (run_dir / resolution).glob("chr*")))
        for chrom in chroms:
            tolerances = ([args.tolerance] if args.tolerance is not None
                          else sorted(int(p.stem.split("_")[1])
                                      for p in (run_dir / resolution / chrom).glob("tol_*.json")))
            for tolerance in tolerances:
                rows.extend(analyse_run(resolution, chrom, tolerance, tracks_for,
                                        args.pools, args.permutations, args.seed,
                                        flank_bins=args.flank_bins, run_dir=run_dir))
        print(f"  {resolution}: {len(rows)} rows so far", flush=True)

    out_dir = Path(args.out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    enrichment_path = out_dir / "stage_boundary_enrichment.tsv"
    with open(enrichment_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {enrichment_path} ({len(rows)} rows)")

    tests = trend_tests(rows)
    trend_path = out_dir / "stage_trend_tests.tsv"
    with open(trend_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TREND_FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tests)
    print(f"wrote {trend_path} ({len(tests)} tests)")

    print()
    print("CTCF density at median boundaries, fold over relocations, union pool")
    header = f"{'bin size':10}" + "".join(f"{s:>10}" for s in STAGE_ORDER)
    print(header)
    for resolution in resolutions:
        line = f"{resolution:10}"
        for stage in STAGE_ORDER:
            folds = [float(r["fold"]) for r in rows
                     if r["resolution"] == resolution and r["stage"] == stage
                     and r["pool"] == "union" and r["feature"] == "ctcf"
                     and r["fold"] != ""]
            line += f"{np.mean(folds):>10.3f}" if folds else f"{'-':>10}"
        print(line)

    print()
    print("Trend from the 2-cell stage on, chromosomes as replicates")
    for row in tests:
        if row["statistic"] != "spearman rho over 2->64":
            continue
        print(f"  {row['resolution']:8} {row['pool']:8} {row['feature']:11} "
              f"rho={row['mean']:+.3f}  {row['positive']}/{row['chroms']} chromosomes  "
              f"p={row['wilcoxon_p']:.2g}")


if __name__ == "__main__":
    main()
