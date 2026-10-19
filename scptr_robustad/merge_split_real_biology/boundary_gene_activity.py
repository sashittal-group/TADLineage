#!/usr/bin/env python3
"""What is at a reorganised boundary: which genes, and are they the ones that change?

Section 1 established that a boundary the embryo reorganises carries 15 to 34 per cent more
CTCF and more genes than a boundary of the same medians that persists. It did not ask which
genes, and it did not ask whether those genes are doing anything at the transition where the
boundary moved. This script asks both, and the answer to both is no. Two questions, one
script, because they share the machinery and because either alone invites the reader to
assume the other.

1. ``activity`` -- are the genes at reorganised boundaries different genes?
   Per gene, three properties read off the Deng et al. 2014 stage atlas: the mean expression
   over the five Hi-C stages, the dynamic range across them, and the absolute log2 change at
   the very transition the boundary was called at. Reorganised and stable boundaries of the
   *same* two medians are compared, which is the design section 1 needs, and the comparison
   is matched on bin size, on the number of genes in the bin and on the gene density of the
   surrounding megabase. Nothing survives on both callers.

   THE ESTIMATOR IS THE TRAP HERE AND IT COST TWO PASSES. Stratifying and then averaging the
   per-stratum *ratio* of class means inflates the effect without bound as the strata get
   finer -- a ratio of two noisy means is upward biased, so adding the neighbourhood stratum
   moved a 1.07 to a 1.25 and an unstratified null to p = 4e-5. The estimator below is
   additive instead: every bin is expressed as a residual from its own stratum mean, and the
   classes are compared on those residuals, which is unbiased however fine the strata are.
   Read ``delta`` as the class difference in log2 expression units; the per-cent column
   divides it by the grand mean only so it can be read beside section 1's ratios.

2. ``loci`` -- do the two callers agree on *where* the reorganisation happens?
   Section 2 showed merge and split calls are reproducible across callers at chance and
   concluded that no property specific to them could survive. That leaves open whether the
   much larger reorganised class is reproducible, which is what a named locus would need.
   Each caller's whole grid is collapsed to a per-bin reorganisation rate, observed over the
   rate expected from the number of reorganised bins in its own callable region, and the
   most reorganised bins of one caller are compared with those of the other. The null is a
   circular shift within a chromosome, which keeps the spatial autocorrelation of the track
   that a plain hypergeometric would throw away.

   The overlap comes out *below* chance. That is the reason this folder names no genes: the
   class is real and its statistics reproduce, but which bin belongs to it does not, so a
   gene list read off either caller's hotspots would be that caller's noise. It is the same
   lesson the PRC1.6 lead taught at the scale of a handful of loci, established here for the
   whole reorganised class.

Usage:
    python boundary_gene_activity.py                # both analyses, both callers
    python boundary_gene_activity.py --skip-loci
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

from annotate_events import GeneIndex
from coordinates import MM10_SIZES, resolution_to_bp
from gene_sets import (STAGE_ORDER, load_gene_table, load_expression, stage_level,
                       transition_change)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT_DIR = RESULTS / "activity"

EVENT_DIRS = {"RobusTAD": RESULTS / "events", "OnTAD": RESULTS / "events_ontad"}
TRANSITIONS = [f"{a}->{b}" for a, b in zip(STAGE_ORDER, STAGE_ORDER[1:])]

# the window whose gene density a bin is matched on, over and above its own gene count
NEIGHBOURHOOD_BP = 1_000_000
# the grid the locus track is built on; the coarser calls are spread over it
LOCUS_GRAIN = 250_000
# a chromosome enters the class comparison only with this many scored bins in both classes
MIN_BINS_PER_CLASS = 30

ACTIVITY_FIELDS = ["caller", "statistic", "matching", "chromosomes", "above_zero",
                   "delta", "percent_of_mean", "sign_p"]
LOCUS_FIELDS = ["reference", "query", "top_bins", "overlap", "shift_null_mean",
                "fold", "p_greater"]


def sign_test(values):
    """Two-sided exact binomial p that a value is as likely positive as negative."""
    values = [v for v in values if v == v and v != 0.0]
    n = len(values)
    if n == 0:
        return 1.0
    positive = sum(1 for v in values if v > 0)
    k = min(positive, n - positive)
    total = 0
    for i in range(k + 1):
        c = 1
        for j in range(i):
            c = c * (n - j) // (j + 1)
        total += c
    return float(min(1.0, 2 * total / 2 ** n))


def gene_profiles(expression):
    """``{symbol: array over the five Hi-C stages}`` for genes the atlas measures at all."""
    profiles = {}
    for symbol in expression:
        levels = [stage_level(expression, symbol, stage) for stage in STAGE_ORDER]
        if all(level is not None for level in levels):
            profiles[symbol] = np.asarray(levels, float)
    return profiles


def classify(event_type):
    """The three boundary classes of section 1, from the ``event_type`` of a boundary row."""
    if event_type == "stable":
        return "stable"
    if event_type in ("split", "merge"):
        return "ms"
    return "gainloss"


def collect_bins(events_dir, profiles, changes, index):
    """One record per boundary bin: its class, its stratum, and its gene statistics.

    A bin with no measured gene is dropped rather than scored as zero -- an empty bin says
    nothing about which genes a boundary sits on, and counting it as zero would let the gene
    *density* of section 1 leak back in as a difference in gene *identity*.
    """
    records = []
    for path in sorted(Path(events_dir).glob("*/*/tol_*_boundaries.tsv")):
        resolution, chrom = path.parts[-3], path.parts[-2]
        with open(path) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        grouped = collections.defaultdict(lambda: collections.defaultdict(set))
        for row in rows:
            grouped[row["transition"]][classify(row["event_type"])].add(
                (int(row["start"]), int(row["end"])))
        for transition, sets in grouped.items():
            if len(sets["gainloss"]) < 10 or len(sets["stable"]) < 10:
                continue
            for cls in ("gainloss", "stable"):
                for start, end in sets[cls]:
                    symbols = [s for s in index.in_window(chrom, start, end)
                               if s in profiles]
                    if not symbols:
                        continue
                    density = index.count_in_window(chrom, max(0, start - NEIGHBOURHOOD_BP),
                                                    end + NEIGHBOURHOOD_BP)
                    stratum = (resolution, min(len(symbols), 6), min(density // 10, 6))
                    records.append({
                        "chrom": chrom, "cls": cls, "stratum": stratum,
                        "mean_level": float(np.mean([profiles[s].mean() for s in symbols])),
                        "dynamic_range": float(np.mean(
                            [profiles[s].max() - profiles[s].min() for s in symbols])),
                        "abs_change_matched": float(np.mean(
                            [changes[s][transition] for s in symbols])),
                    })
    return records


def activity_contrast(records, statistic, matching):
    """Reorganised minus stable, on residuals from the stratum mean; chromosome replicates.

    ``matching`` is ``resolution`` (bin size only) or ``full`` (bin size, the bin's own gene
    count and the gene density of the surrounding megabase).
    """
    def key(record):
        return record["stratum"] if matching == "full" else record["stratum"][0]

    means = collections.defaultdict(list)
    for record in records:
        means[key(record)].append(record[statistic])
    means = {k: float(np.mean(v)) for k, v in means.items()}
    grand = float(np.mean([r[statistic] for r in records]))

    by_chrom = collections.defaultdict(lambda: collections.defaultdict(list))
    for record in records:
        by_chrom[record["chrom"]][record["cls"]].append(
            record[statistic] - means[key(record)])

    deltas = []
    for chrom, classes in by_chrom.items():
        if (len(classes["gainloss"]) < MIN_BINS_PER_CLASS
                or len(classes["stable"]) < MIN_BINS_PER_CLASS):
            continue
        deltas.append(float(np.mean(classes["gainloss"]) - np.mean(classes["stable"])))
    if len(deltas) < 4:
        return None
    return {
        "statistic": statistic, "matching": matching, "chromosomes": len(deltas),
        "above_zero": sum(1 for d in deltas if d > 0),
        "delta": round(float(np.mean(deltas)), 4),
        "percent_of_mean": round(100 * float(np.mean(deltas)) / grand, 2) if grand else "",
        "sign_p": round(sign_test(deltas), 6),
    }


def locus_track(events_dir):
    """Per 250 kb bin, the reorganised-boundary hits observed and expected, over the grid.

    The expectation is analytic rather than permuted: within one run and transition the null
    places the same number of reorganised bins uniformly over that run's callable region, so
    a bin's expected hits are the reorganised count over the callable count, summed over
    every run whose callable region contains it. That is the same relocation null the rest of
    the folder permutes, evaluated in closed form because the window is one bin wide.
    """
    chroms = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
    observed = {c: np.zeros(MM10_SIZES[c] // LOCUS_GRAIN + 2) for c in chroms}
    expected = {c: np.zeros(MM10_SIZES[c] // LOCUS_GRAIN + 2) for c in chroms}
    for path in sorted(Path(events_dir).glob("*/*/tol_*_boundaries.tsv")):
        chrom = path.parts[-2]
        if chrom not in observed:
            continue
        callable_path = path.with_name(path.name.replace("_boundaries", "_callable"))
        if not callable_path.exists():
            continue
        with open(callable_path) as handle:
            callable_rows = list(csv.DictReader(handle, delimiter="\t"))
        if not callable_rows:
            continue
        low = min(int(r["start"]) for r in callable_rows) // LOCUS_GRAIN
        high = min(max(int(r["end"]) for r in callable_rows) // LOCUS_GRAIN,
                   len(observed[chrom]) - 1)
        if high <= low:
            continue
        with open(path) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        per_transition = collections.defaultdict(set)
        for row in rows:
            if row["event_type"] == "stable":
                continue
            per_transition[row["transition"]].add((int(row["start"]), int(row["end"])))
        for bins in per_transition.values():
            if len(bins) < 10:
                continue
            cells = set()
            for start, end in bins:
                cells.update(range(max(low, start // LOCUS_GRAIN),
                                   min(high, (end - 1) // LOCUS_GRAIN) + 1))
            for cell in cells:
                observed[chrom][cell] += 1
            expected[chrom][low:high + 1] += len(cells) / (high - low + 1)
    return observed, expected


def locus_overlap(reference, query, top_bins, permutations, seed=0):
    """Do the two callers' most reorganised bins coincide more than a shifted track would?"""
    ref_obs, ref_exp = reference
    qry_obs, qry_exp = query
    rows, lengths = [], []
    for chrom in ref_obs:
        mask = (ref_exp[chrom] > 0) & (qry_exp[chrom] > 0)
        if mask.sum() < 40:
            continue
        rows.append((ref_obs[chrom][mask] / ref_exp[chrom][mask],
                     qry_obs[chrom][mask] / qry_exp[chrom][mask]))
        lengths.append(int(mask.sum()))
    if not rows:
        return None
    a = np.concatenate([r[0] for r in rows])
    b = np.concatenate([r[1] for r in rows])
    top_bins = min(top_bins, len(a) // 4)
    top_a = set(np.argsort(-a)[:top_bins].tolist())
    observed = len(top_a & set(np.argsort(-b)[:top_bins].tolist()))

    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for i in range(permutations):
        shifted, cursor = b.copy(), 0
        for n in lengths:
            if n > 60:
                shifted[cursor:cursor + n] = np.roll(b[cursor:cursor + n],
                                                     int(rng.integers(20, n - 20)))
            cursor += n
        null[i] = len(top_a & set(np.argsort(-shifted)[:top_bins].tolist()))
    mean = float(null.mean())
    return {
        "top_bins": top_bins, "overlap": observed,
        "shift_null_mean": round(mean, 2),
        "fold": round(observed / mean, 3) if mean else "",
        "p_greater": round(float((np.sum(null >= observed) + 1) / (permutations + 1)), 5),
    }


def write_table(path, fields, rows):
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
    parser.add_argument("--skip-activity", action="store_true")
    parser.add_argument("--skip-loci", action="store_true")
    parser.add_argument("--top-bins", nargs="*", type=int, default=[200, 500, 1000])
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or OUT_DIR)

    if not args.skip_activity:
        expression, _ = load_expression()
        profiles = gene_profiles(expression)
        changes = {s: {t: abs(transition_change(expression, s, t)) for t in TRANSITIONS}
                   for s in profiles}
        index = GeneIndex(load_gene_table())
        rows = []
        for caller, events_dir in EVENT_DIRS.items():
            if not events_dir.exists():
                print(f"skipping {caller}: {events_dir} is missing")
                continue
            records = collect_bins(events_dir, profiles, changes, index)
            for statistic in ("mean_level", "dynamic_range", "abs_change_matched"):
                for matching in ("resolution", "full"):
                    result = activity_contrast(records, statistic, matching)
                    if result:
                        rows.append({"caller": caller, **result})
        write_table(out_dir / "boundary_gene_activity.tsv", ACTIVITY_FIELDS, rows)
        print()
        print("genes at reorganised minus genes at stable boundaries of the same medians")
        print(f"{'caller':>10}{'statistic':>20}{'matching':>12}{'chrom':>7}{'up':>5}"
              f"{'delta':>9}{'%':>8}{'p':>10}")
        for row in rows:
            print(f"{row['caller']:>10}{row['statistic']:>20}{row['matching']:>12}"
                  f"{row['chromosomes']:>7}{row['above_zero']:>5}{row['delta']:>+9.4f}"
                  f"{row['percent_of_mean']:>+8.1f}{row['sign_p']:>10.5f}")

    if not args.skip_loci:
        missing = [c for c, d in EVENT_DIRS.items() if not d.exists()]
        if missing:
            print(f"skipping the locus test: {', '.join(missing)} events are missing")
            return
        tracks = {c: locus_track(d) for c, d in EVENT_DIRS.items()}
        rows = []
        for top_bins in args.top_bins:
            result = locus_overlap(tracks["RobusTAD"], tracks["OnTAD"], top_bins,
                                   args.permutations)
            if result:
                rows.append({"reference": "RobusTAD", "query": "OnTAD", **result})
        write_table(out_dir / "reorganised_locus_overlap.tsv", LOCUS_FIELDS, rows)
        print()
        print("most reorganised bins, RobusTAD against OnTAD, circular-shift null")
        print(f"{'top bins':>10}{'overlap':>9}{'null':>9}{'fold':>7}{'p':>9}")
        for row in rows:
            print(f"{row['top_bins']:>10}{row['overlap']:>9}{row['shift_null_mean']:>9.1f}"
                  f"{row['fold']:>7.2f}{row['p_greater']:>9.4f}")


if __name__ == "__main__":
    main()
