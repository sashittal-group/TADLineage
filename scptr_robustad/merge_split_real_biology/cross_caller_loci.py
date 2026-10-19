#!/usr/bin/env python3
"""Do two TAD callers agree on where the merges and splits are?

Section 12 shows the callers agree on which boundaries are reorganised. This asks the
narrower question the event tables are built for: do they agree on which reorganisations
are splits and merges? Adjacent merge and split calls of each caller are clustered into
loci, and the test counts how many OnTAD loci fall within one slop of a RobusTAD locus.

Null: keep every OnTAD locus on its own chromosome and slide it to a random position inside
the region the medians of that chromosome cover, then recount. That holds the number of
loci, the chromosome and the callable territory fixed and varies only the position.

Usage:
    python cross_caller_loci.py
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT_DIR = RESULTS / "turnover"
FIELDS = ["slop_bp", "reference_loci", "query_loci", "matched", "matched_fraction",
          "null_mean", "fold", "p_greater"]


def load_calls(events_dir, max_gap_bins=1):
    """Focal midpoints of adjacent merge and split events, by chromosome, in bp."""
    calls = collections.defaultdict(list)
    for path in sorted(Path(events_dir).glob("*/*/tol_*_events.tsv")):
        with open(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["event_type"] not in ("split", "merge"):
                    continue
                gap = row["gap_bins"]
                if gap == "" or int(gap) > max_gap_bins:
                    continue
                if row["focal_start"] == "" or row["focal_end"] == "":
                    continue
                calls[row["chrom"]].append(
                    (int(row["focal_start"]) + int(row["focal_end"])) // 2)
    return calls


def cluster(calls, slop):
    """Nearby calls on one chromosome collapse to one locus at their mean position."""
    loci = {}
    for chrom, positions in calls.items():
        out, current = [], []
        for position in sorted(positions):
            if current and position - current[-1] <= slop:
                current.append(position)
            else:
                if current:
                    out.append(int(np.mean(current)))
                current = [position]
        if current:
            out.append(int(np.mean(current)))
        loci[chrom] = out
    return loci


def callable_span(events_dir):
    """``{chrom: (first, last)}`` over every callable table of a caller."""
    spans = collections.defaultdict(lambda: [None, None])
    for path in Path(events_dir).glob("*/*/tol_*_callable.tsv"):
        with open(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                chrom = row["chrom"]
                start, end = int(row["start"]), int(row["end"])
                low, high = spans[chrom]
                spans[chrom] = [start if low is None else min(low, start),
                                end if high is None else max(high, end)]
    return {chrom: tuple(v) for chrom, v in spans.items()}


def count_matches(query, reference, slop):
    total = 0
    for chrom, positions in query.items():
        anchors = np.asarray(sorted(reference.get(chrom, [])))
        if anchors.size == 0:
            continue
        for position in positions:
            if np.min(np.abs(anchors - position)) <= slop:
                total += 1
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default=str(RESULTS / "events"))
    parser.add_argument("--query", default=str(RESULTS / "events_ontad"))
    parser.add_argument("--slops", nargs="*", type=int,
                        default=[500_000, 1_000_000, 2_000_000])
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--max-gap-bins", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    reference_calls = load_calls(args.reference, args.max_gap_bins)
    query_calls = load_calls(args.query, args.max_gap_bins)
    spans = callable_span(args.reference)
    rng = np.random.default_rng(args.seed)

    rows = []
    for slop in args.slops:
        reference = cluster(reference_calls, slop)
        query = cluster(query_calls, slop)
        n_reference = sum(len(v) for v in reference.values())
        n_query = sum(len(v) for v in query.values())
        observed = count_matches(query, reference, slop)
        null = np.empty(args.permutations)
        for i in range(args.permutations):
            shuffled = {}
            for chrom, positions in query.items():
                low, high = spans.get(chrom, (0, 0))
                if high <= low:
                    continue
                shuffled[chrom] = list(rng.integers(low, high, size=len(positions)))
            null[i] = count_matches(shuffled, reference, slop)
        mean = float(null.mean())
        rows.append({
            "slop_bp": slop, "reference_loci": n_reference, "query_loci": n_query,
            "matched": observed,
            "matched_fraction": round(observed / n_query, 4) if n_query else "",
            "null_mean": round(mean, 2),
            "fold": round(observed / mean, 3) if mean else "",
            "p_greater": round(float((np.sum(null >= observed) + 1)
                                     / (args.permutations + 1)), 5),
        })

    out_dir = Path(args.out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cross_caller.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")
    print(f"{'slop':>8}{'ref loci':>10}{'query':>8}{'matched':>9}{'null':>8}{'fold':>7}{'p':>8}")
    for row in rows:
        print(f"{row['slop_bp'] / 1e6:>7.1f}M{row['reference_loci']:>10}"
              f"{row['query_loci']:>8}{row['matched']:>9}{row['null_mean']:>8.1f}"
              f"{row['fold']:>7.2f}{row['p_greater']:>8.3f}")


if __name__ == "__main__":
    main()
