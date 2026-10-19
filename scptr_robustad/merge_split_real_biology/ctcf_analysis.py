#!/usr/bin/env python3
"""Do the boundaries that appear, disappear or survive sit on CTCF sites?

CTCF occupancy is the standard check on whether something called a TAD boundary is one:
mammalian TAD boundaries are marked by CTCF, usually in convergent pairs. Running that check
on these events says whether the boundary changes between stage medians are structural
features or positions the clustering happened to land on.

The same relocation null as the other tests is used, so a window set is compared with
itself placed elsewhere in the callable region, and the statistic is the number of CTCF
peak midpoints per window.

    A bin here is 250 kb to 1 Mb and the ENCODE peak set has over three hundred thousand
    peaks, roughly one every nine kilobases, so essentially every bin contains many peaks.
    This can therefore only detect a difference in how *densely* CTCF is bound, never the
    presence or absence of a site, and it cannot say anything about motif orientation. A
    flat result is the expected outcome of the resolution, not evidence against the events.

The peaks are ENCODE ENCSR179IQN, CTCF ChIP-seq in C57BL/6J mouse embryonic stem cells on
mm10; that is the closest widely used reference to these stages, but it is a cell line, not
an embryo, so it describes where CTCF tends to bind rather than where it was bound in these
cells.

Usage:
    python ctcf_analysis.py
    python ctcf_analysis.py --resolution 500kb --tolerance 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from annotate_events import read_callable, read_events
from coordinates import resolution_to_bp
from enrichment import (
    clip_windows_to_pool,
    benjamini_hochberg,
    build_feature_sets,
    callable_bin_pool,
    read_boundaries,
)
from expression_analysis import build_draws, window_sums
from gene_sets import load_ctcf_peaks

HERE = Path(__file__).resolve().parent
EVENTS_DIR = HERE / "results" / "events"
OUT_DIR = HERE / "results" / "ctcf"

FIELDS = [
    "resolution", "tolerance", "feature_set", "transition", "n_windows",
    "peaks_per_bin", "null_peaks_per_bin", "fold", "z", "p_greater", "p_less",
    "q_two_sided",
]


def peak_tracks(peaks, resolution_bp, n_bins):
    """Prefix sums of CTCF peak midpoints per bin, and of the bins themselves."""
    tracks = {}
    for chrom, size in n_bins.items():
        counts = np.zeros(size + 2)
        for start, end in peaks.get(chrom, []):
            position = min(size + 1, ((start + end) // 2) // resolution_bp + 1)
            counts[position] += 1
        tracks[chrom] = np.cumsum(counts)
    return tracks


def analyse(resolution, tolerance, peaks, permutations=2000, max_gap_bins=1, seed=0,
            min_windows=5, events_dir=None, flank_bins=0):
    resolution_bp = resolution_to_bp(resolution)
    events = read_events(resolution, tolerance, events_dir)
    boundaries = read_boundaries(resolution, tolerance, events_dir)
    features = build_feature_sets(events, boundaries, resolution_bp, max_gap_bins,
                                  flank_bins)
    pool = callable_bin_pool(read_callable(resolution, tolerance, events_dir),
                             resolution_bp)
    # a flanked window may overhang the chromosome end; the null only ever places
    # windows that fit, so clip the observed ones to match
    features = clip_windows_to_pool(features, pool)
    n_bins = {chrom: int(bins.max()) for chrom, bins in pool.items() if len(bins)}
    tracks = peak_tracks(peaks, resolution_bp, n_bins)

    rng = np.random.default_rng(seed)
    rows = []
    for (feature_name, transition), windows in sorted(features.items()):
        if len(windows) < min_windows:
            continue
        draws = build_draws(windows, pool, permutations, rng)
        if not draws:
            continue

        observed = observed_bins = 0.0
        null_total = null_bins = None
        for chrom, (starts, widths, choices) in draws.items():
            prefix = tracks[chrom]
            observed += float(window_sums(prefix, starts, widths).sum())
            observed_bins += float(widths.sum())
            total = window_sums(prefix, choices, widths).sum(axis=1)
            null_total = total if null_total is None else null_total + total
            null_bins = widths.sum() if null_bins is None else null_bins + widths.sum()

        if observed_bins == 0 or null_total is None:
            continue
        observed_rate = observed / observed_bins
        null_rates = null_total / null_bins
        null_mean = float(null_rates.mean())
        null_sd = float(null_rates.std(ddof=1)) if len(null_rates) > 1 else 0.0
        rows.append({
            "resolution": resolution, "tolerance": tolerance,
            "feature_set": feature_name, "transition": transition,
            "n_windows": len(windows),
            "peaks_per_bin": round(observed_rate, 2),
            "null_peaks_per_bin": round(null_mean, 2),
            "fold": round(observed_rate / null_mean, 3) if null_mean else "",
            "z": round((observed_rate - null_mean) / null_sd, 2) if null_sd else "",
            "p_greater": float((np.sum(null_rates >= observed_rate) + 1)
                               / (permutations + 1)),
            "p_less": float((np.sum(null_rates <= observed_rate) + 1)
                            / (permutations + 1)),
        })

    two_sided = [min(1.0, 2 * min(r["p_greater"], r["p_less"])) for r in rows]
    for row, q in zip(rows, benjamini_hochberg(two_sided)):
        row["q_two_sided"] = float(q)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default=None,
                        help="default: every resolution present")
    parser.add_argument("--tolerance", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--max-gap-bins", type=int, default=1)
    parser.add_argument("--flank-bins", type=int, default=0,
                        help="positional tolerance in bins: widen every boundary window by this many bins on each side, because sparse single-cell calls place the same boundary a bin or two apart between medians")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--events-dir", default=None,
                        help="event tables to read; use results/events_unimputed to test the medians recomputed from the unimputed calls")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    peaks = load_ctcf_peaks()
    if peaks is None:
        print("No CTCF peaks in reference/. Run "
              "fetch_reference_data.py --with-ctcf first.")
        return

    events_dir = Path(args.events_dir) if args.events_dir else EVENTS_DIR
    pairs = sorted({(path.parent.parent.name, int(path.name.split("_")[1]))
                    for path in events_dir.glob("*/*/tol_*_events.tsv")})
    if args.resolution:
        pairs = [p for p in pairs if p[0] == args.resolution]
    if args.tolerance is not None:
        pairs = [p for p in pairs if p[1] == args.tolerance]

    rows = []
    for resolution, tolerance in pairs:
        rows.extend(analyse(resolution, tolerance, peaks,
                            permutations=args.permutations,
                            max_gap_bins=args.max_gap_bins, seed=args.seed,
                            flank_bins=args.flank_bins, events_dir=events_dir))

    out_dir = Path(args.out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ctcf_at_boundaries.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} tests over {len(pairs)} configurations)")

    print()
    print("CTCF peak density, pooled over transitions and configurations")
    print(f"{'feature set':24}{'tests':>7}{'peaks/bin':>11}{'null':>9}{'fold':>8}"
          f"{'q<=0.1':>8}")
    names = sorted({row["feature_set"] for row in rows})
    for name in names:
        subset = [r for r in rows if r["feature_set"] == name]
        folds = [r["fold"] for r in subset if r["fold"] != ""]
        print(f"{name:24}{len(subset):>7}"
              f"{np.mean([r['peaks_per_bin'] for r in subset]):>11.1f}"
              f"{np.mean([r['null_peaks_per_bin'] for r in subset]):>9.1f}"
              f"{np.mean(folds):>8.3f}"
              f"{sum(1 for r in subset if r['q_two_sided'] <= 0.1):>8}")


if __name__ == "__main__":
    main()
