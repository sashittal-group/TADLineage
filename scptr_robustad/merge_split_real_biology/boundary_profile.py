#!/usr/bin/env python3
"""How far from a called boundary does CTCF sit?

Panel d of the summary figure asked whether a boundary bin carries more CTCF than a random
bin, and found that it carries less. That test assumes the boundary is where the caller put
it. Single-cell Hi-C is sparse, so it very likely is not: the same physical boundary can be
recovered a bin or two away in the next median, and a one-bin window then misses it.

This script separates the two explanations. For every boundary it walks out bin by bin and
measures CTCF peak density and gene density at each offset, giving a profile centred on the
call:

    a hump at offset 0, a few bins wide  -> real boundaries, blurred by sparsity, and a
                                            positional tolerance will recover them
    flat                                 -> the windows are uninformative at every offset
    a dip at 0 that stays down           -> the calls sit in quiet parts of the genome for
                                            a reason other than jitter

The denominator is exact rather than permuted. A single-bin window relocated uniformly over
the callable region has expected density equal to the mean density over that region, so the
callable-region mean is the null, with no sampling error. Offsets that fall outside the
callable region are dropped rather than counted as zero, which would manufacture a dip at
the edges.

Usage:
    python boundary_profile.py
    python boundary_profile.py --span 10 --resolution 500kb
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

from annotate_events import GeneIndex, read_callable, read_events
from coordinates import resolution_to_bp
from enrichment import build_feature_sets, callable_bin_pool, read_boundaries
from gene_sets import load_ctcf_peaks, load_gene_table

HERE = Path(__file__).resolve().parent
EVENTS_DIR = HERE / "results" / "events"
OUT_DIR = HERE / "results" / "profile"

BOUNDARY_SETS = ["boundary_gained", "boundary_lost", "boundary_stable",
                 "split_boundary", "merge_boundary"]

FIELDS = ["resolution", "tolerance", "feature_set", "offset_bins", "n_windows",
          "ctcf_per_bin", "ctcf_null", "ctcf_fold", "genes_per_bin", "genes_null",
          "gene_fold"]


def per_bin_counts(positions_by_chrom, resolution_bp, n_bins):
    """``{chrom: array}`` of how many features fall in each bin, 1-based."""
    tracks = {}
    for chrom, size in n_bins.items():
        counts = np.zeros(size + 2, dtype=np.int64)
        for position in positions_by_chrom.get(chrom, []):
            counts[min(size + 1, position // resolution_bp + 1)] += 1
        tracks[chrom] = counts
    return tracks


def peak_midpoints(peaks):
    return {chrom: [(start + end) // 2 for start, end in entries]
            for chrom, entries in peaks.items()}


def profile_configuration(resolution, tolerance, peaks, index, span=10,
                          max_gap_bins=1, events_dir=None):
    resolution_bp = resolution_to_bp(resolution)
    events = read_events(resolution, tolerance, events_dir)
    boundaries = read_boundaries(resolution, tolerance, events_dir)
    # flank 0: the profile is what tells us which flank to use, so it must start unwidened
    features = build_feature_sets(events, boundaries, resolution_bp, max_gap_bins,
                                  flank_bins=0)
    pool = callable_bin_pool(read_callable(resolution, tolerance, events_dir),
                             resolution_bp)
    n_bins = {chrom: int(bins.max()) for chrom, bins in pool.items() if len(bins)}
    if not n_bins:
        return []

    ctcf = per_bin_counts(peak_midpoints(peaks), resolution_bp, n_bins)
    genes = per_bin_counts(index.positions, resolution_bp, n_bins)
    inside = {chrom: set(bins.tolist()) for chrom, bins in pool.items()}

    # the null a uniformly relocated single-bin window would see
    callable_total = sum(len(bins) for bins in pool.values())
    ctcf_null = sum(float(ctcf[c][pool[c]].sum()) for c in pool if c in ctcf)
    gene_null = sum(float(genes[c][pool[c]].sum()) for c in pool if c in genes)
    ctcf_null /= max(1, callable_total)
    gene_null /= max(1, callable_total)

    windows = collections.defaultdict(list)
    for (name, _transition), entries in features.items():
        if name in BOUNDARY_SETS:
            windows[name].extend(entries)

    rows = []
    for name, entries in sorted(windows.items()):
        for offset in range(-span, span + 1):
            ctcf_hits, gene_hits, seen = 0.0, 0.0, 0
            for chrom, start, width in entries:
                if chrom not in inside:
                    continue
                # centre of the window, so a two-bin split boundary is not off by one
                centre = start + (width - 1) // 2 + offset
                if centre not in inside[chrom]:
                    continue
                ctcf_hits += ctcf[chrom][centre]
                gene_hits += genes[chrom][centre]
                seen += 1
            if seen == 0:
                continue
            rows.append({
                "resolution": resolution, "tolerance": tolerance, "feature_set": name,
                "offset_bins": offset, "n_windows": seen,
                "ctcf_per_bin": round(ctcf_hits / seen, 2),
                "ctcf_null": round(ctcf_null, 2),
                "ctcf_fold": round((ctcf_hits / seen) / ctcf_null, 3) if ctcf_null else "",
                "genes_per_bin": round(gene_hits / seen, 3),
                "genes_null": round(gene_null, 3),
                "gene_fold": round((gene_hits / seen) / gene_null, 3) if gene_null else "",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--span", type=int, default=10,
                        help="how many bins to walk out on each side")
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--tolerance", type=int, default=None)
    parser.add_argument("--max-gap-bins", type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    peaks = load_ctcf_peaks()
    if peaks is None:
        print("No CTCF peaks in reference/. Run fetch_reference_data.py --with-ctcf.")
        return
    index = GeneIndex(load_gene_table())

    pairs = sorted({(path.parent.parent.name, int(path.name.split("_")[1]))
                    for path in EVENTS_DIR.glob("*/*/tol_*_events.tsv")})
    if args.resolution:
        pairs = [p for p in pairs if p[0] == args.resolution]
    if args.tolerance is not None:
        pairs = [p for p in pairs if p[1] == args.tolerance]

    rows = []
    for resolution, tolerance in pairs:
        rows.extend(profile_configuration(resolution, tolerance, peaks, index,
                                          span=args.span,
                                          max_gap_bins=args.max_gap_bins))

    out_dir = Path(args.out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "boundary_profile.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows over {len(pairs)} configurations)")

    print()
    print("CTCF density relative to the callable-region mean, by offset from the call")
    offsets = sorted({row["offset_bins"] for row in rows})
    header = "".join(f"{o:>7}" for o in offsets)
    print(f"{'feature set':20}{header}")
    for name in BOUNDARY_SETS:
        line = ""
        for offset in offsets:
            folds = [r["ctcf_fold"] for r in rows
                     if r["feature_set"] == name and r["offset_bins"] == offset
                     and r["ctcf_fold"] != ""]
            line += f"{np.mean(folds):>7.2f}" if folds else f"{'-':>7}"
        print(f"{name:20}{line}")


if __name__ == "__main__":
    main()
