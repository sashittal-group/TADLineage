#!/usr/bin/env python3
"""Does chrX lose domain structure faster than the autosomes, and where does it keep it?

This is the one place in the dataset with a published, directional prediction to test.
Collombet et al. 2020 (Nature 580:142), working on these same embryos, report that TADs on
the paternal X are lost as its genes are silenced during imprinted X inactivation, and that
they persist over the regions that escape inactivation. Imprinted X inactivation runs
through preimplantation development and is largely complete by the blastocyst, so if the
median TAD structures carry that signal at all, chrX should lose boundaries faster than the
autosomes over the later transitions, and what it keeps should sit near the escapees.

Two tests.

Rate test
    For each transition and each configuration, the number of boundaries lost per callable
    bin is computed separately for every chromosome. chrX is then placed against the
    distribution of the nineteen autosomes, as a z-score and as a rank. Comparing chrX with
    the autosomes of the same run rather than with a null model keeps the bin size, the
    tolerance and the cell sample fixed, so only the chromosome differs. The whole
    parameter grid is then summarised by how often chrX comes out at the top, which is what
    a real chromosome-wide effect would have to do.

    The TAD calls here are not allele resolved, so a chromosome-wide loss on chrX is diluted
    by the active X of every cell and by the male embryos in the sample. The test is
    therefore conservative: it can support the prediction but a null result does not refute
    it.

Escape test
    On chrX, the distance from each boundary to the nearest X-inactivation escapee is
    compared between the boundaries that survive a transition and those that are lost. The
    prediction is that surviving boundaries sit closer to escapees.

Usage:
    python chrx_analysis.py
    python chrx_analysis.py --transition 8-cell->64-cell
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from annotate_events import read_callable
from coordinates import resolution_to_bp
from enrichment import read_boundaries
from gene_sets import TRANSITIONS, load_curated_sets, load_gene_table

HERE = Path(__file__).resolve().parent
EVENTS_DIR = HERE / "results" / "events"
OUT_DIR = HERE / "results" / "chrx"

RATE_FIELDS = [
    "resolution", "tolerance", "transition", "change", "chrx_rate", "autosome_mean",
    "autosome_sd", "ratio", "z", "rank_of_20", "n_chromosomes",
]
DISTANCE_FIELDS = [
    "resolution", "tolerance", "transition", "change", "n_boundaries",
    "median_distance_kb", "mean_distance_kb",
]


def configurations(events_dir=None):
    root = Path(events_dir or EVENTS_DIR)
    return sorted({(path.parent.parent.name, int(path.name.split("_")[1]))
                   for path in root.glob("*/*/tol_*_boundaries.tsv")})


def rates_by_chromosome(boundaries, callable_spans, resolution_bp, transition, change):
    """Boundaries of one kind per callable bin, for each chromosome."""
    counts = {}
    for row in boundaries:
        if row["transition"] != transition or row["change"] != change:
            continue
        counts[row["chrom"]] = counts.get(row["chrom"], 0) + 1

    rates = {}
    for chrom, spans in callable_spans.items():
        bins = sum((int(end) - int(start)) // resolution_bp for start, end in spans)
        if bins > 0:
            rates[chrom] = counts.get(chrom, 0) / bins
    return rates


def compare_chrx(rates):
    """chrX against the autosomes of the same run: ratio, z-score and rank."""
    if "chrX" not in rates:
        return None
    autosomes = np.array([value for chrom, value in rates.items() if chrom != "chrX"])
    if len(autosomes) < 5:
        return None
    chrx = rates["chrX"]
    mean = float(autosomes.mean())
    sd = float(autosomes.std(ddof=1))
    ordered = sorted(rates.values(), reverse=True)
    return {
        "chrx_rate": round(chrx, 5),
        "autosome_mean": round(mean, 5),
        "autosome_sd": round(sd, 5),
        "ratio": round(chrx / mean, 3) if mean > 0 else "",
        "z": round((chrx - mean) / sd, 2) if sd > 0 else "",
        "rank_of_20": ordered.index(chrx) + 1,
        "n_chromosomes": len(rates),
    }


def escape_distances(boundaries, escapee_positions, transition, change):
    """Distance in bp from each chrX boundary of one kind to the nearest escapee."""
    if not escapee_positions:
        return []
    positions = np.array(sorted(escapee_positions))
    distances = []
    for row in boundaries:
        if row["chrom"] != "chrX" or row["transition"] != transition:
            continue
        if row["change"] != change:
            continue
        centre = (int(row["start"]) + int(row["end"])) / 2
        distances.append(float(np.min(np.abs(positions - centre))))
    return distances


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transition", default=None,
                        help="restrict to one transition; default is all four")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    genes = load_gene_table()
    escapees = load_curated_sets(genes=genes).get("xci_escape_mouse", set())
    escapee_positions = [genes[symbol]["tss"] for symbol in escapees
                         if genes[symbol]["chrom"] == "chrX"]

    transitions = [args.transition] if args.transition else TRANSITIONS
    rate_rows, distance_rows = [], []

    for resolution, tolerance in configurations():
        resolution_bp = resolution_to_bp(resolution)
        boundaries = read_boundaries(resolution, tolerance)
        spans = read_callable(resolution, tolerance)
        for transition in transitions:
            for change in ("lost", "gained", "stable"):
                rates = rates_by_chromosome(boundaries, spans, resolution_bp,
                                            transition, change)
                summary = compare_chrx(rates)
                if summary:
                    rate_rows.append({
                        "resolution": resolution, "tolerance": tolerance,
                        "transition": transition, "change": change, **summary,
                    })
                distances = escape_distances(boundaries, escapee_positions,
                                             transition, change)
                if distances:
                    distance_rows.append({
                        "resolution": resolution, "tolerance": tolerance,
                        "transition": transition, "change": change,
                        "n_boundaries": len(distances),
                        "median_distance_kb": round(float(np.median(distances)) / 1000, 1),
                        "mean_distance_kb": round(float(np.mean(distances)) / 1000, 1),
                    })

    out_dir = Path(args.out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, fields, rows in (
        (out_dir / "chrx_boundary_rates.tsv", RATE_FIELDS, rate_rows),
        (out_dir / "chrx_escape_distances.tsv", DISTANCE_FIELDS, distance_rows),
    ):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")

    print()
    print("chrX boundary loss against the autosomes of the same run")
    print(f"{'transition':18}{'configs':>8}{'mean ratio':>12}{'mean z':>8}"
          f"{'top rank':>10}{'top 3':>7}")
    for transition in transitions:
        subset = [r for r in rate_rows
                  if r["transition"] == transition and r["change"] == "lost"]
        if not subset:
            continue
        ratios = [r["ratio"] for r in subset if r["ratio"] != ""]
        zs = [r["z"] for r in subset if r["z"] != ""]
        ranks = [r["rank_of_20"] for r in subset]
        print(f"{transition:18}{len(subset):>8}{np.mean(ratios):>12.3f}"
              f"{np.mean(zs):>8.2f}{sum(1 for r in ranks if r == 1):>10}"
              f"{sum(1 for r in ranks if r <= 3):>7}")

    print()
    print("distance from a chrX boundary to the nearest escapee (median kb)")
    print(f"{'transition':18}{'lost':>10}{'stable':>10}{'gained':>10}")
    for transition in transitions:
        values = {}
        for change in ("lost", "stable", "gained"):
            subset = [r["median_distance_kb"] for r in distance_rows
                      if r["transition"] == transition and r["change"] == change]
            values[change] = np.mean(subset) if subset else float("nan")
        print(f"{transition:18}{values['lost']:>10.0f}{values['stable']:>10.0f}"
              f"{values['gained']:>10.0f}")


if __name__ == "__main__":
    main()
