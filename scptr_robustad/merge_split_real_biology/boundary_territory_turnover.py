#!/usr/bin/env python3
"""What kind of genomic territory does a transition trade its boundaries for?

Section 9 of the report scores each stage median's own boundaries and finds a minimum at
the 2-cell stage that climbs to 64-cell. That test needs a relocation null, because it
compares a median with the genome. This script asks the same question in a form that needs
no null at all.

Every transition both destroys and creates boundaries. Comparing the two **inside one
transition** holds the chromosome, the caller, the clustering run and the pair of medians
fixed, so the ratio

    log2( mean feature density at the boundaries the transition gained
        / mean feature density at the boundaries it lost )

is a paired contrast with nothing left to control for. Chromosomes are the replicates; the
six tolerances of a bin size are re-analyses of one clustering and are averaged within a
(bin size, chromosome) before the test, and the three bin sizes are tested separately for
the same reason.

The answer is a two-step handover at the start of development, and it reproduces on both
callers at 250 kb and 500 kb:

* **1-cell to 2-cell trades gene-dense territory for gene-poor territory.** The boundaries
  the zygote median loses are markedly denser in every active feature than the ones the
  2-cell median gains.
* **2-cell to 4-cell reverses it**, and the boundaries installed after zygotic genome
  activation are the dense ones again.
* 4-cell to 8-cell is flat, and 8-cell to 64-cell is where the two callers disagree, so
  nothing should be claimed from it.

This is the mechanism under the stage-maturation curve: the 2-cell minimum is not the
medians drifting, it is one transition discarding boundaries in active territory and the
next putting them back.

**It is a territory result and not a factor result.** CTCF, cohesin, CpG islands, H3K4me3,
H3K27ac, H3K9ac, H3K36me3, POLR2A, EP300, NANOG and POU5F1 all move together and all move
with plain gene density, which is what happens to any set of windows that is gene rich or
gene poor as a whole. ``gene_density`` is therefore reported alongside them and is the row
to read first: no track here departs from it, so the honest statement is about active,
gene-dense chromatin and not about any one protein. The control that makes this a statement
rather than an artefact is H3K9me3, the one repressive, gene-poor track in the panel, which
stays flat -- so the effect is specific to active territory rather than to annotation
coverage in general.

The same contrast restricted to split-created against merge-destroyed boundaries points the
same way at 1-cell to 2-cell and at 2-cell to 4-cell but rarely reaches significance: those
events number a few per chromosome. Nothing here is specific to merges and splits; they
follow the same rule as domains appearing and disappearing.

Usage:
    python boundary_territory_turnover.py
    python boundary_territory_turnover.py --caller OnTAD --classes merge_split
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

from cell_calls import CHROMS, SOURCES
from gene_sets import CHROMATIN_MARKS, TRANSITIONS, load_gene_table, load_mark
from annotate_events import GeneIndex
from stage_boundary_maturation import sign_test, wilcoxon_signed_rank

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "territory"
RESOLUTIONS = ["250kb", "500kb", "1000kb"]

# which boundary rows the numerator and the denominator are taken from
CLASS_SETS = {
    # every boundary a transition creates against every one it destroys
    "all": (None, "gained", "lost"),
    # domains appearing against domains disappearing
    "gain_loss": ({"domain_gain", "domain_loss"}, "gained", "lost"),
    # the boundary a split installs against the one a merge removes
    "merge_split": ({"split", "merge"}, "gained", "lost"),
}

FIELDS = ["caller", "classes", "resolution", "transition", "track", "chromosomes",
          "positive", "mean_log2_ratio", "gained_per_bin", "lost_per_bin",
          "sign_p", "wilcoxon_p"]


def counters(marks, index):
    """``{track: callable(chrom, start, end) -> count}``, gene density included."""
    def peaks(intervals):
        starts = {c: [a for a, _ in iv] for c, iv in intervals.items()}
        import bisect

        def count(chrom, start, end):
            positions = starts.get(chrom)
            if not positions:
                return 0
            return bisect.bisect_left(positions, end) - bisect.bisect_left(positions, start)
        return count

    out = {name: peaks(marks[name]) for name in CHROMATIN_MARKS if marks.get(name)}
    out["gene_density"] = index.count_in_window
    return out


def gather(events_dir, resolution, event_types, tracks, chroms=None):
    """``{(chrom, tolerance, transition, change, track): [densities]}`` for one bin size."""
    root = Path(events_dir) / resolution
    values = collections.defaultdict(lambda: collections.defaultdict(list))
    if not root.exists():
        return values
    for path in sorted(root.glob("*/tol_*_boundaries.tsv")):
        chrom = path.parent.name
        if chroms and chrom not in chroms:
            continue
        tolerance = int(path.name.split("_")[1])
        with open(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["change"] == "stable":
                    continue
                if event_types and row["event_type"] not in event_types:
                    continue
                key = (chrom, tolerance, row["transition"], row["change"])
                start, end = int(row["start"]), int(row["end"])
                for name, count in tracks.items():
                    values[key][name].append(count(chrom, start, end))
    return values


def contrast(values, transition, track, numerator, denominator, min_side):
    """One log2 ratio per chromosome, tolerances averaged first."""
    per_chrom = collections.defaultdict(list)
    totals = {numerator: [], denominator: []}
    keys = {(c, t) for (c, t, tr, ch) in values if tr == transition}
    for chrom, tolerance in sorted(keys):
        up = values.get((chrom, tolerance, transition, numerator), {}).get(track, [])
        down = values.get((chrom, tolerance, transition, denominator), {}).get(track, [])
        if len(up) < min_side or len(down) < min_side:
            continue
        mean_up, mean_down = float(np.mean(up)), float(np.mean(down))
        if mean_up <= 0 or mean_down <= 0:
            continue
        per_chrom[chrom].append(np.log2(mean_up / mean_down))
        totals[numerator].append(mean_up)
        totals[denominator].append(mean_down)
    ratios = [float(np.mean(v)) for v in per_chrom.values()]
    return ratios, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caller", action="append", dest="callers",
                        choices=sorted(SOURCES), help="repeatable; default both")
    parser.add_argument("--classes", action="append", dest="class_sets",
                        choices=sorted(CLASS_SETS), help="repeatable; default all three")
    parser.add_argument("--resolution", action="append", dest="resolutions",
                        choices=RESOLUTIONS)
    parser.add_argument("--chroms", nargs="+", default=None)
    parser.add_argument("--min-side", type=int, default=5,
                        help="boundaries needed on each side of a run before it is used")
    parser.add_argument("--min-chromosomes", type=int, default=8)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    callers = args.callers or sorted(SOURCES)
    class_sets = args.class_sets or ["all", "gain_loss", "merge_split"]
    resolutions = args.resolutions or RESOLUTIONS
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = load_gene_table()
    index = GeneIndex(genes)
    marks = {name: load_mark(name) for name in CHROMATIN_MARKS}
    tracks = counters(marks, index)
    print(f"tracks: {', '.join(tracks)}")

    rows = []
    for caller in callers:
        events_dir = SOURCES[caller]["events"]
        for name in class_sets:
            event_types, numerator, denominator = CLASS_SETS[name]
            for resolution in resolutions:
                values = gather(events_dir, resolution, event_types, tracks,
                                args.chroms or CHROMS)
                if not values:
                    continue
                for transition in TRANSITIONS:
                    for track in tracks:
                        ratios, totals = contrast(values, transition, track, numerator,
                                                  denominator, args.min_side)
                        if len(ratios) < args.min_chromosomes:
                            continue
                        rows.append({
                            "caller": caller, "classes": name, "resolution": resolution,
                            "transition": transition, "track": track,
                            "chromosomes": len(ratios),
                            "positive": sum(1 for v in ratios if v > 0),
                            "mean_log2_ratio": round(float(np.mean(ratios)), 4),
                            "gained_per_bin": round(float(np.mean(totals[numerator])), 3),
                            "lost_per_bin": round(float(np.mean(totals[denominator])), 3),
                            "sign_p": round(sign_test(ratios), 6),
                            "wilcoxon_p": round(wilcoxon_signed_rank(ratios), 6)})

    path = out_dir / "turnover_asymmetry.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # the readable summary: gene density and the two controls, class "all"
    print("\nlog2(gained / lost), tolerances averaged within a bin size, "
          "chromosomes as replicates")
    for track in ("gene_density", "ctcf", "h3k9me3"):
        print(f"\n-- {track}")
        header = "  ".join(f"{t.replace('-cell', ''):>17}" for t in TRANSITIONS)
        print(f"   {'caller':9} {'res':7} {header}")
        for caller in callers:
            for resolution in resolutions:
                cells = []
                for transition in TRANSITIONS:
                    match = [r for r in rows if r["caller"] == caller
                             and r["classes"] == "all" and r["resolution"] == resolution
                             and r["transition"] == transition and r["track"] == track]
                    if not match:
                        cells.append(f"{'--':>17}")
                        continue
                    r = match[0]
                    star = "*" if r["sign_p"] < 0.05 else " "
                    cells.append(f"{r['mean_log2_ratio']:+.3f} "
                                 f"{r['positive']:2d}/{r['chromosomes']:<2d}{star}".rjust(17))
                print(f"   {caller:9} {resolution:7} " + "  ".join(cells))
    print(f"\nwrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
