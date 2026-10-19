#!/usr/bin/env python3
"""Which individual merge/split loci survive a test on cells no run ever saw?

One caller at a time. Cross-caller agreement is not used as a criterion here: two TAD
callers carry different positional biases and cannot be asked to place a boundary in the
same 250 kb bin. What replaces it is out-of-sample evidence from the same caller -- a clustering run at one
resolution and chromosome uses a few dozen cells, the per-cell call tree holds several
hundred, and everything the runs did not touch is a held-out set.

**The null is empirical and that is not negotiable.** A 2x2 Fisher test on
(cells with a boundary here) x (stage) gives absurd answers on this data -- hundreds of loci
at FDR 0.10 with q down to 1e-9 -- because it assumes cells are exchangeable and they are
not. Boundary calls are correlated between cells of a stage, so the per-bin penetrance
difference is heavily overdispersed: on a typical chromosome, 2 to 5 per cent of *all* bins
already move by more than 0.3 between two consecutive stages with no event called there.
The background distribution of the statistic is therefore taken from the chromosome itself.

For a transition A -> B and a bin the medians report:

    delta = (fraction of held-out B cells with a boundary in that bin)
          - (fraction of held-out A cells with a boundary in that bin)

referenced by subtracting the mean delta of the boundaries the same transition calls
*stable*, which removes the drift that comes from the stages differing in call depth. The
p-value is the fraction of *background* bins on the same chromosome and transition whose
referenced delta is at least as extreme in the predicted direction -- gained boundaries
should move up, lost boundaries down. Background is restricted to bins that carry a
boundary in at least one of the two stages, since a bin that is empty in both cannot
produce a delta and would only dilute the comparison.

Two nulls are reported. ``p_empirical`` uses the bins of that chromosome alone, which caps
the smallest reachable p at one over the number of background bins -- about 1/229 at 500 kb,
too coarse for Benjamini-Hochberg over several hundred loci. ``p_pooled`` removes that cap:
each chromosome's background is standardised by its own median and robust scale and the
standardised backgrounds are pooled over the genome, giving 30,000 to 50,000 background
values per transition and a floor near 2e-5. The pooled null is the one to read; the
per-chromosome one is kept because it makes no assumption that the chromosomes share a
shape.

The answer is that no individual locus survives either null. That is worth stating
precisely, because the reason is not caller disagreement: this test never leaves OnTAD. The
strongest locus reaches a standardised score above 6 and still only reaches p = 2e-3,
because the background is heavy tailed -- under a Gaussian, z = 6.8 would read as 1e-11.
The class is real (see the aggregate at the foot of the table); its members are not
individually identifiable.

Usage:
    python merge_split_locus_validation.py --caller RobusTAD
    python merge_split_locus_validation.py --caller OnTAD --resolutions 500kb
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from cell_calls import CHROMS, SOURCES, STAGE_ORDER, read_cells

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "merge_split"

RESOLUTION_BP = {"250kb": 250_000, "500kb": 500_000, "1000kb": 1_000_000}
RESOLUTIONS = ["250kb", "500kb", "1000kb"]
TOLERANCES = range(6)
# a stage needs this many held-out cells before its penetrance is worth taking
MIN_HELD_OUT = 20
# a transition needs this many stable boundaries to act as its own reference
MIN_STABLE = 5
# and this many background bins before an empirical p-value means anything
MIN_BACKGROUND = 30
# the direction each reported class predicts for the held-out cells
EXPECTED = {"split": +1, "merge": -1}

FIELDS = ["resolution", "chrom", "transition", "event_type", "bin", "start", "end", "runs",
          "cells_before", "cells_after", "penetrance_before", "penetrance_after",
          "delta", "referenced_delta", "z_score", "background_bins", "p_empirical",
          "p_pooled", "q_empirical", "q_pooled"]


def benjamini_hochberg(pvalues):
    order = np.argsort(pvalues)
    n = len(pvalues)
    q = np.empty(n)
    running = 1.0
    for rank, index in enumerate(order[::-1]):
        running = min(running, pvalues[index] * n / (n - rank))
        q[index] = running
    return q


def held_out_penetrance(pool, chrom, resolution, used):
    """``{stage: (n cells, {bin: count})}`` over the cells no run of this grid touched."""
    out = {}
    for stage in STAGE_ORDER:
        cells = read_cells(pool, stage, chrom, resolution)
        held = [v for name, v in cells.items() if name not in used]
        if len(held) < MIN_HELD_OUT:
            continue
        counts = collections.Counter()
        for boundaries in held:
            for b in set(boundaries):
                counts[b] += 1
        out[stage] = (len(held), counts)
    return out


def chromosome(source, resolution, chrom):
    """Every merge/split bin of one chromosome, scored against the chromosome's own null."""
    used = set()
    calls = collections.defaultdict(lambda: collections.defaultdict(set))
    stable = collections.defaultdict(set)
    for tolerance in TOLERANCES:
        run = Path(source["runs"]) / resolution / chrom / f"tol_{tolerance}.json"
        events = Path(source["events"]) / resolution / chrom / \
            f"tol_{tolerance}_boundaries.tsv"
        if not (run.exists() and events.exists()):
            continue
        used |= {Path(p).parts[-4]
                 for p in json.load(open(run)).get("filtered_paths", [])}
        with open(events) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = (row["from_stage"], row["to_stage"])
                if row["event_type"] == "stable":
                    stable[key].add(int(row["bin"]))
                elif row["event_type"] in EXPECTED:
                    calls[key + (int(row["bin"]),)][row["event_type"]].add(tolerance)
    if not calls:
        return []

    penetrance = held_out_penetrance(source["pool"], chrom, resolution, used)
    rows = []
    standardised = collections.defaultdict(list)
    for (before, after, index), classes in calls.items():
        if before not in penetrance or after not in penetrance:
            continue
        n_before, counts_before = penetrance[before]
        n_after, counts_after = penetrance[after]

        def delta_of(b):
            return counts_after.get(b, 0) / n_after - counts_before.get(b, 0) / n_before

        reference_bins = stable.get((before, after), set())
        if len(reference_bins) < MIN_STABLE:
            continue
        reference = float(np.mean([delta_of(b) for b in reference_bins]))

        # background: every bin carrying a boundary in at least one of the two stages
        touched = set(counts_before) | set(counts_after)
        background = np.array([delta_of(b) - reference for b in sorted(touched)])
        if background.size < MIN_BACKGROUND:
            continue

        observed = delta_of(index) - reference
        centre = float(np.median(background))
        spread = float(np.median(np.abs(background - centre))) * 1.4826 \
            or float(background.std()) or 1.0
        standardised[f"{before}->{after}"].append((background - centre) / spread)
        for event_type, tolerances in classes.items():
            direction = EXPECTED[event_type]
            if direction > 0:
                at_least = int((background >= observed).sum())
            else:
                at_least = int((background <= observed).sum())
            rows.append({
                "resolution": resolution, "chrom": chrom,
                "transition": f"{before}->{after}", "event_type": event_type,
                "bin": index,
                "start": (index - 1) * RESOLUTION_BP[resolution],
                "end": index * RESOLUTION_BP[resolution],
                "runs": len(tolerances),
                "cells_before": n_before, "cells_after": n_after,
                "penetrance_before": round(counts_before.get(index, 0) / n_before, 4),
                "penetrance_after": round(counts_after.get(index, 0) / n_after, 4),
                "delta": round(delta_of(index), 4),
                "referenced_delta": round(observed, 4),
                "z_score": round((observed - centre) / spread, 4),
                "background_bins": int(background.size),
                "p_empirical": round((at_least + 1) / (background.size + 1), 5),
            })
    return rows, standardised


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caller", default="OnTAD", choices=sorted(SOURCES))
    parser.add_argument("--resolutions", nargs="+", default=RESOLUTIONS)
    parser.add_argument("--chroms", nargs="+", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or OUT_DIR / args.caller)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = SOURCES[args.caller]
    print(f"caller: {args.caller}")

    rows = []
    for resolution in args.resolutions:
        found, pooled = [], collections.defaultdict(list)
        for chrom in (args.chroms or CHROMS):
            chrom_rows, chrom_background = chromosome(source, resolution, chrom)
            found += chrom_rows
            for transition, arrays in chrom_background.items():
                pooled[transition] += arrays
        if not found:
            continue
        pooled = {t: np.concatenate(v) for t, v in pooled.items() if v}
        for row in found:
            background = pooled.get(row["transition"])
            z = row["z_score"]
            if background is None or background.size == 0:
                row["p_pooled"] = ""
                continue
            if EXPECTED[row["event_type"]] > 0:
                share = float((background >= z).mean())
            else:
                share = float((background <= z).mean())
            row["p_pooled"] = round(max(share, 1.0 / background.size), 8)
        for field, target in (("p_empirical", "q_empirical"),
                              ("p_pooled", "q_pooled")):
            usable = [r for r in found if r[field] != ""]
            if not usable:
                continue
            q = benjamini_hochberg(np.array([r[field] for r in usable]))
            for row, value in zip(usable, q):
                row[target] = round(float(value), 5)
        rows += found
        survivors = sum(1 for r in found if r.get("q_pooled", 1) < 0.10)
        floor = min((background.size for background in pooled.values()), default=0)
        print(f"{resolution:7s} {len(found):5d} merge/split loci tested, "
              f"{survivors:3d} at FDR < 0.10 (pooled null), "
              f"min p = {min(r['p_pooled'] for r in found if r['p_pooled'] != ''):.2e}, "
              f"null floor {1 / floor if floor else 0:.1e}")

    path = out_dir / "locus_validation.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
