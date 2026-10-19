#!/usr/bin/env python3
"""Splitting the held-out cells in two, so a merge/split call can be ranked and judged.

``merge_split_locus_validation.py`` scores every merge/split bin against the cells no
clustering run ever touched. That single number cannot be used both to choose loci and to
judge them -- it would be the same statistic on both sides. This module cuts the held-out
cells of each stage into two halves and computes the statistic twice, once per half:

* ``z_a`` is the evidence used to **rank** a locus at the structural end of the sweep;
* ``z_b`` is the evidence used to **judge** the selected set.

The two halves are disjoint sets of cells, so ``z_b`` is out of sample for any ranking
built on ``z_a``, and it is out of sample for a ranking built on CTCF as well, because
CTCF is a ChIP track and this is Hi-C.

The statistic itself is unchanged from ``merge_split_locus_validation.py`` and is repeated
here rather than imported because it now has to be computed per half:

    delta   = (fraction of the half's later-stage cells with a boundary in the bin)
            - (fraction of its earlier-stage cells)
    referenced = delta - the mean delta over the boundaries the transition calls stable,
                 which removes the drift from the two stages differing in call depth
    z       = (referenced - median of the chromosome's background)
              / (1.4826 * median absolute deviation of that background)

Background is every bin carrying a boundary in at least one of the two stages, in that
half. Standardising by the chromosome's own background is not decoration: boundary calls
are correlated between cells of a stage, so the per-bin delta is heavily overdispersed and
a parametric test on it reports hundreds of false loci.

``z`` is signed by the class -- a split should gain penetrance in the later stage, a merge
should lose it -- so that a positive value means "as predicted" for both.

The halves are assigned by ``zlib.crc32`` of the cell name rather than Python's ``hash``,
which is salted per process, so a rerun reproduces the same split.

Usage:
    python splithalf_penetrance.py --caller OnTAD
    python splithalf_penetrance.py --caller OnTAD --resolutions 500kb --chroms chr1 chr2
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import zlib
from pathlib import Path

import numpy as np

from cell_calls import CHROMS, SOURCES, STAGE_ORDER, read_cells

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "weighted"

RESOLUTION_BP = {"250kb": 250_000, "500kb": 500_000, "1000kb": 1_000_000}
RESOLUTIONS = ["250kb", "500kb", "1000kb"]
TOLERANCES = range(6)
# a half needs this many cells at both stages before its penetrance is worth taking. Half
# of the 20 that merge_split_locus_validation.py requires of the whole held-out set.
MIN_PER_HALF = 10
MIN_STABLE = 5
MIN_BACKGROUND = 30
EXPECTED = {"split": +1, "merge": -1}

FIELDS = ["resolution", "chrom", "transition", "event_type", "bin", "start", "end",
          "runs", "cells_a", "cells_b", "z_a", "z_b", "background_a", "background_b"]


def half_of(name):
    """Which half a cell belongs to. crc32, never hash(), so reruns agree."""
    return zlib.crc32(name.encode()) & 1


def half_penetrance(pool, chrom, resolution, used):
    """``{stage: [(n cells, {bin: count}) for each half]}`` over the untouched cells."""
    out = {}
    for stage in STAGE_ORDER:
        cells = read_cells(pool, stage, chrom, resolution)
        halves = [[], []]
        for name, boundaries in cells.items():
            if name in used:
                continue
            halves[half_of(name)].append(boundaries)
        if min(len(h) for h in halves) < MIN_PER_HALF:
            continue
        summary = []
        for members in halves:
            counts = collections.Counter()
            for boundaries in members:
                for b in set(boundaries):
                    counts[b] += 1
            summary.append((len(members), counts))
        out[stage] = summary
    return out


def standardise(deltas, values):
    """Centre and scale by the background's own median and robust spread."""
    centre = float(np.median(deltas))
    spread = float(np.median(np.abs(deltas - centre))) * 1.4826
    if not spread:
        spread = float(deltas.std()) or 1.0
    return (np.asarray(values) - centre) / spread, (deltas - centre) / spread


def chromosome(source, resolution, chrom):
    """Split-half rows for one chromosome, plus the half-B background of each transition."""
    used = set()
    calls = collections.defaultdict(lambda: collections.defaultdict(set))
    stable = collections.defaultdict(set)
    for tolerance in TOLERANCES:
        run = Path(source["runs"]) / resolution / chrom / f"tol_{tolerance}.json"
        events = (Path(source["events"]) / resolution / chrom
                  / f"tol_{tolerance}_boundaries.tsv")
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
        return [], {}

    penetrance = half_penetrance(source["pool"], chrom, resolution, used)
    rows, background = [], {}
    grouped = collections.defaultdict(list)
    for (before, after, index), classes in calls.items():
        grouped[(before, after)].append((index, classes))

    for (before, after), members in grouped.items():
        if before not in penetrance or after not in penetrance:
            continue
        transition = f"{before}->{after}"
        reference_bins = stable.get((before, after), set())
        if len(reference_bins) < MIN_STABLE:
            continue

        per_half = []
        for h in (0, 1):
            n_before, counts_before = penetrance[before][h]
            n_after, counts_after = penetrance[after][h]

            def delta_of(b, cb=counts_before, ca=counts_after, nb=n_before, na=n_after):
                return ca.get(b, 0) / na - cb.get(b, 0) / nb

            reference = float(np.mean([delta_of(b) for b in reference_bins]))
            touched = sorted(set(counts_before) | set(counts_after))
            deltas = np.array([delta_of(b) - reference for b in touched])
            if deltas.size < MIN_BACKGROUND:
                per_half = []
                break
            observed = np.array([delta_of(index) - reference for index, _ in members])
            scaled, scaled_background = standardise(deltas, observed)
            per_half.append({"n_before": n_before, "n_after": n_after,
                             "scaled": scaled, "background": scaled_background})
        if len(per_half) != 2:
            continue

        background[transition] = per_half[1]["background"]
        for position, (index, classes) in enumerate(members):
            for event_type, tolerances in classes.items():
                direction = EXPECTED[event_type]
                rows.append({
                    "resolution": resolution, "chrom": chrom, "transition": transition,
                    "event_type": event_type, "bin": index,
                    "start": (index - 1) * RESOLUTION_BP[resolution],
                    "end": index * RESOLUTION_BP[resolution],
                    "runs": len(tolerances),
                    "cells_a": f"{per_half[0]['n_before']}/{per_half[0]['n_after']}",
                    "cells_b": f"{per_half[1]['n_before']}/{per_half[1]['n_after']}",
                    "z_a": round(float(direction * per_half[0]["scaled"][position]), 4),
                    "z_b": round(float(direction * per_half[1]["scaled"][position]), 4),
                    "background_a": int(per_half[0]["background"].size),
                    "background_b": int(per_half[1]["background"].size),
                })
    return rows, background


def build(caller, resolutions=None, chroms=None, quiet=False):
    """``(rows, {(resolution, chrom, transition): half-B background})`` for one caller."""
    source = SOURCES[caller]
    rows, background = [], {}
    for resolution in (resolutions or RESOLUTIONS):
        for chrom in (chroms or CHROMS):
            found, backgrounds = chromosome(source, resolution, chrom)
            rows.extend(found)
            for transition, values in backgrounds.items():
                background[(resolution, chrom, transition)] = values
            if found and not quiet:
                print(f"  {resolution:7} {chrom:6} {len(found):4} loci")
    return rows, background


def cache_paths(caller, out_dir=None):
    base = Path(out_dir or OUT_DIR) / caller
    return base / "splithalf_loci.tsv", base / "splithalf_background.npz"


def load_cached(caller, out_dir=None):
    """The tables written by a previous run, or ``None`` if they are not there."""
    loci_path, background_path = cache_paths(caller, out_dir)
    if not (loci_path.exists() and background_path.exists()):
        return None
    with open(loci_path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["bin"] = int(row["bin"])
        row["start"], row["end"] = int(row["start"]), int(row["end"])
        row["z_a"], row["z_b"] = float(row["z_a"]), float(row["z_b"])
    archive = np.load(background_path, allow_pickle=False)
    background = {tuple(key.split("|")): archive[key] for key in archive.files}
    return rows, background


def save(caller, rows, background, out_dir=None):
    loci_path, background_path = cache_paths(caller, out_dir)
    loci_path.parent.mkdir(parents=True, exist_ok=True)
    with open(loci_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(background_path,
                        **{"|".join(key): values for key, values in background.items()})
    return loci_path, background_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caller", default="OnTAD", choices=sorted(SOURCES))
    parser.add_argument("--resolutions", nargs="+", default=RESOLUTIONS)
    parser.add_argument("--chroms", nargs="+", default=None)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    print(f"caller: {args.caller}")
    rows, background = build(args.caller, args.resolutions, args.chroms)
    loci_path, background_path = save(args.caller, rows, background, args.out_dir)
    print(f"\n{len(rows)} loci with both halves scored")
    if rows:
        za = np.array([r["z_a"] for r in rows])
        zb = np.array([r["z_b"] for r in rows])
        print(f"  mean z_a {za.mean():+.4f}   mean z_b {zb.mean():+.4f}   "
              f"corr(z_a, z_b) {np.corrcoef(za, zb)[0, 1]:+.4f}")
        for event_type in sorted({r["event_type"] for r in rows}):
            mask = np.array([r["event_type"] == event_type for r in rows])
            print(f"  {event_type:6} n={int(mask.sum()):4}  "
                  f"z_a {za[mask].mean():+.4f}  z_b {zb[mask].mean():+.4f}")
    print(f"wrote {loci_path}\n      {background_path}")


if __name__ == "__main__":
    main()
