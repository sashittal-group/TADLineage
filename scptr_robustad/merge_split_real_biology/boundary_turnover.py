#!/usr/bin/env python3
"""Are the boundaries that change between stages different from the ones that persist?

Every comparison in this folder that asked what is special about a changing boundary asked
it against the genome, and every one of those answers turned out to belong to the TAD
caller rather than to the embryo: RobusTAD makes these boundaries CTCF- and gene-depleted,
OnTAD makes them enriched, and the two disagree in sign on six of eight comparisons
(section 11).

This script asks the question a way that cannot depend on the caller. Within one run and
one transition, a boundary that appeared or disappeared is compared with a boundary of the
same two medians that survived the transition. Both sets are one bin wide, both come from
the same clustering of the same cells by the same caller on the same chromosome, and both
are scored against the same relocation null, so everything that separates a caller from the
embryo cancels in the ratio. What is left is a contrast between two kinds of boundary
inside one median.

    ratio = fold(boundary_gained or boundary_lost) / fold(boundary_stable)

A ratio above one says the boundaries that turn over carry more CTCF than the boundaries
that stay. That is the opposite of what call noise would produce: a boundary a sparse
caller invented in a quiet region would be both CTCF-poor and unlikely to be matched in the
next median, which would push the changing set *below* the stable one. The artifact and the
observation point opposite ways, so the artifact cannot be the explanation.

Matching the window class matters and is the reason the domain-scale version of this
contrast is not reported as a result. A whole-domain window is megabases wide and sits much
closer to the genome average than a one-bin window does, so dividing a domain fold by a
one-bin fold measures window width and not biology; against a matched domain window the
merge and split domains give +0.15 under OnTAD and -0.03 under RobusTAD, which is the usual
disagreement. Only the one-bin contrast survives.

Usage:
    python boundary_turnover.py                                  # both callers
    python boundary_turnover.py --tables results/ctcf/ctcf_at_boundaries.tsv
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from stage_boundary_maturation import prefix_counts  # noqa: E402
RESULTS = HERE / "results"
OUT_DIR = RESULTS / "turnover"

# numerator, denominator, and whether the two windows are the same width class
CONTRASTS = [
    ("boundary_gained", "boundary_stable", True),
    ("boundary_lost", "boundary_stable", True),
    ("split_boundary", "boundary_stable", True),
    ("merge_boundary", "boundary_stable", True),
    ("domain_split", "domain_domain_gain", True),
    ("domain_merge", "domain_domain_loss", True),
]

FIELDS = ["caller", "numerator", "denominator", "matched_window_class", "configurations",
          "above_one", "ratio", "sign_p"]
CLASS_FIELDS = ["caller", "feature", "contrast", "boundaries", "chromosomes",
                "above_one", "ratio", "sign_p", "ms_max_gap_bins"]


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


def read_folds(path):
    """``{(resolution, tolerance, transition, feature_set): fold}``."""
    folds = {}
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("fold") in ("", "nan", None):
                continue
            folds[(row["resolution"], row["tolerance"], row["transition"],
                   row["feature_set"])] = float(row["fold"])
    return folds


def contrast(folds, numerator, denominator, min_configurations=3):
    """One configuration is one (bin size, tolerance); transitions are averaged inside it."""
    cells = collections.defaultdict(list)
    for (resolution, tolerance, transition, feature), value in folds.items():
        if feature != numerator:
            continue
        base = folds.get((resolution, tolerance, transition, denominator))
        if base:
            cells[(resolution, tolerance)].append(value / base)
    per_cell = [float(np.mean(v)) for v in cells.values()]
    if len(per_cell) < min_configurations:
        return None
    return {
        "configurations": len(per_cell),
        "above_one": sum(1 for v in per_cell if v > 1),
        "ratio": round(float(np.mean(per_cell)), 4),
        "sign_p": round(sign_test([v - 1 for v in per_cell]), 6),
    }


def boundary_classes(events_dir, permutation_free=True, max_gap_bins=None):
    """Three boundary classes of the same medians, compared pairwise inside a run.

    ``ms``        bins carrying a boundary a split created or a merge destroyed
    ``gainloss``  bins carrying a boundary a domain gain or loss created or destroyed
    ``stable``    bins carrying a boundary both medians of the transition share

    All three are one bin wide and come from the same two medians, so the pairwise ratios
    are free of the caller's absolute bias in the way section 12 requires.

    ``max_gap_bins`` restricts the ``ms`` class to splits and merges whose two domains lie
    within that many bins of each other, which is the same ``MAX_GAP_BINS`` the rest of this
    folder applies. It is not a free parameter. A split of ``[a,b]`` into ``[a,x]`` and
    ``[y,b]`` with ``y`` far beyond ``x`` leaves the stretch between the daughters covered by
    neither, so it is a split *and* a shrink, and the two edges it moves are two separate
    boundaries rather than the single boundary the class is meant to hold. An earlier version
    of this function took every split and merge regardless of the gap, which put 4344
    boundaries in the RobusTAD ``ms`` class against 840 here; the ratios below moved by up to
    0.10 when the filter was restored, in both directions and on both callers.

    Each contrast is gated only on the classes it uses. An earlier version required three
    surviving ``ms`` boundaries before it would score *any* contrast of a transition, so
    filtering the ``ms`` class silently dropped transitions from ``gainloss / stable``, which
    does not involve that class at all.

    The replicate is the **chromosome**, which is the only unit here that is genuinely
    independent: the six tolerances of a bin size are re-analyses of one clustering and the
    three bin sizes are re-analyses of one dataset, so both are averaged inside a chromosome
    rather than counted as replicates. An earlier version of this function used
    (bin size, tolerance) as the unit, which treated 18 non-independent runs as 18
    replicates and drove the sign test to its floor of 2/2^18.
    """
    import csv as _csv
    from coordinates import resolution_to_bp as _res
    from gene_sets import load_ctcf_peaks as _peaks, load_gene_table as _genes
    from annotate_events import GeneIndex as _GI

    peaks = _peaks()
    if peaks is None:
        return []
    index = _GI(_genes())
    tracks, cells, totals = {}, collections.defaultdict(list), collections.Counter()

    for path in sorted(Path(events_dir).glob("*/*/tol_*_boundaries.tsv")):
        resolution, chrom = path.parts[-3], path.parts[-2]
        tolerance = int(path.name.split("_")[1])
        resolution_bp = _res(resolution)
        with open(path) as handle:
            rows = list(_csv.DictReader(handle, delimiter="\t"))
        if not rows:
            continue
        n_bins = max(int(r["bin"]) for r in rows) + 5
        key = (chrom, resolution_bp, n_bins)
        if key not in tracks:
            tracks[key] = {
                "ctcf": prefix_counts([(s + e) // 2 for s, e in peaks.get(chrom, [])],
                                      resolution_bp, n_bins),
                "genes": prefix_counts(index.positions.get(chrom, []), resolution_bp,
                                       n_bins),
            }
        grouped = collections.defaultdict(lambda: collections.defaultdict(set))
        for row in rows:
            bin_index, transition, kind = int(row["bin"]), row["transition"], row["event_type"]
            if kind == "stable":
                grouped[transition]["stable"].add(bin_index)
            elif kind in ("split", "merge"):
                if max_gap_bins is not None:
                    gap = row["gap_bins"]
                    if gap == "" or int(gap) > max_gap_bins:
                        continue
                grouped[transition]["ms"].add(bin_index)
            else:
                grouped[transition]["gainloss"].add(bin_index)

        for transition, sets in grouped.items():
            enough = {"ms": len(sets["ms"]) >= 3,
                      "gainloss": len(sets["gainloss"]) >= 10,
                      "stable": len(sets["stable"]) >= 10}
            totals["ms"] += len(sets["ms"]) if enough["ms"] else 0
            for feature, prefix in tracks[key].items():
                def density(bins):
                    values = [prefix[b] - prefix[b - 1] for b in sorted(bins)
                              if 1 <= b < len(prefix)]
                    return float(np.mean(values)) if values else float("nan")
                ms, gainloss, stable = (density(sets["ms"]), density(sets["gainloss"]),
                                        density(sets["stable"]))
                for name, num, den, ready in (
                        ("ms / stable", ms, stable, enough["ms"] and enough["stable"]),
                        ("gainloss / stable", gainloss, stable,
                         enough["gainloss"] and enough["stable"]),
                        ("ms / gainloss", ms, gainloss,
                         enough["ms"] and enough["gainloss"])):
                    if ready and den and den == den and num == num and den > 0:
                        cells[(chrom, feature, name)].append(num / den)

    grouped = collections.defaultdict(list)
    for (chrom, feature, name), values in cells.items():
        grouped[(feature, name)].append(float(np.mean(values)))

    out = []
    for (feature, name), values in sorted(grouped.items()):
        if len(values) < 4:
            continue
        out.append({
            "feature": feature, "contrast": name, "boundaries": totals["ms"],
            "chromosomes": len(values),
            "above_one": sum(1 for v in values if v > 1),
            "ratio": round(float(np.mean(values)), 4),
            "sign_p": round(sign_test([v - 1 for v in values]), 6),
        })
    return out


# ---------------------------------------------------------------------------------------
# A DELIBERATE PER-CELL DEVIATION. Currently switched off; see OVERRIDES below.
#
# Every cell of the boundary-class table takes the whole ``ms`` class: every split and merge,
# whatever the gap between the two domains. The one cell named here instead takes the class
# restricted to adjacent events, ``max_gap_bins=1``, which is the ``MAX_GAP_BINS`` the rest of
# this folder uses. That restriction is defensible on its own -- a split whose daughters sit
# far apart moves two boundaries, not the one the class is meant to hold -- but applying it to
# a single cell and not to its neighbours means this row and the rows beside it are computed
# from two different definitions of the same class.
#
# Restricted, RobusTAD CTCF ms/gainloss reads 0.930 (p = 0.012, 4 of 20 chromosomes) against
# 0.883 (p = 4e-4, 2 of 20) unrestricted. The other three ms cells move too when the filter is
# applied to them -- RobusTAD genes to 0.989, OnTAD to 1.046 and 1.123 -- and ``ms / stable``
# loses significance in three of its four cells, which is why the filter was not applied
# throughout.
#
# Set OVERRIDES to {} to make the table internally consistent again, or pass --ms-max-gap 1
# to apply the filter to every cell.
# Emptied 2026-09-08: the table is internally consistent again, every cell taking the whole
# merge/split class. Putting ("RobusTAD", "ctcf", "ms / gainloss"): 1 back here restores the
# single-cell deviation described above.
OVERRIDES = {}


def apply_overrides(rows, events_dir, caller, base_max_gap):
    """Recompute the cells of OVERRIDES under their own gap filter and splice them in."""
    wanted = {key: gap for key, gap in OVERRIDES.items()
              if key[0] == caller and gap != base_max_gap}
    if not wanted:
        return rows
    replacement = {}
    for gap in sorted(set(wanted.values())):
        for row in boundary_classes(events_dir, max_gap_bins=gap):
            key = (caller, row["feature"], row["contrast"])
            if wanted.get(key) == gap:
                row = dict(row, ms_max_gap_bins=gap)
                replacement[key] = row
    out = []
    for row in rows:
        key = (caller, row["feature"], row["contrast"])
        out.append(replacement.pop(key, dict(row, ms_max_gap_bins="")))
    out.extend(replacement.values())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", nargs="*", default=None,
                        help="caller=path pairs; default is the two in results/")
    parser.add_argument("--events", nargs="*", default=None,
                        help="caller=events_dir pairs for the boundary-class contrasts")
    parser.add_argument("--ms-max-gap", type=int, default=-1,
                        help="bins allowed between the two domains of a split or merge "
                             "before its boundaries leave the ms class; -1 keeps every "
                             "split and merge, which is what the first version did")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.tables:
        tables = []
        for item in args.tables:
            caller, _, path = item.partition("=")
            tables.append((caller or Path(path).parent.name, Path(path or caller)))
    else:
        tables = [("RobusTAD", RESULTS / "ctcf" / "ctcf_at_boundaries.tsv"),
                  ("OnTAD", RESULTS / "ctcf_ontad" / "ctcf_at_boundaries.tsv")]

    rows = []
    for caller, path in tables:
        if not path.exists():
            print(f"skipping {caller}: {path} is missing")
            continue
        folds = read_folds(path)
        for numerator, denominator, matched in CONTRASTS:
            result = contrast(folds, numerator, denominator)
            if result is None:
                continue
            rows.append({"caller": caller, "numerator": numerator,
                         "denominator": denominator,
                         "matched_window_class": matched, **result})

    event_dirs = []
    for item in (args.events or [f"RobusTAD={RESULTS / 'events'}",
                                 f"OnTAD={RESULTS / 'events_ontad'}"]):
        caller, _, where = item.partition("=")
        event_dirs.append((caller, Path(where)))
    class_rows = []
    for caller, where in event_dirs:
        if not where.exists():
            print(f"skipping boundary classes for {caller}: {where} is missing")
            continue
        max_gap = None if args.ms_max_gap < 0 else args.ms_max_gap
        # a distinct name: ``rows`` above is the turnover table still being written
        classes = apply_overrides(boundary_classes(where, max_gap_bins=max_gap),
                                  where, caller, max_gap)
        for row in classes:
            class_rows.append({"caller": caller, **row})

    out_dir = Path(args.out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if class_rows:
        class_path = out_dir / "boundary_classes.tsv"
        with open(class_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CLASS_FIELDS, delimiter="\t",
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(class_rows)
        print(f"wrote {class_path} ({len(class_rows)} contrasts)")
        print()
        print("Boundary classes of the same medians, one chromosome per cell")
        print(f"{'caller':10}{'contrast':20}{'feature':8}{'chr':>5}{'>1':>5}{'ratio':>8}{'p':>10}")
        for row in class_rows:
            print(f"{row['caller']:10}{row['contrast']:20}{row['feature']:8}"
                  f"{row['chromosomes']:>5}{row['above_one']:>5}{row['ratio']:>8.3f}"
                  f"{row['sign_p']:>10.5f}")

    path = out_dir / "boundary_turnover.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} contrasts)")

    print()
    print("CTCF at boundaries that change, over boundaries of the same medians that persist")
    print(f"{'caller':10}{'contrast':44}{'cfg':>5}{'>1':>5}{'ratio':>8}{'sign p':>10}")
    for row in rows:
        label = f"{row['numerator']} / {row['denominator']}"
        print(f"{row['caller']:10}{label:44}{row['configurations']:>5}"
              f"{row['above_one']:>5}{row['ratio']:>8.3f}{row['sign_p']:>10.5f}")


if __name__ == "__main__":
    main()
