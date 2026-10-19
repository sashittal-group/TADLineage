#!/usr/bin/env python3
"""Is there anything special about the places where a domain splits or two domains merge?

Splits and merges are the events the rest of this project is built to find, so they are the
ones worth asking a biological question about. Section 4 of the report already scored them
against relocated windows and found them slightly depleted of genes and CTCF. That test is
now known to be confounded: `stage_boundary_maturation.py` shows that the boundaries of a
stage median are depleted early in development and not depleted late, so any window set
pooled over stages inherits whichever stages contributed most to it.

This script removes that confound by changing the comparison. Instead of asking whether a
merge or split boundary looks unusual against the genome, it asks whether one looks unusual
against *the other boundaries of the same two medians* -- same chromosome, same caller, same
stage pair, same clustering run. A ratio of one then means a merge or split boundary is an
ordinary boundary of its own medians, which is the question the confounded test could not
answer.

Four tests are run, and every one of them is null:

``context``
    CTCF and gene density at adjacent merge and split boundaries, divided by the density at
    every boundary of the two medians the event was called between.
``persistence``
    whether a boundary created by a split survives into the 64-cell median more often than
    any other boundary of the median it was created in. A boundary installed by a real
    developmental event should outlast one that appears and vanishes.
``pooled``
    the same context ratio with every chromosome pooled into one number per (bin size,
    tolerance), because adjacent merge and split events are rare enough that a single
    chromosome carries a handful.
``loci``
    curated gene sets against the reproducible loci of ``results/candidates.tsv``, by a
    null that slides each locus to a random position on its own chromosome.

The ``loci`` test is where the one apparent signal lives, and it does not survive. Three
subunits of the PRC1.6 complex -- PCGF6, E2F6 and RYBP -- sit under three separate merge
loci, which is more than chance at p = 0.003. That hypothesis was generated from these same
loci, so the p-value is not a discovery p-value, and the honest way to judge it is the
prediction it makes: PRC1.6 exists to repress germline and meiotic genes, so those targets
should sit on merge loci too. Not one of the forty-five does, against an expectation of
about one. The subunit result is therefore reported as a coincidence, not a finding.

A null result is only worth reporting if it had the power to be otherwise, so the context
test also reports the smallest ratio it could have separated from one at each bin size.

Usage:
    python merge_split_context.py --permutations 5000
    python merge_split_context.py --resolution 500kb --max-gap-bins 1
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
from coordinates import MM10_SIZES, resolution_to_bp
from gene_sets import load_ctcf_peaks, load_curated_sets, load_gene_table
from annotate_events import GeneIndex
from stage_boundary_maturation import (
    boundary_bins,
    prefix_counts,
    sign_test,
    wilcoxon_signed_rank,
)

EVENTS_DIR = HERE / "results" / "events"
OUT_DIR = HERE / "results" / "merge_split"
STAGE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
TRANSITIONS = [f"{a}->{b}" for a, b in zip(STAGE_ORDER, STAGE_ORDER[1:])]

CONTEXT_FIELDS = ["resolution", "chrom", "tolerance", "transition", "kind", "feature",
                  "n_event_bins", "n_background_bins", "event_per_bin",
                  "background_per_bin", "ratio"]
PERSIST_FIELDS = ["resolution", "chrom", "tolerance", "transition", "n_split",
                  "split_surviving", "n_background", "background_surviving", "difference"]
POOLED_FIELDS = ["resolution", "tolerance", "kind", "feature", "n_event_bins",
                 "event_per_bin", "background_per_bin", "ratio"]
EVENT_FIELDS = ["kind", "feature", "background", "max_gap_bins", "n_events",
                "observed_per_bin", "null_per_bin", "ratio", "p_two_sided"]
LOCUS_FIELDS = ["locus_set", "n_loci", "gene_set", "set_size", "observed", "null_mean",
                "fold", "p_greater", "hits"]
SUMMARY_FIELDS = ["test", "resolution", "kind", "feature", "cells", "positive", "mean",
                  "wilcoxon_p", "sign_p", "detectable_ratio"]


def read_boundary_rows(resolution, chrom, tolerance, events_dir=None):
    path = (Path(events_dir or EVENTS_DIR) / str(resolution) / chrom
            / f"tol_{tolerance}_boundaries.tsv")
    if not path.exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def event_boundary_bins(rows, transition, kind, max_gap_bins):
    """Bins carrying an adjacent split or merge boundary of one transition.

    Only events whose two domains are adjacent to within ``max_gap_bins`` are taken, since
    those are the ones where a single boundary demonstrably appeared or disappeared; a
    split whose daughters are far apart replaced a domain with two distant pieces and its
    edges are not one boundary change.
    """
    out = set()
    for row in rows:
        if row["transition"] != transition or row["event_type"] != kind:
            continue
        gap = row["gap_bins"]
        if gap != "" and int(gap) <= max_gap_bins:
            out.add(int(row["bin"]))
    return sorted(out)


def mean_density(prefix, bins):
    """Mean features per bin over a list of bin indices."""
    values = [prefix[b] - prefix[b - 1] for b in bins if 1 <= b < len(prefix)]
    return float(np.mean(values)) if values else float("nan")


def sum_density(prefix, bins):
    """Total features and bin count, for pooling across chromosomes."""
    values = [prefix[b] - prefix[b - 1] for b in bins if 1 <= b < len(prefix)]
    return float(np.sum(values)), len(values)


def analyse_runs(resolutions, chroms_of, tolerances, peaks, index, max_gap_bins,
                 run_dir, events_dir, min_event_bins=3):
    """The context and persistence tables, one row per run, transition and feature."""
    context, persistence, pooled = [], [], {}
    for resolution in resolutions:
        resolution_bp = resolution_to_bp(resolution)
        for chrom in chroms_of(resolution):
            for tolerance in tolerances:
                path = Path(run_dir) / resolution / chrom / f"tol_{tolerance}.json"
                if not path.exists():
                    continue
                by_stage, _ = stage_medians(json.load(open(path)))
                if len(by_stage) < len(STAGE_ORDER):
                    continue
                rows = read_boundary_rows(resolution, chrom, tolerance, events_dir)
                if not rows:
                    continue

                covered = {b for intervals in by_stage.values()
                           for a, z in intervals for b in range(int(a), int(z) + 1)}
                n_bins = max(covered)
                tracks = {
                    "ctcf": prefix_counts([(s + e) // 2 for s, e in peaks.get(chrom, [])],
                                          resolution_bp, n_bins),
                    "genes": prefix_counts(index.positions.get(chrom, []),
                                           resolution_bp, n_bins),
                }
                edges = {stage: boundary_bins(intervals)
                         for stage, intervals in by_stage.items()}
                final = set(edges["64-cell"])

                for transition in TRANSITIONS:
                    source, target = transition.split("->")
                    background = sorted(set(edges[source]) | set(edges[target]))
                    if len(background) < 10:
                        continue
                    for kind in ("split", "merge"):
                        bins = event_boundary_bins(rows, transition, kind, max_gap_bins)
                        for feature, prefix in tracks.items():
                            total, count = sum_density(prefix, bins)
                            back_total, back_count = sum_density(prefix, background)
                            key = (resolution, tolerance, kind, feature)
                            store = pooled.setdefault(key, [0.0, 0, 0.0, 0])
                            store[0] += total
                            store[1] += count
                            store[2] += back_total
                            store[3] += back_count
                            if len(bins) < min_event_bins:
                                continue
                            event = mean_density(prefix, bins)
                            back = mean_density(prefix, background)
                            context.append({
                                "resolution": resolution, "chrom": chrom,
                                "tolerance": tolerance, "transition": transition,
                                "kind": kind, "feature": feature,
                                "n_event_bins": len(bins),
                                "n_background_bins": len(background),
                                "event_per_bin": round(event, 4),
                                "background_per_bin": round(back, 4),
                                "ratio": round(event / back, 4) if back else "",
                            })

                    # persistence is only defined while a later median still exists
                    if target == "64-cell":
                        continue
                    splits = event_boundary_bins(rows, transition, "split", max_gap_bins)
                    if len(splits) < min_event_bins:
                        continue

                    def surviving(bins):
                        return float(np.mean([any(abs(b - f) <= 1 for f in final)
                                              for b in bins]))
                    split_rate = surviving(splits)
                    back_rate = surviving(edges[target])
                    persistence.append({
                        "resolution": resolution, "chrom": chrom,
                        "tolerance": tolerance, "transition": transition,
                        "n_split": len(splits),
                        "split_surviving": round(split_rate, 4),
                        "n_background": len(edges[target]),
                        "background_surviving": round(back_rate, 4),
                        "difference": round(split_rate - back_rate, 4),
                    })

    pooled_rows = []
    for (resolution, tolerance, kind, feature), store in sorted(pooled.items()):
        total, count, back_total, back_count = store
        if count == 0 or back_count == 0:
            continue
        event = total / count
        back = back_total / back_count
        pooled_rows.append({
            "resolution": resolution, "tolerance": tolerance, "kind": kind,
            "feature": feature, "n_event_bins": count,
            "event_per_bin": round(event, 4), "background_per_bin": round(back, 4),
            "ratio": round(event / back, 4) if back else "",
        })
    return context, persistence, pooled_rows


def interior_bins(intervals, gap=1):
    """Bins between two domains of one median that lie within ``gap`` bins of each other.

    This is the background a merge or split boundary actually needs. Such a boundary is by
    construction the wall between two adjacent domains, and a wall is not the same object
    as the outer edge of an isolated domain: pooled over the grid, an interior bin carries
    about 40 CTCF peaks against about 57 for a domain edge. Scoring merge and split
    boundaries against every boundary of a median therefore compares them with a
    population they are not drawn from, and manufactures a depletion of roughly ten per
    cent at p = 0.0003 that disappears the moment the background is matched.
    """
    ordered = sorted((int(a), int(b)) for a, b in intervals)
    out = set()
    for (_, left_end), (right_start, _) in zip(ordered, ordered[1:]):
        if 0 <= right_start - left_end - 1 <= gap:
            out.add(left_end)
            out.add(right_start)
    return out


def event_level_test(resolutions, chroms_of, tolerances, peaks, index, run_dir,
                     events_dir, permutations, seed, max_gap_bins=2):
    """Every merge or split boundary is a unit, not every chromosome.

    The chromosome-level test in ``summarise`` has at most ten replicates and can only see
    a departure of about half, which is too blunt to call anything. This test keeps the
    same within-run logic but makes the event the unit: within each run and transition it
    draws as many bins from that run's own background as the run contributed events, so
    every run's chromosome, caller and stage pair are preserved, and pools the draws. That
    is roughly five hundred units instead of ten.

    Both backgrounds are reported, because the difference between them is the whole story:
    against every boundary of the medians the events look depleted, and against the
    matched interior boundaries they do not.
    """
    strata = {}
    for resolution in resolutions:
        resolution_bp = resolution_to_bp(resolution)
        for chrom in chroms_of(resolution):
            for tolerance in tolerances:
                path = Path(run_dir) / resolution / chrom / f"tol_{tolerance}.json"
                if not path.exists():
                    continue
                by_stage, _ = stage_medians(json.load(open(path)))
                if len(by_stage) < len(STAGE_ORDER):
                    continue
                rows = read_boundary_rows(resolution, chrom, tolerance, events_dir)
                if not rows:
                    continue
                covered = {b for intervals in by_stage.values()
                           for a, z in intervals for b in range(int(a), int(z) + 1)}
                n_bins = max(covered)
                tracks = {
                    "ctcf": prefix_counts([(s + e) // 2 for s, e in peaks.get(chrom, [])],
                                          resolution_bp, n_bins),
                    "genes": prefix_counts(index.positions.get(chrom, []),
                                           resolution_bp, n_bins),
                }
                for transition in TRANSITIONS:
                    source, target = transition.split("->")
                    backgrounds = {
                        "all boundaries": sorted(set(boundary_bins(by_stage[source]))
                                                 | set(boundary_bins(by_stage[target]))),
                        "interior only": sorted(interior_bins(by_stage[source])
                                                | interior_bins(by_stage[target])),
                    }
                    for kind in ("split", "merge"):
                        bins = event_boundary_bins(rows, transition, kind, max_gap_bins)
                        if not bins:
                            continue
                        for name, background in backgrounds.items():
                            if len(background) < 10:
                                continue
                            for feature, prefix in tracks.items():
                                event = np.array([prefix[b] - prefix[b - 1] for b in bins
                                                  if 1 <= b < len(prefix)])
                                back = np.array([prefix[b] - prefix[b - 1]
                                                 for b in background
                                                 if 1 <= b < len(prefix)])
                                if len(event) == 0 or len(back) == 0:
                                    continue
                                strata.setdefault((kind, feature, name), []).append(
                                    (event, back))

    rng = np.random.default_rng(seed)
    out = []
    for (kind, feature, name), blocks in sorted(strata.items()):
        observed = float(np.concatenate([e for e, _ in blocks]).mean())
        total = sum(len(e) for e, _ in blocks)
        null = np.zeros(permutations)
        for event, back in blocks:
            drawn = rng.integers(0, len(back), size=(permutations, len(event)))
            null += back[drawn].sum(axis=1)
        null /= total
        mean = float(null.mean())
        tail = min(int((null >= observed).sum()), int((null <= observed).sum()))
        out.append({
            "kind": kind, "feature": feature, "background": name,
            "max_gap_bins": max_gap_bins, "n_events": total,
            "observed_per_bin": round(observed, 3),
            "null_per_bin": round(mean, 3),
            "ratio": round(observed / mean, 4) if mean else "",
            "p_two_sided": round(min(1.0, 2 * (tail + 1) / (permutations + 1)), 5),
        })
    return out


def detectable_ratio(values):
    """The smallest departure from one this many cells of this spread would have shown.

    A null result means nothing without it. With ``n`` cells of standard deviation ``s``,
    a two-sided test at the 0.05 level has about 80% power against a shift of
    ``2.8 * s / sqrt(n)``, so a ratio further from one than that would have been caught.
    """
    values = np.asarray([v for v in values if v == v], dtype=float)
    if len(values) < 3:
        return float("nan")
    return float(2.8 * values.std(ddof=1) / np.sqrt(len(values)))


def summarise(context, persistence):
    """Chromosomes as replicates, tolerances averaged, one test per bin size."""
    out = []
    cells = {}
    for row in context:
        if row["ratio"] == "":
            continue
        key = (row["resolution"], row["kind"], row["feature"], row["chrom"])
        cells.setdefault(key, []).append(float(row["ratio"]))
    grouped = {}
    for (resolution, kind, feature, chrom), values in cells.items():
        grouped.setdefault((resolution, kind, feature), []).append(float(np.mean(values)))
    for (resolution, kind, feature), values in sorted(grouped.items()):
        deltas = [v - 1.0 for v in values]
        out.append({
            "test": "context ratio", "resolution": resolution, "kind": kind,
            "feature": feature, "cells": len(values),
            "positive": sum(1 for d in deltas if d > 0),
            "mean": round(float(np.mean(values)), 4),
            "wilcoxon_p": round(wilcoxon_signed_rank(deltas), 6),
            "sign_p": round(sign_test(deltas), 6),
            "detectable_ratio": round(detectable_ratio(values), 4),
        })

    cells = {}
    for row in persistence:
        key = (row["resolution"], row["chrom"])
        cells.setdefault(key, []).append(float(row["difference"]))
    grouped = {}
    for (resolution, chrom), values in cells.items():
        grouped.setdefault(resolution, []).append(float(np.mean(values)))
    for resolution, values in sorted(grouped.items()):
        out.append({
            "test": "split persistence", "resolution": resolution, "kind": "split",
            "feature": "survives to 64-cell", "cells": len(values),
            "positive": sum(1 for v in values if v > 0),
            "mean": round(float(np.mean(values)), 4),
            "wilcoxon_p": round(wilcoxon_signed_rank(values), 6),
            "sign_p": round(sign_test(values), 6),
            "detectable_ratio": round(detectable_ratio(values), 4),
        })
    return out


def locus_gene_sets(gene_sets, index, permutations, seed, candidates_path=None):
    """Curated sets against the reproducible loci, by sliding each locus on its chromosome."""
    path = Path(candidates_path or HERE / "results" / "candidates.tsv")
    if not path.exists():
        return []
    with open(path) as handle:
        loci = list(csv.DictReader(handle, delimiter="\t"))
    if not loci:
        return []

    by_chrom = {}
    for chrom, positions in index.positions.items():
        by_chrom[chrom] = (np.asarray(positions, dtype=np.int64),
                           list(index.symbols[chrom]))

    def count(windows, symbols):
        found = []
        for chrom, start, end in windows:
            entry = by_chrom.get(chrom)
            if entry is None:
                continue
            positions, names = entry
            left = int(np.searchsorted(positions, start, "left"))
            right = int(np.searchsorted(positions, end, "left"))
            found.extend(name for name in names[left:right] if name in symbols)
        return found

    rng = np.random.default_rng(seed)
    rows = []
    for label in ("merge", "split", "all"):
        chosen = [l for l in loci if label == "all" or l["event_type"] == label]
        if not chosen:
            continue
        windows = [(l["chrom"], int(l["start"]), int(l["end"])) for l in chosen]
        widths = [(chrom, end - start) for chrom, start, end in windows]
        draws = []
        for _ in range(permutations):
            drawn = []
            for chrom, width in widths:
                room = max(1, MM10_SIZES[chrom] - width)
                start = int(rng.integers(0, room))
                drawn.append((chrom, start, start + width))
            draws.append(drawn)
        for name, symbols in sorted(gene_sets.items()):
            if not symbols:
                continue
            observed = count(windows, symbols)
            null = np.array([len(count(d, symbols)) for d in draws], dtype=float)
            mean = float(null.mean())
            rows.append({
                "locus_set": label, "n_loci": len(chosen), "gene_set": name,
                "set_size": len(symbols), "observed": len(observed),
                "null_mean": round(mean, 3),
                "fold": round(len(observed) / mean, 3) if mean else "",
                "p_greater": round(float((np.sum(null >= len(observed)) + 1)
                                         / (permutations + 1)), 5),
                "hits": ",".join(sorted(set(observed))),
            })
    return rows


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--tolerance", type=int, default=None)
    parser.add_argument("--max-gap-bins", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--candidates", default=None,
                        help="candidates.tsv for the locus gene-set test; default is the "
                             "one beside this script's results")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    peaks = load_ctcf_peaks()
    if peaks is None:
        print("No CTCF peaks in reference/. Run fetch_reference_data.py --with-ctcf.")
        return
    gene_table = load_gene_table()
    index = GeneIndex(gene_table)
    gene_sets = load_curated_sets(genes=gene_table)

    run_dir = Path(args.run_dir or DEFAULT_RUN_DIR)
    events_dir = Path(args.events_dir or EVENTS_DIR)
    resolutions = ([args.resolution] if args.resolution
                   else sorted(p.name for p in run_dir.iterdir() if p.is_dir()))
    tolerances = ([args.tolerance] if args.tolerance is not None else list(range(6)))

    def chroms_of(resolution):
        return sorted(p.name for p in (run_dir / resolution).glob("chr*"))

    context, persistence, pooled = analyse_runs(
        resolutions, chroms_of, tolerances, peaks, index, args.max_gap_bins,
        run_dir, events_dir)
    summary = summarise(context, persistence)
    loci = locus_gene_sets(gene_sets, index, args.permutations, args.seed,
                           candidates_path=args.candidates)
    events = event_level_test(resolutions, chroms_of, tolerances, peaks, index, run_dir,
                              events_dir, args.permutations, args.seed,
                              max_gap_bins=max(2, args.max_gap_bins))

    out_dir = Path(args.out_dir or OUT_DIR)
    write(out_dir / "boundary_context.tsv", CONTEXT_FIELDS, context)
    write(out_dir / "split_persistence.tsv", PERSIST_FIELDS, persistence)
    write(out_dir / "pooled_context.tsv", POOLED_FIELDS, pooled)
    write(out_dir / "locus_gene_sets.tsv", LOCUS_FIELDS, loci)
    write(out_dir / "merge_split_summary.tsv", SUMMARY_FIELDS, summary)
    write(out_dir / "event_level_test.tsv", EVENT_FIELDS, events)

    print()
    print("Merge and split boundaries against the other boundaries of the same medians")
    print(f"{'test':18}{'bin size':10}{'kind':7}{'feature':9}{'cells':>6}{'ratio':>8}"
          f"{'p':>9}{'detectable':>12}")
    for row in summary:
        print(f"{row['test']:18}{row['resolution']:10}{row['kind']:7}"
              f"{row['feature'][:8]:9}{row['cells']:>6}{row['mean']:>8.3f}"
              f"{row['wilcoxon_p']:>9.3g}{row['detectable_ratio']:>12.3f}")

    print()
    print("Event-level test: every boundary a unit, background matched or not")
    print(f"{'kind':7}{'feature':8}{'background':17}{'events':>8}{'ratio':>8}{'p':>10}")
    for row in events:
        print(f"{row['kind']:7}{row['feature']:8}{row['background']:17}"
              f"{row['n_events']:>8}{row['ratio']:>8.3f}{row['p_two_sided']:>10.4f}")

    print()
    print("Curated gene sets at the reproducible loci (p <= 0.05 only)")
    for row in loci:
        if row["p_greater"] <= 0.05:
            print(f"  {row['locus_set']:6} {row['gene_set']:32} "
                  f"observed {row['observed']} vs {row['null_mean']:.2f} "
                  f"p={row['p_greater']:.4f}  {row['hits']}")


if __name__ == "__main__":
    main()
