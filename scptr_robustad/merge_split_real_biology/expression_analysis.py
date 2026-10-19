#!/usr/bin/env python3
"""Do the genes inside a rearranged region change expression at that same transition?

The gene-set tests in ``enrichment.py`` ask whether a window contains a member of a list.
That throws away most of what the transcriptome knows, and with a few dozen split or merge
events it has very little power. This asks the continuous version of the same question: of
the genes that sit in the windows, how much does their expression actually move across the
transition the windows were called at, compared with the same windows placed elsewhere?

Two statistics per window set, both over the genes whose start site falls in a window:

``mean_abs_change``
    the mean of ``|log2 fold change|`` across the transition. This is the one to read for
    "did transcription move here at all", in either direction.

``mean_change``
    the signed mean. A merge that removes insulation and lets an enhancer reach further
    would push this up; a split that installs insulation would push it down. It is the
    directional claim and needs the stronger evidence of the two.

The null is the same relocation used for the gene-set tests: the windows keep their count,
their widths and their chromosomes, and slide to a random position inside the callable
region, with the whole window required to fit. Every window set is also scored against the
three transitions it did *not* come from, so a real effect can be told from a region that
is simply transcriptionally busy: a transition-specific effect is larger on the matched
transition than on the others, and that contrast is reported as ``matched_minus_other``.

Usage:
    python expression_analysis.py --resolution 500kb --tolerance 1
    python expression_analysis.py --all --permutations 5000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from annotate_events import GeneIndex, read_callable, read_events
from coordinates import resolution_to_bp
from enrichment import (
    clip_windows_to_pool,
    benjamini_hochberg,
    build_feature_sets,
    callable_bin_pool,
    read_boundaries,
)
from gene_sets import TRANSITIONS, load_expression, load_gene_table, transition_change

HERE = Path(__file__).resolve().parent
EVENTS_DIR = HERE / "results" / "events"
OUT_DIR = HERE / "results" / "expression"

RESULT_FIELDS = [
    "resolution", "tolerance", "feature_set", "transition", "scored_transition",
    "matched", "n_windows", "n_genes", "statistic", "observed", "null_mean", "null_sd",
    "difference", "z", "p_greater", "p_less", "q_two_sided",
]
CONTRAST_FIELDS = [
    "resolution", "tolerance", "feature_set", "transition", "statistic",
    "matched_value", "other_mean", "matched_minus_other", "z_matched", "z_other_mean",
]


def change_tracks(index, expression, resolution_bp, n_bins):
    """Per bin, the summed and the summed absolute log2 change, plus the gene count.

    Prefix sums over bins let a window of any width be scored in constant time, which is
    what makes thousands of relocations cheap.
    """
    tracks = {}
    for transition in TRANSITIONS:
        per_chrom = {}
        for chrom, size in n_bins.items():
            signed = np.zeros(size + 2)
            absolute = np.zeros(size + 2)
            counts = np.zeros(size + 2)
            for tss, symbol in zip(index.positions.get(chrom, []),
                                   index.symbols.get(chrom, [])):
                change = transition_change(expression, symbol, transition)
                if change is None:
                    continue
                position = min(size + 1, tss // resolution_bp + 1)
                signed[position] += change
                absolute[position] += abs(change)
                counts[position] += 1
            per_chrom[chrom] = (np.cumsum(signed), np.cumsum(absolute), np.cumsum(counts))
        tracks[transition] = per_chrom
    return tracks


def window_sums(prefix, starts, widths):
    last = len(prefix) - 1
    ends = np.minimum(starts + widths - 1, last)
    starts = np.clip(starts, 1, last)
    return prefix[ends] - prefix[starts - 1]


def score_windows(draws, tracks, statistic):
    """Observed and null values of one statistic over one window set."""
    observed_total = observed_count = 0.0
    null_total = null_count = None
    for chrom, (starts, widths, choices) in draws.items():
        signed, absolute, counts = tracks[chrom]
        values = signed if statistic == "mean_change" else absolute

        observed_total += float(window_sums(values, starts, widths).sum())
        observed_count += float(window_sums(counts, starts, widths).sum())

        total = window_sums(values, choices, widths).sum(axis=1)
        number = window_sums(counts, choices, widths).sum(axis=1)
        null_total = total if null_total is None else null_total + total
        null_count = number if null_count is None else null_count + number

    if observed_count == 0 or null_total is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        null_values = np.where(null_count > 0, null_total / np.maximum(null_count, 1),
                               np.nan)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0:
        return None
    return observed_total / observed_count, null_values, int(observed_count)


def build_draws(features_windows, pool, permutations, rng):
    """Relocations of one window set, whole windows kept inside the callable region."""
    by_chrom = {}
    for chrom, start_bin, width in features_windows:
        if chrom in pool and len(pool[chrom]):
            by_chrom.setdefault(chrom, []).append((start_bin, width))
    draws = {}
    for chrom, entries in by_chrom.items():
        bins = pool[chrom]
        last = int(bins.max())
        starts = np.array([s for s, _ in entries], dtype=np.int64)
        widths = np.array([w for _, w in entries], dtype=np.int64)
        choices = np.empty((permutations, len(entries)), dtype=np.int64)
        for width in np.unique(widths):
            columns = np.flatnonzero(widths == width)
            room = bins[bins + width - 1 <= last]
            if len(room) == 0:
                room = bins[:1]
            choices[:, columns] = rng.choice(room, size=(permutations, len(columns)))
        draws[chrom] = (starts, widths, choices)
    return draws


def analyse_configuration(resolution, tolerance, index, expression, permutations=2000,
                          max_gap_bins=1, chroms=None, seed=0, min_windows=5,
                          events_dir=None, flank_bins=0):
    resolution_bp = resolution_to_bp(resolution)
    events = read_events(resolution, tolerance, events_dir, chroms)
    boundaries = read_boundaries(resolution, tolerance, events_dir, chroms)
    features = build_feature_sets(events, boundaries, resolution_bp, max_gap_bins,
                                  flank_bins)
    pool = callable_bin_pool(read_callable(resolution, tolerance, events_dir, chroms),
                             resolution_bp)
    # a flanked window may overhang the chromosome end; the null only ever places
    # windows that fit, so clip the observed ones to match
    features = clip_windows_to_pool(features, pool)
    n_bins = {chrom: int(bins.max()) for chrom, bins in pool.items() if len(bins)}
    tracks = change_tracks(index, expression, resolution_bp, n_bins)

    rng = np.random.default_rng(seed)
    rows, contrasts = [], []
    for (feature_name, transition), windows in sorted(features.items()):
        if len(windows) < min_windows:
            continue
        draws = build_draws(windows, pool, permutations, rng)
        if not draws:
            continue

        for statistic in ("mean_abs_change", "mean_change"):
            scored_rows = {}
            for scored_transition in TRANSITIONS:
                result = score_windows(draws, tracks[scored_transition], statistic)
                if result is None:
                    continue
                observed, null_values, n_genes = result
                null_mean = float(null_values.mean())
                null_sd = float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0
                p_greater = float((np.sum(null_values >= observed) + 1)
                                  / (len(null_values) + 1))
                p_less = float((np.sum(null_values <= observed) + 1)
                               / (len(null_values) + 1))
                row = {
                    "resolution": resolution, "tolerance": tolerance,
                    "feature_set": feature_name, "transition": transition,
                    "scored_transition": scored_transition,
                    "matched": "yes" if scored_transition == transition else "no",
                    "n_windows": len(windows), "n_genes": n_genes,
                    "statistic": statistic, "observed": round(observed, 4),
                    "null_mean": round(null_mean, 4), "null_sd": round(null_sd, 4),
                    "difference": round(observed - null_mean, 4),
                    "z": round((observed - null_mean) / null_sd, 2) if null_sd else "",
                    "p_greater": p_greater, "p_less": p_less,
                }
                rows.append(row)
                scored_rows[scored_transition] = row

            matched = scored_rows.get(transition)
            others = [r for t, r in scored_rows.items() if t != transition]
            if matched and others:
                other_mean = float(np.mean([r["difference"] for r in others]))
                other_z = [r["z"] for r in others if r["z"] != ""]
                contrasts.append({
                    "resolution": resolution, "tolerance": tolerance,
                    "feature_set": feature_name, "transition": transition,
                    "statistic": statistic,
                    "matched_value": matched["difference"],
                    "other_mean": round(other_mean, 4),
                    "matched_minus_other": round(matched["difference"] - other_mean, 4),
                    "z_matched": matched["z"],
                    "z_other_mean": round(float(np.mean(other_z)), 2) if other_z else "",
                })

    two_sided = [min(1.0, 2 * min(row["p_greater"], row["p_less"])) for row in rows]
    for row, q in zip(rows, benjamini_hochberg(two_sided)):
        row["q_two_sided"] = float(q)
    return rows, contrasts


def write_table(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(rows, contrasts, alpha=0.1):
    matched = [r for r in rows if r["matched"] == "yes"]
    significant = [r for r in matched if r["q_two_sided"] <= alpha]
    print(f"{len(significant)} of {len(matched)} transition-matched tests at "
          f"q <= {alpha}")
    if significant:
        print(f"{'feature set':22}{'transition':18}{'statistic':18}"
              f"{'obs':>9}{'null':>9}{'z':>7}{'q':>9}")
        for row in sorted(significant, key=lambda r: r["q_two_sided"])[:20]:
            print(f"{row['feature_set']:22}{row['transition']:18}{row['statistic']:18}"
                  f"{row['observed']:>9.3f}{row['null_mean']:>9.3f}"
                  f"{str(row['z']):>7}{row['q_two_sided']:>9.4f}")
    print()
    strongest = sorted(contrasts, key=lambda c: -abs(c["matched_minus_other"]))[:10]
    print("largest matched-minus-other contrasts (transition specificity):")
    print(f"{'feature set':22}{'transition':18}{'statistic':18}"
          f"{'matched':>9}{'other':>9}{'diff':>9}")
    for row in strongest:
        print(f"{row['feature_set']:22}{row['transition']:18}{row['statistic']:18}"
              f"{row['matched_value']:>9.3f}{row['other_mean']:>9.3f}"
              f"{row['matched_minus_other']:>9.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default="500kb")
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--max-gap-bins", type=int, default=1)
    parser.add_argument("--flank-bins", type=int, default=0,
                        help="positional tolerance in bins: widen every boundary window by this many bins on each side, because sparse single-cell calls place the same boundary a bin or two apart between medians")
    parser.add_argument("--min-windows", type=int, default=5)
    parser.add_argument("--chroms", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--events-dir", default=None,
                        help="event tables to read; use results/events_unimputed to test the medians recomputed from the unimputed calls")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    index = GeneIndex(load_gene_table())
    expression, _ = load_expression()
    out_dir = Path(args.out_dir or OUT_DIR)
    events_dir = Path(args.events_dir) if args.events_dir else EVENTS_DIR

    if not args.all:
        rows, contrasts = analyse_configuration(
            args.resolution, args.tolerance, index, expression,
            permutations=args.permutations, max_gap_bins=args.max_gap_bins,
            chroms=args.chroms, seed=args.seed, min_windows=args.min_windows,
            flank_bins=args.flank_bins, events_dir=events_dir,
        )
        stem = f"{args.resolution}_tol{args.tolerance}"
        write_table(out_dir / f"{stem}_expression.tsv", RESULT_FIELDS, rows)
        write_table(out_dir / f"{stem}_contrasts.tsv", CONTRAST_FIELDS, contrasts)
        print(f"wrote {out_dir / (stem + '_expression.tsv')}")
        report(rows, contrasts)
        return

    configurations = sorted(
        {(path.parent.parent.name, int(path.name.split("_")[1]))
         for path in events_dir.glob("*/*/tol_*_events.tsv")}
    )
    all_rows, all_contrasts = [], []
    for resolution, tolerance in configurations:
        rows, contrasts = analyse_configuration(
            resolution, tolerance, index, expression, permutations=args.permutations,
            max_gap_bins=args.max_gap_bins, chroms=args.chroms, seed=args.seed,
            min_windows=args.min_windows, flank_bins=args.flank_bins,
            events_dir=events_dir,
        )
        stem = f"{resolution}_tol{tolerance}"
        write_table(out_dir / f"{stem}_expression.tsv", RESULT_FIELDS, rows)
        write_table(out_dir / f"{stem}_contrasts.tsv", CONTRAST_FIELDS, contrasts)
        all_rows.extend(rows)
        all_contrasts.extend(contrasts)
        hits = sum(1 for r in rows if r["matched"] == "yes" and r["q_two_sided"] <= 0.1)
        print(f"{resolution} tol {tolerance}: {hits} matched tests at q <= 0.1")
    write_table(out_dir / "all_configurations_expression.tsv", RESULT_FIELDS, all_rows)
    write_table(out_dir / "all_configurations_contrasts.tsv", CONTRAST_FIELDS,
                all_contrasts)
    print(f"wrote {out_dir / 'all_configurations_expression.tsv'}")


if __name__ == "__main__":
    main()
