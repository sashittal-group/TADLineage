#!/usr/bin/env python3
"""Do the domains and boundaries that change land on genes that matter at that transition?

Every test here compares one set of event windows against one set of genes, and asks
whether the genes fall inside the windows more often than they would if the same windows
sat somewhere else on the same chromosome.

The null
    Genes are not spread evenly along a chromosome and neither are RobusTAD calls, so a
    hypergeometric test over the whole genome would call almost anything significant. The
    null used here instead keeps the windows exactly as they are -- same count, same widths,
    same chromosomes -- and slides each one to a random position inside the callable region
    of its own chromosome, the stretch the medians of that run actually cover. The p-value
    is the fraction of relocations that put at least as many genes in the windows as the
    real placement did, and the fold enrichment is against the mean of those relocations.

The tests
    Event windows are grouped by what happened and when: boundaries gained or lost, the
    boundary of a split or a merge specifically, and the domain bodies. Each group is cut
    by transition, so a group from the 2-cell to 4-cell transition is tested against the
    genes that change at *that* transition and, as a control, against the genes that change
    at the other three. A signal that is real should be strongest on the matched diagonal;
    one that is an artefact of where genes sit will be flat across the row.

    The curated sets are tested the same way. ``housekeeping_control`` is there to check the
    method rather than to make a claim: boundaries are known to be gene dense and
    housekeeping rich, so a boundary set that shows nothing at all for it is a warning that
    the windows are not behaving like boundaries.

Multiplicity is handled with Benjamini-Hochberg over every test of a run.

Usage:
    python enrichment.py --resolution 500kb --tolerance 1
    python enrichment.py --all --permutations 5000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from annotate_events import (
    GeneIndex,
    read_boundaries,
    read_callable,
    read_events,
)
from coordinates import resolution_to_bp
from gene_sets import (
    derive_transition_modules,
    load_curated_sets,
    load_expression,
    load_gene_table,
)

HERE = Path(__file__).resolve().parent
EVENTS_DIR = HERE / "results" / "events"
OUT_DIR = HERE / "results" / "enrichment"

# gene sets larger than this are summarised by a count instead of a symbol list
HIT_LIST_LIMIT = 500

RESULT_FIELDS = [
    "resolution", "tolerance", "feature_set", "transition", "n_windows", "window_bins",
    "gene_set", "set_size", "observed", "null_mean", "null_sd", "fold", "z",
    "p_greater", "q_greater", "matched", "hit_genes",
]


def build_feature_sets(events, boundaries, resolution_bp, max_gap_bins=1, flank_bins=0):
    """``{(name, transition): [(chrom, start_bin, width_bins), ...]}``.

    A boundary window is the single bin carrying it, widened by ``flank_bins`` on each
    side. A domain window is the whole domain, and is never widened. Splits and merges are
    also offered on their own, restricted to the events whose two domains are adjacent or
    nearly so, which are the ones where a single boundary really did appear or disappear
    rather than a domain being replaced by two distant pieces.

    ``flank_bins`` is positional tolerance, and exists because these calls come from sparse
    single-cell contact maps: a boundary recovered at bin 10 in one median and bin 13 in
    the next is plausibly the same boundary seen through noise. Widening the window lets a
    feature a few bins away still be counted. The null windows are widened identically, so
    the comparison stays fair -- but a wider window is also a blunter one, and every fold
    is pulled toward 1 as the flank grows, so read the profile in ``boundary_profile.py``
    alongside any flanked fold.
    """
    features = {}

    def add(name, transition, chrom, start_bp, end_bp, flank=0):
        start_bin = int(start_bp) // resolution_bp + 1
        width = max(1, (int(end_bp) - int(start_bp)) // resolution_bp)
        if flank:
            start_bin = max(1, start_bin - flank)
            width += 2 * flank
        features.setdefault((name, transition), []).append((chrom, start_bin, width))

    for row in boundaries:
        transition = row["transition"]
        chrom = row["chrom"]
        start, end = int(row["start"]), int(row["end"])
        change = row["change"]
        add(f"boundary_{change}", transition, chrom, start, end, flank_bins)
        if row["event_type"] in ("split", "merge"):
            gap = row["gap_bins"]
            if gap != "" and int(gap) <= max_gap_bins:
                add(f"{row['event_type']}_boundary", transition, chrom, start, end,
                    flank_bins)

    for row in events:
        transition = row["transition"]
        chrom = row["chrom"]
        start, end = int(row["region_start"]), int(row["region_end"])
        add(f"domain_{row['event_type']}", transition, chrom, start, end)

    return features


def clip_windows_to_pool(features, pool):
    """Trim every window to the bins its chromosome actually has, and drop the homeless.

    ``build_feature_sets`` widens a boundary window by ``flank_bins`` without knowing where
    the chromosome ends, so a boundary within a flank of the last callable bin comes out
    overhanging. The relocation null only ever places a window that fits entirely, so an
    overhanging observed window would be silently truncated when its features are counted
    and would score against a null that was never truncated. Clipping here makes the two
    sides the same shape: the window keeps the width it can actually occupy, and the null
    is then drawn at that width.
    """
    clipped = {}
    for key, entries in features.items():
        kept = []
        for chrom, start, width in entries:
            bins = pool.get(chrom)
            if bins is None or len(bins) == 0:
                continue
            first, last = int(bins.min()), int(bins.max())
            begin = max(first, int(start))
            end = min(last, int(start) + int(width) - 1)
            if end < begin:
                continue
            kept.append((chrom, begin, end - begin + 1))
        if kept:
            clipped[key] = kept
    return clipped


def callable_bin_pool(callable_spans, resolution_bp):
    """``{chrom: array of bin indices}`` an event could have been placed at."""
    pool = {}
    for chrom, spans in callable_spans.items():
        bins = []
        for start, end in spans:
            first = int(start) // resolution_bp + 1
            last = int(end) // resolution_bp
            bins.extend(range(first, last + 1))
        pool[chrom] = np.array(sorted(set(bins)), dtype=np.int64)
    return pool


def set_bin_counts(index, chrom, n_bins, resolution_bp, symbols):
    """Prefix sums of how many genes of ``symbols`` start in each bin of a chromosome."""
    counts = np.zeros(n_bins + 2, dtype=np.int64)
    positions = index.positions.get(chrom, [])
    names = index.symbols.get(chrom, [])
    for tss, symbol in zip(positions, names):
        if symbol in symbols:
            counts[min(n_bins + 1, tss // resolution_bp + 1)] += 1
    return np.cumsum(counts)


def windows_count(prefix, starts, widths):
    """Genes in each window, from a prefix-sum array; ``starts`` may be a 2-D draw."""
    last = len(prefix) - 1
    ends = np.minimum(starts + widths - 1, last)
    starts = np.clip(starts, 1, last)
    return prefix[ends] - prefix[starts - 1]


def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return pvalues
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def test_configuration(resolution, tolerance, index, gene_sets, permutations=2000,
                       max_gap_bins=1, chroms=None, seed=0, min_windows=5,
                       events_dir=None, flank_bins=0):
    """Every (event window set, gene set) test of one resolution and tolerance."""
    resolution_bp = resolution_to_bp(resolution)
    events = read_events(resolution, tolerance, events_dir, chroms)
    boundaries = read_boundaries(resolution, tolerance, events_dir, chroms)
    if not events:
        raise FileNotFoundError(f"No events for {resolution} tol {tolerance}")

    features = build_feature_sets(events, boundaries, resolution_bp, max_gap_bins,
                                  flank_bins)
    pool = callable_bin_pool(read_callable(resolution, tolerance, events_dir, chroms),
                             resolution_bp)
    features = clip_windows_to_pool(features, pool)
    n_bins = {chrom: int(bins.max()) for chrom, bins in pool.items() if len(bins)}

    # the prefix sums are shared by every test of a gene set, so they are built once
    prefixes = {
        name: {chrom: set_bin_counts(index, chrom, n_bins[chrom], resolution_bp, symbols)
               for chrom in n_bins}
        for name, symbols in gene_sets.items()
    }

    rng = np.random.default_rng(seed)
    results = []
    for (feature_name, transition), windows in sorted(features.items()):
        if len(windows) < min_windows:
            continue
        by_chrom = {}
        for chrom, start_bin, width in windows:
            if chrom not in pool or len(pool[chrom]) == 0:
                continue
            by_chrom.setdefault(chrom, []).append((start_bin, width))
        if not by_chrom:
            continue

        # One set of relocations, reused for every gene set of this window set. A window
        # is only ever placed where the whole of it still fits inside the callable region:
        # letting a wide window hang off the end would count the genes of a truncated
        # window against an untruncated observation and manufacture enrichment.
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
                    # a window as wide as the callable region itself can only sit where it
                    # already is, so it carries no information and is pinned
                    room = bins[:1]
                choices[:, columns] = rng.choice(
                    room, size=(permutations, len(columns))
                )
            draws[chrom] = (starts, widths, choices)

        mean_width = float(np.mean([w for _, _, w in windows]))
        for set_name, symbols in sorted(gene_sets.items()):
            if not symbols:
                continue
            observed = 0
            null_total = np.zeros(permutations, dtype=np.int64)
            hits = set()
            for chrom, (starts, widths, choices) in draws.items():
                prefix = prefixes[set_name][chrom]
                observed += int(windows_count(prefix, starts, widths).sum())
                null_total += windows_count(prefix, choices, widths).sum(axis=1)
                for start, width in zip(starts, widths):
                    lo = (start - 1) * resolution_bp
                    hi = (start + width - 1) * resolution_bp
                    hits.update(g for g in index.in_window(chrom, lo, hi) if g in symbols)

            null_mean = float(null_total.mean())
            null_sd = float(null_total.std(ddof=1)) if permutations > 1 else 0.0
            # add-one so that a p-value is never reported as exactly zero
            p_greater = float((np.sum(null_total >= observed) + 1) / (permutations + 1))
            results.append({
                "resolution": resolution, "tolerance": tolerance,
                "feature_set": feature_name, "transition": transition,
                "n_windows": len(windows), "window_bins": round(mean_width, 2),
                "gene_set": set_name, "set_size": len(symbols),
                "observed": observed, "null_mean": round(null_mean, 2),
                "null_sd": round(null_sd, 2),
                "fold": round(observed / null_mean, 3) if null_mean > 0 else "",
                "z": round((observed - null_mean) / null_sd, 2) if null_sd > 0 else "",
                "p_greater": p_greater,
                "matched": matched_flag(set_name, transition),
                # only the small curated sets list their hits; the derived modules and
                # all_genes run to thousands of symbols and are summarised by the counts
                "hit_genes": (",".join(sorted(hits)) if len(symbols) <= HIT_LIST_LIMIT
                              else f"{len(hits)} genes"),
            })

    qvalues = benjamini_hochberg([row["p_greater"] for row in results])
    for row, q in zip(results, qvalues):
        row["q_greater"] = float(q)
        row["p_greater"] = float(row["p_greater"])
    return results


def matched_flag(set_name, transition):
    """Whether a derived gene set belongs to the transition the windows came from."""
    if "@" not in set_name:
        return ""
    return "yes" if set_name.split("@", 1)[1] == transition else "no"


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["p_greater"]):
            writer.writerow(row)


def report(rows, limit=25, alpha=0.1):
    significant = [r for r in rows if r["q_greater"] <= alpha]
    print(f"{len(significant)} of {len(rows)} tests at q <= {alpha}")
    if not significant:
        return
    print(f"{'feature set':22}{'transition':20}{'gene set':30}"
          f"{'obs':>6}{'null':>8}{'fold':>7}{'q':>9}  match")
    for row in sorted(significant, key=lambda r: r["p_greater"])[:limit]:
        print(f"{row['feature_set']:22}{row['transition']:20}{row['gene_set']:30}"
              f"{row['observed']:>6}{row['null_mean']:>8}{str(row['fold']):>7}"
              f"{row['q_greater']:>9.4f}  {row['matched']}")


def load_all_gene_sets(genes, threshold=1.0):
    """The curated sets, the derived transition modules, and every gene as a density read.

    ``all_genes`` is not a hypothesis. It reports how gene dense a window set is compared
    with the rest of its chromosome, which is the number every other row of the table has
    to be read against: a window set that is gene poor overall will look depleted for
    every gene set, and one that is gene dense will look enriched for every gene set.
    """
    sets = load_curated_sets(genes=genes)
    expression, _ = load_expression()
    sets.update(derive_transition_modules(expression, threshold=threshold))
    sets["all_genes"] = set(genes)
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default="500kb")
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--max-gap-bins", type=int, default=1,
                        help="how far apart the two domains of a split or merge may be "
                             "for it to count as one boundary change")
    parser.add_argument("--flank-bins", type=int, default=0,
                        help="positional tolerance in bins: widen every boundary window by this many bins on each side, because sparse single-cell calls place the same boundary a bin or two apart between medians")
    parser.add_argument("--min-windows", type=int, default=5,
                        help="skip window sets smaller than this")
    parser.add_argument("--chroms", nargs="*", default=None)
    parser.add_argument("--fold-threshold", type=float, default=1.0,
                        help="log2 fold change for the derived expression modules")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--events-dir", default=None,
                        help="event tables to read; use results/events_unimputed to test the medians recomputed from the unimputed calls")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--all", action="store_true",
                        help="every resolution and tolerance present")
    args = parser.parse_args()

    genes = load_gene_table()
    index = GeneIndex(genes)
    gene_sets = load_all_gene_sets(genes, threshold=args.fold_threshold)
    out_dir = Path(args.out_dir or OUT_DIR)
    events_dir = Path(args.events_dir) if args.events_dir else EVENTS_DIR

    if not args.all:
        rows = test_configuration(
            args.resolution, args.tolerance, index, gene_sets,
            permutations=args.permutations, max_gap_bins=args.max_gap_bins,
            chroms=args.chroms, seed=args.seed, min_windows=args.min_windows,
            flank_bins=args.flank_bins, events_dir=events_dir,
        )
        path = out_dir / f"{args.resolution}_tol{args.tolerance}_enrichment.tsv"
        write_results(path, rows)
        print(f"wrote {path}")
        report(rows)
        return

    configurations = sorted(
        {(path.parent.parent.name, int(path.name.split("_")[1]))
         for path in events_dir.glob("*/*/tol_*_events.tsv")}
    )
    combined = []
    for resolution, tolerance in configurations:
        rows = test_configuration(
            resolution, tolerance, index, gene_sets, permutations=args.permutations,
            max_gap_bins=args.max_gap_bins, chroms=args.chroms, seed=args.seed,
            min_windows=args.min_windows, flank_bins=args.flank_bins,
            events_dir=events_dir,
        )
        write_results(out_dir / f"{resolution}_tol{tolerance}_enrichment.tsv", rows)
        combined.extend(rows)
        significant = sum(1 for r in rows if r["q_greater"] <= 0.1)
        print(f"{resolution} tol {tolerance}: {len(rows)} tests, {significant} at q <= 0.1")
    write_results(out_dir / "all_configurations_enrichment.tsv", combined)
    print(f"wrote {out_dir / 'all_configurations_enrichment.tsv'}")


if __name__ == "__main__":
    main()
