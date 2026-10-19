#!/usr/bin/env python3
"""Pathway over-representation of the genes at reorganised TAD boundaries, per transition.

For one clustering configuration -- a caller, a bin size and a tolerance -- the median TAD
set of each embryonic stage is compared with the next one along the chain

    1-cell -> 2-cell -> 4-cell -> 8-cell -> 64-cell

and every boundary is labelled by what happened to it across that transition: it was
gained, it was lost, or it stayed.  This script asks, one transition at a time, whether the
genes sitting on the boundaries that changed are drawn disproportionately from any pathway.

The test is Fisher's exact test, one-sided for over-representation, with

    foreground  genes whose transcription start site falls in a bin carrying a gained or
                lost boundary in this transition,
    universe    genes whose start site falls anywhere in the callable span of this
                configuration and that the collection annotates at all,

and Benjamini-Hochberg applied across the gene sets of one collection within one run.  A
gene is assigned to a bin by its start site, which is the same rule ``annotate_events.py``
and the boundary analyses of ``merge_split_real_biology`` use, so the gene lists here and
there mean the same thing.

Runs are reported one per row.  Nothing is pooled across callers or bin sizes and no
pathway is asked to replicate: a configuration stands on its own.

Usage:
    python pathway_enrichment.py                         # every configuration
    python pathway_enrichment.py --collection mh.all     # one collection
    python pathway_enrichment.py --caller RobusTAD --resolution 500kb
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "merge_split_real_biology"
if str(SIBLING) not in sys.path:
    sys.path.insert(0, str(SIBLING))

from annotate_events import GeneIndex  # noqa: E402
from gene_sets import STAGE_ORDER, load_gene_table  # noqa: E402

GENE_SET_DIR = HERE / "gene_sets"
RESULTS = HERE / "results"

EVENT_DIRS = {
    "RobusTAD": SIBLING / "results" / "events",
    "OnTAD": SIBLING / "results" / "events_ontad",
}
TRANSITIONS = [f"{a}->{b}" for a, b in zip(STAGE_ORDER, STAGE_ORDER[1:])]
CHANGED = ("gained", "lost")
CHANGED_EVENTS = ("domain_gain", "domain_loss", "split", "merge")

# the usual over-representation size window, applied after cutting each set to the universe
MIN_SET_SIZE = 10
MAX_SET_SIZE = 500
# a reorganised domain is taken up to this span.  The event tables are hierarchical, so the
# outermost nested call on a chromosome runs to 180 Mb; a region that size contains every
# gene on the chromosome and so says nothing about where anything sits.  Two megabases is
# both the median reorganised domain here and the usual size of a mouse TAD, and it keeps
# the domain foreground near the boundary foreground -- about 16% of the annotated genes
# against 8% -- so that an overlap ratio means the same thing in either readout.  At a
# 10 Mb cap the domain foreground is 59% of the genome and every overlap ratio is inflated.
MAX_DOMAIN_BP = 2_000_000
# a run is skipped if this transition barely changed anything, since a handful of genes
# cannot over-represent anything and only adds tests to the correction
MIN_FOREGROUND = 25

FIELDS = [
    "feature",
    "caller", "resolution", "tolerance", "transition", "collection", "gene_set",
    "set_size", "overlap", "n_foreground", "n_universe", "expected", "fold",
    "overlap_ratio", "p", "q", "genes",
]


def load_collection(path):
    """``{set name: [symbols]}`` from a GMT file."""
    sets = {}
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sets[parts[0]] = parts[2:]
    return sets


def collections_available():
    """``{short name: path}`` for the GMTs that ``fetch_gene_sets.py`` has downloaded."""
    found = {}
    for path in sorted(GENE_SET_DIR.glob("*.symbols.gmt")):
        found[path.name.split(".v")[0]] = path
    return found


def read_rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def configuration_boundaries(events_dir, resolution, tolerance):
    """``({transition: {(chrom, start, end)}}, callable spans)`` for one configuration.

    The changed boundaries are pooled over chromosomes, which is what makes a run a
    genome-wide gene list rather than twenty small ones.
    """
    changed = defaultdict(set)
    callable_spans = []
    root = Path(events_dir) / str(resolution)
    if not root.exists():
        return changed, callable_spans
    for path in sorted(root.glob(f"*/tol_{tolerance}_boundaries.tsv")):
        chrom = path.parent.name
        for row in read_rows(path):
            if row["change"] in CHANGED:
                changed[row["transition"]].add(
                    (chrom, int(row["start"]), int(row["end"])))
        span_path = path.with_name(f"tol_{tolerance}_callable.tsv")
        if span_path.exists():
            for row in read_rows(span_path):
                callable_spans.append((chrom, int(row["start"]), int(row["end"])))
    return changed, callable_spans


def configuration_domains(events_dir, resolution, tolerance):
    """``{transition: {(chrom, start, end)}}`` for the bodies of the reorganised domains.

    A boundary is one bin wide, so the boundary foreground is the genes sitting right on
    the structural change.  The domain body is the whole region the change happened to,
    which is the set of genes that changed neighbourhood rather than the ones at the seam.
    """
    changed = defaultdict(set)
    root = Path(events_dir) / str(resolution)
    if not root.exists():
        return changed
    for path in sorted(root.glob(f"*/tol_{tolerance}_events.tsv")):
        chrom = path.parent.name
        for row in read_rows(path):
            if row["event_type"] not in CHANGED_EVENTS:
                continue
            start, end = int(row["region_start"]), int(row["region_end"])
            if end - start > MAX_DOMAIN_BP:
                continue
            changed[row["transition"]].add((chrom, start, end))
    return changed


def genes_in(index, spans):
    """Every gene symbol whose start site falls in one of ``spans``."""
    symbols = set()
    for chrom, start, end in spans:
        symbols.update(index.in_window(chrom, start, end))
    return symbols


def benjamini_hochberg(p):
    """BH q-values for a vector of p-values, order preserved."""
    p = np.asarray(p, float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def build_membership(sets, universe):
    """``(names, boolean matrix sets x universe, sizes)`` for the sets that stay in range."""
    position = {symbol: i for i, symbol in enumerate(sorted(universe))}
    names, rows = [], []
    for name, symbols in sets.items():
        idx = [position[s] for s in symbols if s in position]
        if not (MIN_SET_SIZE <= len(idx) <= MAX_SET_SIZE):
            continue
        row = np.zeros(len(position), bool)
        row[idx] = True
        names.append(name)
        rows.append(row)
    if not rows:
        return [], np.zeros((0, len(position)), bool), position
    return names, np.vstack(rows), position


def enrich(names, membership, position, foreground):
    """Fisher's exact test of each set against one foreground, as arrays."""
    flag = np.zeros(len(position), bool)
    for symbol in foreground:
        i = position.get(symbol)
        if i is not None:
            flag[i] = True
    n_universe = len(position)
    n_foreground = int(flag.sum())
    sizes = membership.sum(axis=1)
    # cast off bool: numpy's matmul on boolean arrays is logical, not a count
    overlap = membership.astype(np.int32) @ flag.astype(np.int32)
    # P(X >= overlap) for X hypergeometric: the one-sided Fisher p for over-representation
    p = hypergeom.sf(overlap - 1, n_universe, sizes, n_foreground)
    expected = sizes * n_foreground / n_universe
    with np.errstate(divide="ignore", invalid="ignore"):
        fold = np.where(expected > 0, overlap / expected, np.nan)
    return {
        "names": names, "sizes": sizes, "overlap": overlap, "expected": expected,
        "fold": fold, "p": p, "q": benjamini_hochberg(p),
        "n_foreground": n_foreground, "n_universe": n_universe, "flag": flag,
    }


def run(args):
    available = collections_available()
    if not available:
        print("No GMT files. Run fetch_gene_sets.py first.", file=sys.stderr)
        return 1
    wanted = [c for c in available if not args.collection or c in args.collection]

    genes = load_gene_table()
    index = GeneIndex(genes)

    callers = [c for c in EVENT_DIRS if not args.caller or c in args.caller]
    resolutions = args.resolution or ["250kb", "500kb", "1000kb"]
    tolerances = args.tolerance or list(range(6))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS / f"enrichment_{args.feature}.tsv"
    handle = open(out_path, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
    writer.writeheader()

    loaded = {c: load_collection(available[c]) for c in wanted}
    annotated = {c: {s for symbols in sets.values() for s in symbols}
                 for c, sets in loaded.items()}

    n_rows = 0
    for caller in callers:
        for resolution in resolutions:
            for tolerance in tolerances:
                boundaries, spans = configuration_boundaries(
                    EVENT_DIRS[caller], resolution, tolerance)
                if not boundaries:
                    continue
                if args.feature == "boundary":
                    changed = boundaries
                else:
                    domains = configuration_domains(
                        EVENT_DIRS[caller], resolution, tolerance)
                    if args.feature == "domain":
                        changed = domains
                    else:
                        changed = {t: boundaries.get(t, set()) | domains.get(t, set())
                                   for t in TRANSITIONS}
                universe_all = genes_in(index, spans)
                # the membership matrix depends only on the universe, so it is built once
                # per configuration and reused across the four transitions
                built = {}
                for collection in wanted:
                    universe = universe_all & annotated[collection]
                    names, membership, position = build_membership(
                        loaded[collection], universe)
                    if names:
                        built[collection] = (names, membership, position, universe)
                for transition in TRANSITIONS:
                    foreground_all = genes_in(index, changed.get(transition, ()))
                    foreground_all &= universe_all
                    if len(foreground_all) < MIN_FOREGROUND:
                        continue
                    for collection, (names, membership, position, universe) in built.items():
                        res = enrich(names, membership, position,
                                     foreground_all & universe)
                        keep = np.nonzero(res["q"] <= args.q_max)[0]
                        by_index = sorted(position.items(), key=lambda kv: kv[1])
                        for i in keep:
                            members = membership[i] & res["flag"]
                            symbols = [sym for sym, j in by_index if members[j]]
                            writer.writerow({
                                "feature": args.feature,
                                "caller": caller, "resolution": resolution,
                                "tolerance": tolerance, "transition": transition,
                                "collection": collection, "gene_set": names[i],
                                "set_size": int(res["sizes"][i]),
                                "overlap": int(res["overlap"][i]),
                                "n_foreground": res["n_foreground"],
                                "n_universe": res["n_universe"],
                                "expected": round(float(res["expected"][i]), 2),
                                "fold": round(float(res["fold"][i]), 3),
                                "overlap_ratio": round(
                                    float(res["overlap"][i] / res["sizes"][i]), 4),
                                "p": f"{res['p'][i]:.3e}",
                                "q": f"{res['q'][i]:.3e}",
                                "genes": ";".join(symbols),
                            })
                            n_rows += 1
                    print(f"{caller} {resolution} tol{tolerance} {transition}: "
                          f"{len(foreground_all)} foreground genes", flush=True)
    handle.close()
    print(f"\n{n_rows} rows at q <= {args.q_max} -> {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caller", nargs="*")
    parser.add_argument("--resolution", nargs="*")
    parser.add_argument("--tolerance", nargs="*", type=int)
    parser.add_argument("--collection", nargs="*")
    parser.add_argument("--feature", default="boundary",
                        choices=("boundary", "domain", "both"),
                        help="genes on the changed boundaries, inside the reorganised "
                             "domains, or either")
    parser.add_argument("--out", help="output table (default results/enrichment_<feature>.tsv)")
    parser.add_argument("--q-max", type=float, default=0.25,
                        help="only write rows at or below this q-value")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
