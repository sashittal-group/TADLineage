#!/usr/bin/env python3
"""Do the domains a caller reports merging or splitting land on imprinted clusters?

One caller at a time. Cross-caller agreement is dropped as a criterion, on the
argument that two TAD callers carry different positional biases and cannot be asked to place
a boundary in the same 250 kb bin; what replaces it is (a) an honest unit of counting and
(b) replication across bin sizes within the one caller.

**The unit is the cluster, not the gene, and that is the whole point of this script.**
The 125 canonical imprinted genes sit in about a dozen real loci: 20 of them are the
Igf2/H19/Kcnq1 cluster on chr7, 15 are Dlk1-Dio3 on chr12, 7 are Snrpn/Ube3a, 7 are Peg3.
A single merge domain that happens to cover Kcnq1 therefore scores 20 "hits" and the
gene-counting version of this test reports a fold of 6. Collapsing the set by single linkage
at ``--linkage`` base pairs and counting *distinct clusters overlapped* removes that, and it
is the same lesson the chromatin panel learned: matching or counting a clustered annotation
by its elements measures the clustering, not the biology.

The null keeps each window on its own chromosome, keeps its width, and slides it to a
random position inside the callable region of that chromosome -- the stretch the medians of
that run actually cover -- and recounts distinct clusters. This is the relocation null
``enrichment.py`` uses, applied to the collapsed set.

Every configuration is reported separately. The six tolerances of one bin size share their
cells and are not independent replicates; the three bin sizes are much closer to
independent, so read the bin-size column, not the row count.

Usage:
    python merge_split_cluster_enrichment.py --caller RobusTAD
    python merge_split_cluster_enrichment.py --caller OnTAD --gene-sets hox_clusters
"""

from __future__ import annotations

import argparse
import collections
import csv
import zlib
from pathlib import Path

import numpy as np

from annotate_events import read_callable, read_events
from cell_calls import SOURCES
from gene_sets import load_curated_sets, load_gene_table

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "merge_split"

RESOLUTIONS = ["250kb", "500kb", "1000kb"]
TOLERANCES = range(6)
TRANSITIONS = ["1-cell->2-cell", "2-cell->4-cell", "4-cell->8-cell", "8-cell->64-cell"]
# events wider than this are whole-arm artefacts of a sparse median and are dropped, the
# same cut annotate_events.py uses
MAX_WINDOW_BP = 20_000_000
# a configuration needs this many windows before its fold is worth reporting
MIN_WINDOWS = 5
PERMUTATIONS = 5000
LINKAGE_BP = 3_000_000

FIELDS = ["gene_set", "event_class", "resolution", "tolerance", "transition", "n_windows",
          "n_clusters", "observed", "null_mean", "null_sd", "fold", "p_greater"]


def cluster_set(symbols, genes, linkage=LINKAGE_BP):
    """``[(chrom, start, end)]``: the gene set collapsed by single linkage."""
    positions = sorted((genes[s]["chrom"], genes[s]["tss"]) for s in symbols if s in genes)
    if not positions:
        return []
    clusters, current = [], [positions[0]]
    for chrom, tss in positions[1:]:
        if chrom == current[-1][0] and tss - current[-1][1] <= linkage:
            current.append((chrom, tss))
        else:
            clusters.append(current)
            current = [(chrom, tss)]
    clusters.append(current)
    return [(c[0][0], c[0][1], c[-1][1]) for c in clusters]


def windows_of(rows, transition, event_class):
    """``{chrom: [(start, end)]}`` for the domain bodies of one event class."""
    wanted = {"merge": {"merge"}, "split": {"split"},
              "gain": {"domain_gain"}, "loss": {"domain_loss"}}[event_class]
    out = collections.defaultdict(list)
    for row in rows:
        if row["transition"] != transition or row["event_type"] not in wanted:
            continue
        start, end = int(row["region_start"]), int(row["region_end"])
        if end - start > MAX_WINDOW_BP or end <= start:
            continue
        out[row["chrom"]].append((start, end))
    return out


def count_clusters(windows, clusters_by_chrom):
    """How many distinct clusters at least one window overlaps."""
    hit = set()
    for chrom, spans in windows.items():
        for index, (start, end) in clusters_by_chrom.get(chrom, []):
            for window_start, window_end in spans:
                if window_start <= end and start <= window_end:
                    hit.add(index)
                    break
    return len(hit)


def null_counts(windows, callable_spans, clusters_by_chrom, permutations, rng):
    """Distinct clusters hit under ``permutations`` relocations, all draws vectorised.

    A window of width ``w`` starting at ``x`` overlaps a cluster spanning ``[s, e]`` exactly
    when ``s - w <= x <= e``, so one comparison per cluster per draw replaces the interval
    intersection, and the whole null is a few array operations per chromosome.
    """
    hits = np.zeros((permutations, 0), bool)
    columns = []
    for chrom, spans in windows.items():
        pool = callable_spans.get(chrom)
        clusters = clusters_by_chrom.get(chrom)
        if not pool:
            return None
        if not clusters:
            continue
        # the widest callable stretch that can hold each window
        starts = np.empty((permutations, len(spans)), np.int64)
        for column, (start, end) in enumerate(spans):
            width = end - start
            options = [(a, b) for a, b in pool if b - a >= width]
            if not options:
                starts[:, column] = start
                continue
            picked = rng.integers(len(options), size=permutations)
            lows = np.array([options[i][0] for i in picked])
            highs = np.array([options[i][1] - width for i in picked])
            starts[:, column] = lows + (rng.random(permutations)
                                        * (highs - lows + 1)).astype(np.int64)
        widths = np.array([e - s for s, e in spans], np.int64)
        for index, (cluster_start, cluster_end) in clusters:
            inside = (starts >= cluster_start - widths) & (starts <= cluster_end)
            columns.append((index, inside.any(axis=1)))
    if not columns:
        return np.zeros(permutations, np.int64)
    by_cluster = {}
    for index, column in columns:
        by_cluster[index] = column if index not in by_cluster \
            else (by_cluster[index] | column)
    hits = np.column_stack(list(by_cluster.values()))
    return hits.sum(axis=1)


def analyse(gene_set_name, symbols, genes, args):
    clusters = cluster_set(symbols, genes, args.linkage)
    by_chrom = collections.defaultdict(list)
    for index, (chrom, start, end) in enumerate(clusters):
        by_chrom[chrom].append((index, (start, end)))
    by_chrom = {c: [(i, s) for i, s in v] for c, v in by_chrom.items()}

    rows = []
    for resolution in args.resolutions:
        for tolerance in TOLERANCES:
            try:
                events = read_events(resolution, tolerance, events_dir=args.events_dir)
            except FileNotFoundError:
                continue
            if not events:
                continue
            spans = read_callable(resolution, tolerance, events_dir=args.events_dir)
            for event_class in args.classes:
                for transition in TRANSITIONS:
                    windows = windows_of(events, transition, event_class)
                    total = sum(len(v) for v in windows.values())
                    if total < MIN_WINDOWS:
                        continue
                    observed = count_clusters(windows, by_chrom)
                    seed = zlib.crc32(
                        f"{gene_set_name}{resolution}{tolerance}{transition}"
                        f"{event_class}".encode())
                    rng = np.random.default_rng(seed)
                    drawn = null_counts(windows, spans, by_chrom,
                                        args.permutations, rng)
                    if drawn is None:
                        continue
                    null = drawn.astype(float)
                    at_least = int((null >= observed).sum())
                    rows.append({
                        "gene_set": gene_set_name, "event_class": event_class,
                        "resolution": resolution, "tolerance": tolerance,
                        "transition": transition, "n_windows": total,
                        "n_clusters": len(clusters), "observed": observed,
                        "null_mean": round(float(null.mean()), 4),
                        "null_sd": round(float(null.std()), 4),
                        "fold": round(observed / null.mean(), 4) if null.mean() else "",
                        "p_greater": round((at_least + 1) / (len(null) + 1), 5),
                    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gene-sets", nargs="+",
                        default=["imprinted_canonical",
                                 "imprinted_noncanonical_h3k27me3",
                                 "hox_clusters", "pluripotency",
                                 "lineage_first_segregation", "prc16_germline_targets",
                                 "housekeeping_control"])
    parser.add_argument("--classes", nargs="+",
                        default=["merge", "split", "gain", "loss"])
    parser.add_argument("--resolutions", nargs="+", default=RESOLUTIONS)
    parser.add_argument("--caller", default="OnTAD", choices=sorted(SOURCES))
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--linkage", type=int, default=LINKAGE_BP)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    args.events_dir = args.events_dir or str(SOURCES[args.caller]["events"])
    out_dir = Path(args.out_dir or OUT_DIR / args.caller)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"caller: {args.caller}  events: {args.events_dir}")

    genes = load_gene_table()
    curated = load_curated_sets(genes=genes)

    rows = []
    for name in args.gene_sets:
        if name not in curated:
            print(f"  no such gene set: {name}")
            continue
        found = analyse(name, curated[name], genes, args)
        rows += found
        clusters = cluster_set(curated[name], genes, args.linkage)
        print(f"{name}: {len(curated[name])} genes -> {len(clusters)} clusters, "
              f"{len(found)} configurations scored")

    path = out_dir / "cluster_enrichment.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
