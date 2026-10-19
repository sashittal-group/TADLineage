#!/usr/bin/env python3
"""Pathway over-representation of the genes in reorganised TADs, one edge at a time.

The foreground is the plain thing: **every gene whose start site falls inside a TAD that
reorganises along the edge**, where reorganised means called at one endpoint of the edge
and not at the other (see ``reorganized_tads.py``).  No event typing, no boundary bins, no
size window on the domains.

The test is the standard one for this question:

    Fisher's exact test, one-sided for over-representation
    universe   every gene inside the callable span that the collection annotates at all
    correction Benjamini-Hochberg across the gene sets of one collection within one edge

Chromosomes are pooled, so each row is a genome-wide statement about one edge.  Set sizes
are cut to the usual 10-500 window after restricting to the universe.

Alongside the statistics each row carries three descriptive columns -- ``top_chrom_frac``,
``locus_frac`` and ``family_frac`` -- saying how concentrated the overlapping genes are on
one chromosome, in one 2 Mb window, and in one gene-family root.  They are there so a hit
driven by a single tandem array (the Hist1 cluster, the keratins) can be recognised on
sight.  **Nothing is filtered on them**; every significant row is written.

Usage:
    python pathway_enrichment.py --caller ontad --resolution 250kb --tolerance 3
    python pathway_enrichment.py --collection mh.all --q-max 0.1
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

HERE = Path(__file__).resolve().parent
SCPTR = HERE.parent.parent / "scptr_robustad" / "merge_split_real_biology"
if str(SCPTR) not in sys.path:
    sys.path.insert(0, str(SCPTR))

from annotate_events import GeneIndex  # noqa: E402
from gene_sets import load_gene_table  # noqa: E402
from reorganized_tads import edges_for  # noqa: E402

GENE_SET_DIR = HERE / "gene_sets"
RESULTS = HERE / "results"

# the usual over-representation size window, applied after cutting each set to the universe
MIN_SET_SIZE = 10
MAX_SET_SIZE = 500
# an edge with barely any reorganisation cannot over-represent anything and only adds tests
MIN_FOREGROUND = 25
# window for the descriptive "is this one tandem array?" column
LOCUS_WINDOW_BP = 2_000_000

# How many 2 Mb windows the "is this just a tandem array?" column is allowed to use.
# One window is not enough: the alpha-defensins sit beside the Bpif and Pglyrp arrays, and
# the interferon-inducible Ifi2xx cluster sits apart from the Gbps, so a set can be three
# blocks of paralogs and still score low on any single window.
CLUSTER_WINDOWS = 3

FIELDS = ["caller", "resolution", "tolerance", "edge", "collection", "gene_set",
          "set_size", "overlap", "n_foreground", "n_universe", "expected", "fold",
          "overlap_ratio", "p", "q", "top_chrom", "top_chrom_frac",
          "locus_frac", "locus_at", "cluster_frac", "cluster_at",
          "family_frac", "family", "genes"]


def resolution_bp(label: str) -> int:
    label = label.lower()
    if label.endswith("kb"):
        return int(float(label[:-2]) * 1_000)
    if label.endswith("mb"):
        return int(float(label[:-2]) * 1_000_000)
    return int(label)


def load_collection(path: Path) -> dict[str, list[str]]:
    sets = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                sets[parts[0]] = parts[2:]
    return sets


def collections_available() -> dict[str, Path]:
    return {p.name.split(".v")[0]: p for p in sorted(GENE_SET_DIR.glob("*.symbols.gmt"))}


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def bin_window(start_bin: int, end_bin: int, res_bp: int) -> tuple[int, int]:
    """Bins are 1-based and inclusive, so bin b covers ``[(b-1)*res, b*res)``."""
    return (start_bin - 1) * res_bp, end_bin * res_bp


def collect(caller: str, resolution: str, tolerance: int, max_domain_bp: int):
    """``({edge: spans}, callable spans)`` in base pairs, pooled over chromosomes."""
    res_bp = resolution_bp(resolution)
    root = RESULTS / "reorg" / caller / resolution / f"tol_{tolerance}"
    changed: dict[str, set] = defaultdict(set)
    if not root.exists():
        return changed, []

    for path in sorted(root.glob("chr*.tsv")):
        for row in read_rows(path):
            start, end = bin_window(int(row["start_bin"]), int(row["end_bin"]), res_bp)
            if max_domain_bp and end - start > max_domain_bp:
                continue
            changed[row["edge"]].add((row["chrom"], start, end))

    covered = [(r["chrom"],) + bin_window(int(r["start_bin"]), int(r["end_bin"]), res_bp)
               for r in read_rows(root / "covered.tsv")]
    return changed, covered


def locus_concentration(positions):
    """Largest share of these start sites inside any one ``LOCUS_WINDOW_BP`` window."""
    total = len(positions)
    if not total:
        return 0.0, ""
    by_chrom = defaultdict(list)
    for chrom, position in positions:
        by_chrom[chrom].append(position)
    best, where = 0.0, ""
    for chrom, values in by_chrom.items():
        values.sort()
        j = 0
        for i, left in enumerate(values):
            while j < len(values) and values[j] - left <= LOCUS_WINDOW_BP:
                j += 1
            if (j - i) / total > best:
                best, where = (j - i) / total, f"{chrom}:{left // 1_000_000}Mb"
    return best, where


def cluster_concentration(positions):
    """Share of these start sites covered by the ``CLUSTER_WINDOWS`` densest 2 Mb windows.

    Chosen greedily: take the window holding the most genes, drop those genes, repeat.  This
    is ``locus_concentration`` generalised past a single block, and it is what separates a
    pathway whose genes are scattered over the genome from one that is two or three tandem
    arrays wearing a pathway name.
    """
    total = len(positions)
    if not total:
        return 0.0, ""
    remaining = list(positions)
    taken, where = 0, []
    for _ in range(CLUSTER_WINDOWS):
        if not remaining:
            break
        by_chrom = defaultdict(list)
        for chrom, position in remaining:
            by_chrom[chrom].append(position)
        best_count, best_span = 0, None
        for chrom, values in by_chrom.items():
            values.sort()
            j = 0
            for i, left in enumerate(values):
                while j < len(values) and values[j] - left <= LOCUS_WINDOW_BP:
                    j += 1
                if j - i > best_count:
                    best_count, best_span = j - i, (chrom, left, left + LOCUS_WINDOW_BP)
        if best_span is None or best_count < 2:
            break
        chrom, lo, hi = best_span
        remaining = [(c, p) for c, p in remaining
                     if not (c == chrom and lo <= p <= hi)]
        taken += best_count
        where.append(f"{chrom}:{lo // 1_000_000}Mb")
    return taken / total, "+".join(where)


def family_concentration(symbols):
    """Largest share of these symbols sharing one gene-family root, and that root.

    The root is the leading run of letters, so Krt10/Krt2a collapse to ``Krt`` and H2ac10
    to ``H``.  Catches a paralog family split across two clusters, which one window cannot.
    """
    if not symbols:
        return 0.0, ""
    roots = Counter()
    for symbol in symbols:
        match = re.match(r"^([A-Za-z]+)", symbol)
        if match:
            roots[match.group(1)] += 1
    if not roots:
        return 0.0, ""
    root, count = roots.most_common(1)[0]
    return count / len(symbols), root


def genes_in(index: GeneIndex, spans) -> set[str]:
    symbols = set()
    for chrom, start, end in spans:
        symbols.update(index.in_window(chrom, start, end))
    return symbols


def benjamini_hochberg(p):
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
    """Fisher's exact test of every set against one foreground, vectorised."""
    flag = np.zeros(len(position), bool)
    for symbol in foreground:
        i = position.get(symbol)
        if i is not None:
            flag[i] = True
    n_universe = len(position)
    n_foreground = int(flag.sum())
    sizes = membership.sum(axis=1)
    # cast off bool first: numpy's matmul on boolean arrays is logical, not a count
    overlap = membership.astype(np.int32) @ flag.astype(np.int32)
    # P(X >= overlap) for X hypergeometric is the one-sided Fisher p for over-representation
    p = hypergeom.sf(overlap - 1, n_universe, sizes, n_foreground)
    expected = sizes * n_foreground / n_universe
    with np.errstate(divide="ignore", invalid="ignore"):
        fold = np.where(expected > 0, overlap / expected, np.nan)
    return {"names": names, "sizes": sizes, "overlap": overlap, "expected": expected,
            "fold": fold, "p": p, "q": benjamini_hochberg(p),
            "n_foreground": n_foreground, "n_universe": n_universe, "flag": flag}


def run(args) -> int:
    available = collections_available()
    if not available:
        print("No GMT files. Run fetch_gene_sets.py first.", file=sys.stderr)
        return 1
    wanted = [c for c in available if not args.collection or c in args.collection]

    genes = load_gene_table()
    index = GeneIndex(genes)
    chrom_of = {s: r["chrom"] for s, r in genes.items()}
    tss_of = {s: r["tss"] for s, r in genes.items()}

    loaded = {c: load_collection(available[c]) for c in wanted}
    annotated = {c: {s for symbols in sets.values() for s in symbols}
                 for c, sets in loaded.items()}

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS / "enrichment.tsv"
    handle = out_path.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
    writer.writeheader()

    n_rows = 0
    for caller in args.caller:
        for resolution in args.resolution:
            for tolerance in args.tolerance:
                changed, spans = collect(caller, resolution, tolerance, args.max_domain_bp)
                if not changed:
                    print(f"{caller} {resolution} tol={tolerance}: no calls, skipped",
                          file=sys.stderr)
                    continue
                universe_all = genes_in(index, spans)

                print(f"\n=== {caller} {resolution} tol={tolerance} ===")
                print(f"{'edge':<12}{'foreground':>11}{'universe':>10}{'fraction':>10}")
                for edge in (f"{p}->{c}" for p, c in edges_for(caller)):
                    fg = genes_in(index, changed.get(edge, ()))
                    frac = len(fg) / len(universe_all) if universe_all else 0.0
                    print(f"{edge:<12}{len(fg):>11}{len(universe_all):>10}{frac:>10.1%}")

                for collection in wanted:
                    universe = universe_all & annotated[collection]
                    names, membership, position = build_membership(
                        loaded[collection], universe)
                    if not names:
                        continue
                    symbols = sorted(position, key=position.get)
                    for edge in (f"{p}->{c}" for p, c in edges_for(caller)):
                        foreground = genes_in(index, changed.get(edge, ())) & universe
                        if len(foreground) < MIN_FOREGROUND:
                            continue
                        stats = enrich(names, membership, position, foreground)
                        for i in np.where(stats["q"] <= args.q_max)[0]:
                            hit = [symbols[j]
                                   for j in np.where(membership[i] & stats["flag"])[0]]
                            chroms = [chrom_of.get(g) for g in hit]
                            top_chrom, top_frac = "", 0.0
                            if chroms:
                                top_chrom = max(set(chroms), key=chroms.count)
                                top_frac = chroms.count(top_chrom) / len(chroms)
                            placed = [(chrom_of[g], tss_of[g])
                                      for g in hit if g in tss_of]
                            locus_frac, locus_at = locus_concentration(placed)
                            cluster_frac, cluster_at = cluster_concentration(placed)
                            family_frac, family = family_concentration(hit)
                            writer.writerow({
                                "caller": caller, "resolution": resolution,
                                "tolerance": tolerance, "edge": edge,
                                "collection": collection, "gene_set": stats["names"][i],
                                "set_size": int(stats["sizes"][i]),
                                "overlap": int(stats["overlap"][i]),
                                "n_foreground": stats["n_foreground"],
                                "n_universe": stats["n_universe"],
                                "expected": round(float(stats["expected"][i]), 2),
                                "fold": round(float(stats["fold"][i]), 3),
                                "overlap_ratio": round(
                                    float(stats["overlap"][i]) / float(stats["sizes"][i]), 4),
                                "p": f"{stats['p'][i]:.3e}", "q": f"{stats['q'][i]:.3e}",
                                "top_chrom": top_chrom, "top_chrom_frac": round(top_frac, 3),
                                "locus_frac": round(locus_frac, 3), "locus_at": locus_at,
                                "cluster_frac": round(cluster_frac, 3),
                                "cluster_at": cluster_at,
                                "family_frac": round(family_frac, 3), "family": family,
                                "genes": ";".join(hit),
                            })
                            n_rows += 1

    handle.close()
    print(f"\nWrote {out_path}  ({n_rows} rows at q <= {args.q_max})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caller", action="append", choices=["ontad", "robustad"])
    ap.add_argument("--resolution", action="append")
    ap.add_argument("--tolerance", action="append", type=int)
    ap.add_argument("--collection", action="append")
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--max-domain-bp", type=int, default=0,
                    help="drop reorganised domains longer than this; 0 keeps all")
    ap.add_argument("--out")
    args = ap.parse_args()
    args.caller = args.caller or ["ontad"]
    args.resolution = args.resolution or ["250kb"]
    args.tolerance = args.tolerance or [3]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
