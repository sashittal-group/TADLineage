#!/usr/bin/env python3
"""Which genes sit on the domains that rearranged, and on the boundaries that moved.

An event is annotated at two scales, because they answer different questions.

Domain scale
    The genes inside the domain that split, merged, appeared or disappeared. This is the
    set whose regulatory neighbourhood the rearrangement changed.

Boundary scale
    The genes inside the focal window, the bin or bins where insulation appeared (a split)
    or was removed (a merge). This is the sharper of the two and the one the enrichment
    tests are run on, but at a 250 kb to 1 Mb bin it is still a wide window, so the genes
    it names are candidates, not identified targets.

A gene is counted as being in a window when its transcription start site falls inside it.
Anchoring on the start site rather than on any overlap keeps a handful of very long genes
from being counted into every window they run through, which would bias every test towards
gene deserts and long genes.

One configuration is a resolution and a tolerance, pooled over all chromosomes, since that
is the unit the enrichment and the figures use.

Usage:
    python annotate_events.py --resolution 500kb --tolerance 1
    python annotate_events.py --all
"""

from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path

from gene_sets import genes_by_chromosome, load_gene_table

HERE = Path(__file__).resolve().parent
EVENTS_DIR = HERE / "results" / "events"
OUT_DIR = HERE / "results" / "annotated"

ANNOTATED_FIELDS = [
    "resolution", "chrom", "tolerance", "transition", "event_type",
    "region_start", "region_end", "focal_start", "focal_end",
    "n_region_genes", "region_genes",
]
PAIR_FIELDS = [
    "resolution", "chrom", "tolerance", "transition", "event_type",
    "scale", "change", "start", "end", "gene", "tss",
]


class GeneIndex:
    """Genes on each chromosome, sorted by transcription start site."""

    def __init__(self, genes):
        self.tss = {symbol: record["tss"] for symbol, record in genes.items()}
        buckets = genes_by_chromosome(genes)
        self.positions = {chrom: [tss for tss, _, _ in rows]
                          for chrom, rows in buckets.items()}
        self.symbols = {chrom: [symbol for _, symbol, _ in rows]
                        for chrom, rows in buckets.items()}

    def in_window(self, chrom, start, end):
        """Symbols whose start site falls in ``[start, end)``, in position order."""
        positions = self.positions.get(chrom)
        if not positions or end <= start:
            return []
        left = bisect.bisect_left(positions, int(start))
        right = bisect.bisect_left(positions, int(end))
        return self.symbols[chrom][left:right]

    def tss_of(self, symbol):
        return self.tss.get(symbol, "")

    def count_in_window(self, chrom, start, end):
        positions = self.positions.get(chrom)
        if not positions or end <= start:
            return 0
        return (bisect.bisect_left(positions, int(end))
                - bisect.bisect_left(positions, int(start)))


def read_events(resolution, tolerance, events_dir=None, chroms=None):
    """Every event row of one configuration, pooled over chromosomes."""
    root = Path(events_dir or EVENTS_DIR) / str(resolution)
    if not root.exists():
        raise FileNotFoundError(f"No events under {root}. Run call_median_events.py first.")
    rows = []
    for path in sorted(root.glob(f"*/tol_{tolerance}_events.tsv")):
        if chroms and path.parent.name not in chroms:
            continue
        with open(path) as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    return rows


def read_boundaries(resolution, tolerance, events_dir=None, chroms=None):
    """Every boundary row of one configuration, pooled over chromosomes."""
    root = Path(events_dir or EVENTS_DIR) / str(resolution)
    rows = []
    for path in sorted(root.glob(f"*/tol_{tolerance}_boundaries.tsv")):
        if chroms and path.parent.name not in chroms:
            continue
        with open(path) as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    return rows


def read_callable(resolution, tolerance, events_dir=None, chroms=None):
    """``{chrom: [(start, end), ...]}``: where an event could have been called."""
    root = Path(events_dir or EVENTS_DIR) / str(resolution)
    spans = {}
    for path in sorted(root.glob(f"*/tol_{tolerance}_callable.tsv")):
        if chroms and path.parent.name not in chroms:
            continue
        with open(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                spans.setdefault(row["chrom"], []).append(
                    (int(row["start"]), int(row["end"]))
                )
    return spans


def annotate(rows, index, flank_bp=0, max_domain_bp=20_000_000):
    """Add the gene lists to each event row and return them with the long-form pairs.

    Domains wider than ``max_domain_bp`` get a gene count but no gene list. The medians
    contain domains spanning most of a chromosome, and naming the eight hundred genes
    inside one of those says nothing about any of them: it is not a neighbourhood a gene
    can be said to belong to, and listing it swamps the output with rows no analysis uses.
    The boundary scale is never capped, since a boundary window is one bin.
    """
    annotated, pairs = [], []
    for row in rows:
        chrom = row["chrom"]
        region = (int(row["region_start"]), int(row["region_end"]))
        wide = (region[1] - region[0]) > max_domain_bp
        region_genes = ([] if wide else index.in_window(chrom, *region))
        region_count = (index.count_in_window(chrom, *region) if wide
                        else len(region_genes))

        # the focal span of a split or a merge runs from one daughter domain to the other
        # and can be tens of megabases wide, so it is reported but never gene listed; the
        # boundary scale below is built from the single-bin boundary table instead
        focal = (row.get("focal_start", ""), row.get("focal_end", ""))

        annotated.append({
            **{key: row[key] for key in
               ("resolution", "chrom", "tolerance", "transition", "event_type")},
            "region_start": region[0], "region_end": region[1],
            "focal_start": focal[0], "focal_end": focal[1],
            "n_region_genes": region_count,
            "region_genes": ",".join(region_genes) if not wide else "",
        })

        base = {key: row[key] for key in
                ("resolution", "chrom", "tolerance", "transition", "event_type")}
        for gene in region_genes:
            pairs.append({**base, "scale": "domain", "change": "", "start": region[0],
                          "end": region[1], "gene": gene, "tss": index.tss_of(gene)})
    return annotated, pairs


def annotate_boundaries(rows, index, flank_bp=0):
    """Genes in each single-bin boundary window, the scale the tests are anchored on."""
    pairs = []
    for row in rows:
        start = int(row["start"]) - flank_bp
        end = int(row["end"]) + flank_bp
        for gene in index.in_window(row["chrom"], start, end):
            pairs.append({
                "resolution": row["resolution"], "chrom": row["chrom"],
                "tolerance": row["tolerance"], "transition": row["transition"],
                "event_type": row["event_type"], "scale": "boundary",
                "change": row["change"], "start": start, "end": end,
                "gene": gene, "tss": index.tss_of(gene),
            })
    return pairs


def write_table(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def annotate_configuration(resolution, tolerance, index, out_dir=None, flank_bp=0,
                           events_dir=None, quiet=False, max_domain_bp=20_000_000):
    rows = read_events(resolution, tolerance, events_dir)
    if not rows:
        raise FileNotFoundError(f"No events for {resolution} tol {tolerance}")
    annotated, pairs = annotate(rows, index, max_domain_bp=max_domain_bp)
    pairs.extend(annotate_boundaries(
        read_boundaries(resolution, tolerance, events_dir), index, flank_bp=flank_bp
    ))

    out_dir = Path(out_dir or OUT_DIR)
    stem = f"{resolution}_tol{tolerance}"
    write_table(out_dir / f"{stem}_events_annotated.tsv", ANNOTATED_FIELDS, annotated)
    write_table(out_dir / f"{stem}_event_genes.tsv", PAIR_FIELDS, pairs)

    if not quiet:
        by_type = {}
        for row in annotated:
            entry = by_type.setdefault(row["event_type"], [0, 0])
            entry[0] += 1
            entry[1] += row["n_region_genes"]
        boundary_pairs = sum(1 for row in pairs if row["scale"] == "boundary")
        print(f"{resolution} tol {tolerance}: {len(annotated)} events, {len(pairs)} "
              f"event-gene pairs ({boundary_pairs} at boundaries)")
        print(f"  {'event type':14}{'events':>8}{'genes/domain':>14}")
        for name in sorted(by_type):
            count, region_total = by_type[name]
            print(f"  {name:14}{count:>8}{region_total / count:>14.1f}")
    return annotated, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default="500kb")
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--flank-bp", type=int, default=0,
                        help="widen the focal boundary window by this much on each side")
    parser.add_argument("--max-domain-mb", type=float, default=20.0,
                        help="domains wider than this are counted but not gene listed")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--all", action="store_true",
                        help="every resolution and tolerance present under results/events")
    args = parser.parse_args()

    index = GeneIndex(load_gene_table())

    if not args.all:
        annotate_configuration(args.resolution, args.tolerance, index,
                               args.out_dir, args.flank_bp,
                               max_domain_bp=int(args.max_domain_mb * 1e6))
        return

    configurations = sorted(
        {(path.parent.parent.name, int(path.name.split("_")[1]))
         for path in EVENTS_DIR.glob("*/*/tol_*_events.tsv")}
    )
    for resolution, tolerance in configurations:
        annotate_configuration(resolution, tolerance, index, args.out_dir,
                               args.flank_bp, quiet=True,
                               max_domain_bp=int(args.max_domain_mb * 1e6))
    print(f"annotated {len(configurations)} configurations into "
          f"{Path(args.out_dir or OUT_DIR)}")


if __name__ == "__main__":
    main()
