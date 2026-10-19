#!/usr/bin/env python3
"""Which merge and split events are called again and again, and what sits on them.

A single run is one clustering of a few dozen cells at one bin size and one boundary
tolerance, so any one event it reports could be noise. An event that is called at the same
place, at the same transition, at three different bin sizes and across several tolerances
is a different proposition. This groups the boundary changes of every configuration by
position and ranks them by how much of the parameter grid recovers them.

Support is counted two ways, and the second is the one that matters. ``configurations``
counts every (bin size, tolerance) pair that called the event, but tolerances of the same
bin size are re-analyses of the same clustering and are far from independent.
``resolutions`` counts how many of the three bin sizes called it, and only an event seen at
more than one bin size is worth reading as a locus rather than as a parameter artefact.

Each consensus event is annotated with the genes in its window and with how those genes
move across that transition in the Deng et al. 2014 atlas, so the output is a ranked list
of candidate loci with the transcription that accompanies them.

Only the merges and splits whose two domains are adjacent are used, since those are the
ones where a single boundary really appeared or disappeared.

Usage:
    python consensus_events.py
    python consensus_events.py --event-type split --min-resolutions 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from annotate_events import GeneIndex
from gene_sets import load_expression, load_gene_table, transition_change

HERE = Path(__file__).resolve().parent
EVENTS_DIR = HERE / "results" / "events"
OUT_DIR = HERE / "results" / "consensus"

FIELDS = [
    "event_type", "transition", "chrom", "start", "end", "width_kb",
    "configurations", "resolutions", "tolerances", "calls",
    "resolutions_seen", "n_genes", "genes", "top_up_gene", "top_up_change",
    "top_down_gene", "top_down_change", "mean_abs_change",
]


def read_boundary_calls(events_dir=None, max_gap_bins=1, event_types=("split", "merge")):
    """Every adjacent split or merge boundary of every configuration, in bp."""
    root = Path(events_dir or EVENTS_DIR)
    calls = []
    for path in sorted(root.glob("*/*/tol_*_boundaries.tsv")):
        for row in csv.DictReader(open(path), delimiter="\t"):
            if row["event_type"] not in event_types:
                continue
            gap = row["gap_bins"]
            if gap == "" or int(gap) > max_gap_bins:
                continue
            calls.append({
                "event_type": row["event_type"], "transition": row["transition"],
                "chrom": row["chrom"], "start": int(row["start"]), "end": int(row["end"]),
                "resolution": row["resolution"], "tolerance": row["tolerance"],
            })
    return calls


def cluster_calls(calls, slop=0):
    """Merge calls of the same type, transition and chromosome that overlap within ``slop``.

    The windows come from three bin sizes, so the same boundary is a 250 kb window in one
    configuration and a 1 Mb window in another. Clustering on overlap after padding by
    ``slop`` puts those together without letting distant events join up.
    """
    groups = {}
    for call in calls:
        key = (call["event_type"], call["transition"], call["chrom"])
        groups.setdefault(key, []).append(call)

    clusters = []
    for (event_type, transition, chrom), members in groups.items():
        members.sort(key=lambda c: (c["start"], c["end"]))
        current = None
        for call in members:
            if current is not None and call["start"] <= current["end"] + slop:
                current["end"] = max(current["end"], call["end"])
                current["members"].append(call)
                continue
            if current is not None:
                clusters.append(current)
            current = {"event_type": event_type, "transition": transition, "chrom": chrom,
                       "start": call["start"], "end": call["end"], "members": [call]}
        if current is not None:
            clusters.append(current)
    return clusters


def summarise(cluster, index, expression):
    """One consensus row: its support, its genes and how they move at that transition."""
    members = cluster["members"]
    configurations = {(m["resolution"], m["tolerance"]) for m in members}
    resolutions = sorted({m["resolution"] for m in members})
    tolerances = sorted({int(m["tolerance"]) for m in members})

    genes = index.in_window(cluster["chrom"], cluster["start"], cluster["end"])
    changes = []
    for gene in genes:
        change = transition_change(expression, gene, cluster["transition"])
        if change is not None:
            changes.append((change, gene))

    top_up = max(changes, default=None)
    top_down = min(changes, default=None)
    mean_abs = (sum(abs(c) for c, _ in changes) / len(changes)) if changes else ""

    return {
        "event_type": cluster["event_type"], "transition": cluster["transition"],
        "chrom": cluster["chrom"], "start": cluster["start"], "end": cluster["end"],
        "width_kb": round((cluster["end"] - cluster["start"]) / 1000),
        "configurations": len(configurations), "resolutions": len(resolutions),
        "tolerances": len(tolerances), "calls": len(members),
        "resolutions_seen": ",".join(resolutions),
        "n_genes": len(genes), "genes": ",".join(genes),
        "top_up_gene": top_up[1] if top_up else "",
        "top_up_change": round(top_up[0], 3) if top_up else "",
        "top_down_gene": top_down[1] if top_down else "",
        "top_down_change": round(top_down[0], 3) if top_down else "",
        "mean_abs_change": round(mean_abs, 3) if mean_abs != "" else "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-gap-bins", type=int, default=1)
    parser.add_argument("--slop", type=int, default=0,
                        help="bp of padding when deciding whether two calls are the same "
                             "event; 0 means they must overlap")
    parser.add_argument("--event-type", nargs="*", default=["split", "merge"])
    parser.add_argument("--min-resolutions", type=int, default=2,
                        help="only report events called at this many bin sizes")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    index = GeneIndex(load_gene_table())
    expression, _ = load_expression()

    calls = read_boundary_calls(max_gap_bins=args.max_gap_bins,
                                event_types=tuple(args.event_type))
    clusters = cluster_calls(calls, slop=args.slop)
    rows = [summarise(cluster, index, expression) for cluster in clusters]
    rows.sort(key=lambda r: (-r["resolutions"], -r["configurations"], r["chrom"]))

    out_dir = Path(args.out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "consensus_events.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(calls)} adjacent split/merge boundary calls -> {len(rows)} distinct loci")
    support = {}
    for row in rows:
        support[row["resolutions"]] = support.get(row["resolutions"], 0) + 1
    for n in sorted(support, reverse=True):
        print(f"  called at {n} of 3 bin sizes: {support[n]} loci")
    print(f"wrote {path}")

    reproducible = [r for r in rows if r["resolutions"] >= args.min_resolutions]
    print()
    print(f"loci called at >= {args.min_resolutions} bin sizes: {len(reproducible)}")
    if not reproducible:
        return
    print(f"{'type':7}{'transition':18}{'locus':26}{'res':>4}{'cfg':>5}  genes")
    for row in reproducible[:args.top]:
        locus = f"{row['chrom']}:{row['start'] / 1e6:.1f}-{row['end'] / 1e6:.1f}Mb"
        genes = row["genes"][:60] + ("..." if len(row["genes"]) > 60 else "")
        print(f"{row['event_type']:7}{row['transition']:18}{locus:26}"
              f"{row['resolutions']:>4}{row['configurations']:>5}  {genes or '-'}")


if __name__ == "__main__":
    main()
