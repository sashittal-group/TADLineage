#!/usr/bin/env python3
"""The short list: merge and split loci worth taking to the bench, and what sits on them.

Everything else in this folder is a test. This is the product of the tests: the handful of
loci that survive being asked for at more than one bin size, annotated with their genes,
with how those genes move across that transition in the Deng et al. 2014 atlas, and with
which curated gene sets they belong to.

Positional tolerance enters here, and this is the one place it earns its keep. Calls come
from three bin sizes, so the same boundary is a 250 kb window in one configuration and a
1 Mb window in another, and requiring strict overlap throws away agreements that differ by
a bin or two of sparse-data jitter. Padding by ``--slop`` before clustering recovers them:
the count of loci seen at two bin sizes runs 15, 22, 26, 43 at 0, 0.5, 1 and 2 Mb.

    This is a ranked list of hypotheses, not a set of findings. None of these loci is
    individually significant, the genome-wide tests behind them are null, and the ranking
    rewards agreement between configurations that share input data. Treat a locus as worth
    a look because of what is on it, and check it against orthogonal data before believing
    it.

Usage:
    python candidate_loci.py
    python candidate_loci.py --slop 1000000 --min-resolutions 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import consensus_events
from annotate_events import GeneIndex
from gene_sets import load_curated_sets, load_expression, load_gene_table

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"

FIELDS = [
    "rank", "event_type", "transition", "chrom", "start", "end", "width_kb",
    "resolutions", "resolutions_seen", "configurations", "calls", "n_genes",
    "mean_abs_change", "top_up_gene", "top_up_change", "top_down_gene",
    "top_down_change", "curated_sets", "genes",
]

# What is known about these loci, from the literature rather than from this analysis.
# Keyed by (chrom, start megabase, transition, event_type), because one chromosome can
# carry two loci of the same type at the same transition. The start is matched to the
# nearest megabase so that a small change in --slop does not silently drop a note.
NOTES = {
    ("chr7", 58, "2-cell->4-cell", "split"): (
        "The Prader-Willi / Angelman imprinted domain. The window is Atp10a, Ube3a, Ipw, "
        "Snord116 and Snord64 and almost nothing else, which is the mouse orthologue of "
        "the best characterised imprinted chromatin domain in mammals. It is imprinted, "
        "it is known to be organised into an allele-specific domain, and a boundary "
        "appearing inside it is the single most checkable prediction in this list."),
    ("chr17", 36, "4-cell->8-cell", "split"): (
        "The window spans Zfp57 in the H2 / MHC region. ZFP57 with TRIM28 is the factor "
        "that maintains methylation at imprinting control regions through the "
        "preimplantation divisions, and it is strongly maternal: it falls steadily from "
        "the zygote (-2.6, -2.9, -2.0 across the first three transitions) before the "
        "embryonic copy comes up at the blastocyst. A boundary appearing here as the "
        "maternal store runs out is a coherent story, though the window also carries a "
        "large olfactory receptor cluster and is gene dense for other reasons."),
    ("chr8", 43, "8-cell->64-cell", "merge"): (
        "Triml1, Triml2 and Zfp42 (Rex1). All three are preimplantation specific: Zfp42 "
        "is the classic naive pluripotency marker and rises 1.9 across this transition, "
        "Triml2 rises 0.6. Only nine genes in the window, so the annotation is unusually "
        "clean for a locus this size."),
    ("chr19", 47, "8-cell->64-cell", "merge"): (
        "Pcgf6 falls 1.9 across this transition. PCGF6 is the defining subunit of "
        "PRC1.6, the complex that keeps germline and meiotic genes off in the early "
        "embryo, and its loss at the blastocyst is expected. See also the chr6 and "
        "chr12 loci in this list, which carry two more subunits of the same complex."),
    ("chr6", 99, "8-cell->64-cell", "merge"): (
        "Rybp falls 1.9 across this transition. RYBP is the core subunit shared by all "
        "variant PRC1 complexes, PRC1.6 included."),
    ("chr12", 14, "2-cell->4-cell", "merge"): (
        "E2f6 rises 1.3 across this transition. E2F6 is the DNA-binding subunit of "
        "PRC1.6. With Pcgf6 and Rybp above, three subunits of one complex sit under "
        "three separate merge loci -- an observation, not a test, and one that a "
        "properly powered gene set test would have to confirm."),
    ("chr3", 28, "1-cell->2-cell", "merge"): (
        "Merges at 1-cell to 2-cell and splits back at 2-cell to 4-cell, the only "
        "reversible locus in the list, and the one with the largest transcriptional "
        "movement (mean |log2 FC| 0.97 then 0.66). Carries Prkci, the atypical PKC that "
        "sets up apical-basal polarity for the first cell fate decision, Phc3, a "
        "canonical PRC1 subunit, and Mecom."),
    ("chr2", 86, "4-cell->8-cell", "split"): (
        "140 genes, nearly all of them olfactory receptors, plus the Pramel cluster. "
        "Large olfactory receptor arrays are gene dense but transcriptionally silent, so "
        "this is the locus in the list most likely to be a gene density artefact."),
}


def curated_membership(symbols, sets):
    hits = []
    for name, members in sorted(sets.items()):
        overlap = sorted(set(symbols) & set(members))
        if overlap:
            hits.append(f"{name}:{'|'.join(overlap)}")
    return ";".join(hits)


def build(slop, min_resolutions, max_gap_bins, event_types, top,
          events_dir=None):
    index = GeneIndex(load_gene_table())
    expression, _ = load_expression()
    curated = load_curated_sets(genes=load_gene_table())

    calls = consensus_events.read_boundary_calls(events_dir=events_dir,
                                                 max_gap_bins=max_gap_bins,
                                                 event_types=tuple(event_types))
    clusters = consensus_events.cluster_calls(calls, slop=slop)
    rows = []
    for cluster in clusters:
        row = consensus_events.summarise(cluster, index, expression)
        if int(row["resolutions"]) < min_resolutions:
            continue
        symbols = [s for s in row["genes"].split(",") if s]
        row["curated_sets"] = curated_membership(symbols, curated)
        rows.append(row)

    rows.sort(key=lambda r: (-int(r["resolutions"]), -int(r["calls"]),
                             -float(r["mean_abs_change"] or 0)))
    if top:
        rows = rows[:top]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def write_markdown(path, rows, slop, min_resolutions):
    lines = [
        "# Candidate merge and split loci",
        "",
        f"Generated by `candidate_loci.py --slop {slop} "
        f"--min-resolutions {min_resolutions}`.",
        "",
        "Loci where an adjacent merge or split between consecutive stage medians was "
        "called at more than one bin size, after allowing "
        f"{slop / 1e6:g} Mb of positional tolerance when matching calls across "
        "configurations. Expression changes are mean log2(RPKM+1) differences between the "
        "matching stages of the Deng et al. 2014 atlas.",
        "",
        "**These are hypotheses.** The genome-wide enrichment, expression and CTCF tests "
        "behind them are null (see `REPORT.md`), no locus here is individually "
        "significant, and configurations that agree share input data. The list is worth "
        "reading for what sits on the loci, not as evidence that the loci are real.",
        "",
        "One theme does recur. Two of the annotated loci below are about imprinting -- a "
        "split inside the Prader-Willi / Angelman domain on chr7, and a split over Zfp57, "
        "the factor that maintains imprints through exactly these divisions -- and "
        "imprinted genes are separately the only curated set whose enrichment crosses 1 "
        "at all, and only barely, when boundaries are given positional tolerance "
        "(section 8 of `REPORT.md`). "
        "Neither line is significant on its own. Together they are the reason to test "
        "imprinting directly, on allele-resolved calls this data cannot provide.",
        "",
        "| # | event | transition | locus | bin sizes | genes | strongest up | "
        "strongest down |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        locus = (f"{row['chrom']}:{int(row['start']) / 1e6:.1f}-"
                 f"{int(row['end']) / 1e6:.1f} Mb")
        up = (f"{row['top_up_gene']} ({row['top_up_change']})"
              if row["top_up_gene"] else "-")
        down = (f"{row['top_down_gene']} ({row['top_down_change']})"
                if row["top_down_gene"] else "-")
        lines.append(
            f"| {row['rank']} | {row['event_type']} | {row['transition']} | {locus} | "
            f"{row['resolutions_seen']} | {row['n_genes']} | {up} | {down} |"
        )

    lines += ["", "## The ones with a story", ""]
    for row in rows:
        start_mb = int(row["start"]) // 1_000_000
        note = next((text for (chrom, mb, transition, kind), text in NOTES.items()
                     if chrom == row["chrom"] and transition == row["transition"]
                     and kind == row["event_type"] and abs(mb - start_mb) <= 1), None)
        if not note:
            continue
        locus = (f"{row['chrom']}:{int(row['start']) / 1e6:.1f}-"
                 f"{int(row['end']) / 1e6:.1f} Mb")
        lines += [f"### {row['event_type']} at {row['transition']}, {locus}", "", note, ""]

    lines += [
        "## How to read the rest",
        "",
        "`candidates.tsv` alongside this file carries every column, including the full "
        "gene list and which curated gene sets each window intersects. A locus with no "
        "note above is one where nothing in the window stood out; that is the usual case "
        "and is the reason this list is short.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slop", type=int, default=2000000,
                        help="bp of positional tolerance when matching calls across "
                             "configurations")
    parser.add_argument("--min-resolutions", type=int, default=2)
    parser.add_argument("--max-gap-bins", type=int, default=1)
    parser.add_argument("--event-type", nargs="*", default=["split", "merge"])
    parser.add_argument("--top", type=int, default=0, help="0 keeps all")
    parser.add_argument("--events-dir", default=None,
                        help="event tables to read; use results/events_ontad for the "
                             "OnTAD replication")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    rows = build(args.slop, args.min_resolutions, args.max_gap_bins, args.event_type,
                 args.top, events_dir=args.events_dir)
    out_dir = Path(args.out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = out_dir / "candidates.tsv"
    with open(table, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    markdown = out_dir / "CANDIDATES.md"
    write_markdown(markdown, rows, args.slop, args.min_resolutions)
    print(f"wrote {table}")
    print(f"wrote {markdown} ({len(rows)} loci)")


if __name__ == "__main__":
    main()
