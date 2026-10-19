#!/usr/bin/env python3
"""What biology, if any, each edge carries -- read off the whole configuration grid.

For every edge of the tree this prints the gene sets that

  * reach ``--q-max`` after Benjamini-Hochberg,
  * overlap at least ``--min-overlap`` genes at ``--min-fold`` or better,
  * match the haematopoietic vocabulary of ``plot_dotplot.FOCUS_TOKENS``,

ranked by how many of the twenty-four configurations (2 callers x 2 resolutions x 6
tolerances) support them.  Configuration count is the honest summary statistic here: the
grid was searched for hits, so a single spectacular q means less than a set that survives
resolution and tolerance being changed underneath it.

Nothing is filtered on gene-cluster concentration: ``clust`` is printed beside every row
so the leading edge is visible, the way a leading-edge gene list normally is.

``--genes`` prints the overlapping genes, which is the only way to tell
``RUNX1_REGULATES_GENES_INVOLVED_IN_MEGAKARYOCYTE_DIFFERENTIATION`` (27 of 29 genes are the
chr13 Hist1 cluster) from ``GOBP_PLATELET_MORPHOGENESIS`` (Mpl, Tal1, Zfpm1, Nbeal2, Tubb1).

Usage:
    python per_edge_biology.py
    python per_edge_biology.py --genes --top 6
    python per_edge_biology.py --all --q-max 0.05
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_dotplot import in_focus, tandem  # noqa: E402

RESULTS = HERE / "results"
# every edge either tree has, in lineage order; RobusTAD has no ST call so it goes LT->MPP
EDGES = ["LT->ST", "LT->MPP", "ST->MPP", "MPP->CMP", "MPP->CLP", "CMP->MEP",
         "CMP->GMP", "MEP->MKP", "MKP->MK", "GMP->G"]
WHAT = {
    "LT->ST": "long-term to short-term HSC: exit from dormancy",
    "LT->MPP": "HSC to multipotent progenitor (RobusTAD tree, no ST)",
    "ST->MPP": "loss of self-renewal, cell-cycle entry",
    "MPP->CMP": "myeloid commitment",
    "MPP->CLP": "lymphoid commitment",
    "CMP->MEP": "erythro-megakaryocytic commitment",
    "CMP->GMP": "granulocyte-monocyte commitment",
    "MEP->MKP": "megakaryocyte commitment",
    "MKP->MK": "megakaryocyte maturation, endomitosis",
    "GMP->G": "granulocyte maturation, cell-cycle exit",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=str(RESULTS / "enrichment_grid.tsv"))
    ap.add_argument("--q-max", type=float, default=0.10)
    ap.add_argument("--min-overlap", type=int, default=10)
    ap.add_argument("--min-fold", type=float, default=1.5)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--dispersed-only", action="store_true",
                    help="drop rows whose overlap is mostly a few tandem arrays; off by "
                         "default, so what prints is what Fisher returned")
    ap.add_argument("--genes", action="store_true")
    args = ap.parse_args()

    with open(args.grid) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    keep = [r for r in rows
            if float(r["q"]) <= args.q_max
            and int(r["overlap"]) >= args.min_overlap
            and float(r["fold"]) >= args.min_fold
            and in_focus(r["gene_set"])
            and not (args.dispersed_only and tandem(r))]

    support: dict = defaultdict(set)
    best: dict = {}
    for r in keep:
        key = (r["edge"], r["gene_set"])
        support[key].add((r["caller"], r["resolution"], r["tolerance"]))
        if key not in best or float(r["q"]) < float(best[key]["q"]):
            best[key] = r

    by_edge = defaultdict(list)
    for key, configs in support.items():
        by_edge[key[0]].append((len(configs), sorted({c[0] for c in configs}),
                                key[1], best[key]))

    label = "dispersed only" if args.dispersed_only else "all, tandem arrays included"
    print(f"Haematopoietic gene sets at q <= {args.q_max}, overlap >= {args.min_overlap}, "
          f"fold >= {args.min_fold}  ({label})")
    print("'clust' is the share of the overlap inside three 2 Mb windows -- a description "
          "of the leading edge, not a filter.\n")
    for edge in EDGES:
        hits = sorted(by_edge.get(edge, []), reverse=True)
        print(f"===== {edge:<10} {WHAT[edge]}  ({len(hits)} sets) =====")
        if not hits:
            print("   nothing\n")
            continue
        for n, callers, name, r in hits[:args.top]:
            print(f"  {n:2d}/24 cfg  {'+'.join(callers):<16} q={float(r['q']):.1e} "
                  f"fold={float(r['fold']):>5.2f} overlap={r['overlap']}/{r['set_size']} "
                  f"clust={r['cluster_frac']}")
            print(f"             {name}")
            if args.genes:
                print(f"             {r['genes'].replace(';', ' ')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
