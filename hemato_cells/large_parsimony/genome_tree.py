#!/usr/bin/env python3
"""
genome_tree.py

One genome-wide large parsimony tree over the ten cell types, instead of one tree
per chromosome (infer_full_tree.py). The tree is shared by all chromosomes and its
cost is the sum of the per-chromosome costs: no TAD spans two chromosomes and
laminarity only compares TADs of one chromosome, so for a fixed tree

    genome-wide cost(tree) = sum over chromosomes c of cost_c(tree).

This is the concatenation ("supermatrix") approach of phylogenetics: every
chromosome votes on the same topology, so per-chromosome noise averages out.

Search (exact, all co-optimal trees): the branch and bound of topology_search.py.
The Fitch bound is taken on all chromosomes' characters at once (the chromosomes'
bins are shifted apart, so their TADs and boundaries are distinct characters), and
a complete tree is scored by summing the per-chromosome exact costs, each from that
chromosome's own TopologySearch cascade with its share of the upper bound.

Usage (from an analysis folder):
    python ../genome_tree.py --data-dir ../.. --resolution 1mb --tolerance 1 \
        --weighting ontad_score --output results/jsons/genome_1mb_1.json
"""

import argparse
from collections import defaultdict
import os
import tempfile
import time
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from topology_search import TopologySearch  # noqa: E402
from infer_full_tree import (NODES, REFERENCE_EDGES, ontad_score_weights,  # noqa: E402
                             robustad_score_weights, quartet_scorer, newick_of_edges)
from infer_quartet_tree import harmonize_private, tad_path, write_json  # noqa: E402

CHROMS = [str(c) for c in range(1, 20)]
OFFSET = 10 ** 6          # bins; far above any chromosome's length in bins


class GenomeSearch(TopologySearch):
    """TopologySearch whose characters are all chromosomes' (shifted apart) and whose
    exact tree cost is the sum of per-chromosome exact costs."""

    def __init__(self, parts, threads=None, time_limit=None, verbose=False, Ct=1.0, Cb=1.0):
        # parts: list of (leaf_tads, weights) per chromosome
        self.parts = [TopologySearch(tads, Ct=Ct, Cb=Cb, threads=threads, time_limit=time_limit, **w)
                      for tads, w in parts]
        shift = lambda t, o: (t[0] + o, t[1] + o)  # noqa: E731
        k = len(parts[0][0])
        leaf_tads = [set() for _ in range(k)]
        tw, sw, ew = {}, {}, {}
        for c, (tads, w) in enumerate(parts):
            o = c * OFFSET
            for i in range(k):
                leaf_tads[i] |= {shift(t, o) for t in tads[i]}
            tw.update({shift(t, o): v for t, v in (w.get("tad_weights") or {}).items()})
            sw.update({p + o: v for p, v in (w.get("start_weights") or {}).items()})
            ew.update({p + o: v for p, v in (w.get("end_weights") or {}).items()})
        super().__init__(leaf_tads, Ct=Ct, Cb=Cb, threads=threads, time_limit=time_limit,
                         verbose=verbose, tad_weights=tw or None, start_weights=sw or None,
                         end_weights=ew or None, build_groups=False)

    # Each chromosome's all-arcs LP model is cached (rebuilding it for every tree
    # is what makes scoring slow) up to a budget of candidate TADs: ~120 KB of
    # memory per candidate, so 40000 ~ 5 GB. Least recently used models go first.
    LP_CANDIDATE_BUDGET = int(os.environ.get("LP_BUDGET", 40000))

    def _release_lps(self, keep):
        order = [p for p in getattr(self, "_lp_order", []) if p is not keep]
        if keep._lp is not None:
            order.insert(0, keep)
        total, kept = 0, []
        for i, p in enumerate(order):
            total += len(p._lp[3])
            if i == 0 or total <= self.LP_CANDIDATE_BUDGET:
                kept.append(p)
            else:
                p._lp[0].dispose()
                p._lp = None
        self._lp_order = kept

    def _progress(self):
        parts = defaultdict(int)
        for p in self.parts:
            for key, v in p.counts.items():
                parts[key] += v
        sec = defaultdict(float)
        for p in self.parts:
            for key, v in p.seconds.items():
                sec[key] += v
        return {"trees": dict(self.counts), "chromosome steps": dict(parts),
                "seconds": {k: round(v) for k, v in sec.items()}, "cached LPs": len(getattr(self, "_lp_order", []))}

    def evaluate(self, edges, ub, fitch=None):
        """Exact genome-wide cost of the tree if it can be <= ub, else None.

        The per-chromosome cascade (topology_search.TopologySearch.evaluate) run in
        stages across chromosomes: every chromosome's cheap bounds first, so a tree
        is ruled out by the SUM of the bounds before any ILP runs, and each ILP gets
        the cutoff ub minus the other chromosomes' best bounds (or exact costs)."""
        adj = self._adj(edges)
        n = len(self.parts)
        lb = [p.lower_bound(adj) for p in self.parts]       # a lower bound per chromosome
        exact = [None] * n                                   # exact cost once known

        def out(stage):
            self.counts["ruled_out_" + stage] += 1
            return None

        # stage 1: Fitch labeling (upper bound, laminar or None) and grouped bound
        h = [None] * n
        for i, p in enumerate(self.parts):
            t0 = time.time()
            h[i] = p.labeling_cost(adj)
            p.seconds["labeling"] += time.time() - t0
            if h[i] is not None and p._reaches(lb[i], h[i]):
                p.counts["fitch_labeling"] += 1
                exact[i] = lb[i] = h[i]
                continue
            t0 = time.time()
            lb[i] = max(lb[i], p.group_bound(adj))
            p.seconds["group_bound"] += time.time() - t0
            if h[i] is not None and p._reaches(lb[i], h[i]):
                p.counts["group_bound"] += 1
                exact[i] = lb[i] = h[i]
        if self._above(sum(lb), ub):
            return out("group")

        # stage 2: LP relaxations of the chromosomes still open
        for i, p in enumerate(self.parts):
            if exact[i] is not None:
                continue
            t0 = time.time()
            lp, lp_exact = p.lp_bound(edges)
            p.seconds["lp"] += time.time() - t0
            self._release_lps(p)
            lb[i] = max(lb[i], lp)
            if h[i] is not None and p._reaches(lb[i], h[i]):
                p.counts["lp_reaches_labeling"] += 1
                exact[i] = lb[i] = h[i]
            elif lp_exact:
                p.counts["lp_integral"] += 1
                exact[i] = lb[i] = lp
            if self._above(sum(lb), ub):
                return out("lp")

        # stage 3: ILPs, open chromosomes with the widest bound gap first
        open_ = sorted((i for i in range(n) if exact[i] is None),
                       key=lambda i: -((h[i] if h[i] is not None else lb[i]) - lb[i]))
        for i in open_:
            cost = self.parts[i].ilp_cost(edges, ub - (sum(lb) - lb[i]))
            if cost is None:
                return out("ilp")
            exact[i] = lb[i] = cost
            if self._above(sum(lb), ub):
                return out("ilp")
        self.counts["exact"] += 1
        return sum(exact)

def top_fraction_copy(path, fraction, out):
    """Copy of an OnTAD tads.tad keeping the ``fraction`` of its calls with the
    highest score (column 5; ties broken by file order), score column included."""
    with open(path) as fh:
        rows = [line.split() for line in fh if len(line.split()) >= 5]
    rows.sort(key=lambda r: -float(r[4]))
    with open(out, "w") as fh:
        for r in rows[:max(1, round(fraction * len(rows)))]:
            fh.write("\t".join(r) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="../..")
    ap.add_argument("--resolution", required=True, nargs="+",
                    help="one or more resolutions; several are concatenated like chromosomes "
                         "(every (chromosome, resolution) is its own part)")
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("--chroms", nargs="*", default=CHROMS)
    ap.add_argument("--Ct", type=float, default=1.0)
    ap.add_argument("--Cb", type=float, default=1.0)
    ap.add_argument("--weighting", default="none", choices=["none", "ontad_score", "robustad_score"])
    ap.add_argument("--robustad-root", default=str(HERE.parents[1] / "robustad_predictions"))
    ap.add_argument("--top-fraction", type=float, default=1.0,
                    help="keep only this fraction of each cell's OnTAD calls per chromosome, "
                         "highest score first, before harmonization (1 = all)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    res, tol = "+".join(args.resolution), args.tolerance
    parts, info = [], {}
    tmp = tempfile.TemporaryDirectory(prefix="top_fraction_")
    for chrom, r in ((c, r) for r in args.resolution for c in args.chroms):
        raw_paths = {n: tad_path(args.data_dir, n, chrom, r) for n in NODES}
        if args.top_fraction < 1:
            raw_paths = {n: top_fraction_copy(p, args.top_fraction, os.path.join(tmp.name, f"{n}_{chrom}_{r}.tad"))
                         for n, p in raw_paths.items()}
        harmonized = harmonize_private(raw_paths, tol)
        w = {}
        if args.weighting == "ontad_score":
            tw, sw, ew, _ = ontad_score_weights(raw_paths, harmonized, tol)
            w = {"tad_weights": tw, "start_weights": sw, "end_weights": ew}
        elif args.weighting == "robustad_score":
            tw, sw, ew, _ = robustad_score_weights(raw_paths, harmonized, tol, chrom, r, args.robustad_root)
            w = {"tad_weights": tw, "start_weights": sw, "end_weights": ew}
        parts.append(([harmonized[n] for n in NODES], w))
        info[f"chr{chrom}_{r}"] = {n: len(harmonized[n]) for n in NODES}

    t0 = time.time()
    search = GenomeSearch(parts, threads=args.threads or None, verbose=args.verbose, Ct=args.Ct, Cb=args.Cb)
    result = search.run()
    score = quartet_scorer(NODES, REFERENCE_EDGES)
    per_tree = [score(search.split_masks(search.edges_of(p))) for p in result["optimal_paths"]]
    order = sorted(range(len(per_tree)), key=lambda i: -per_tree[i])
    summary = {
        "resolution": res, "tolerance": tol, "chroms": args.chroms, "weighting": args.weighting,
        "Ct": args.Ct, "Cb": args.Cb, "top_fraction": args.top_fraction, "status": result["status"], "cost": result["cost"],
        "n_optimal_trees": len(per_tree),
        "quartet_consistency": sum(per_tree) / len(per_tree),
        "quartet_consistency_min": min(per_tree), "quartet_consistency_max": max(per_tree),
        "optimal_trees": [{"newick": newick_of_edges(search, search.edges_of(result["optimal_paths"][i]), NODES),
                           "quartet_consistency": per_tree[i]} for i in order[:100]],
        "settled_by": result["settled_by"], "n_bnb_nodes": result["n_bnb_nodes"],
        "n_complete_survivors": result["n_complete_survivors"], "seconds": time.time() - t0,
        "n_tads_after_harmonization": info,
    }
    print(f"{res} tol={tol} {args.weighting} top={args.top_fraction:g}: cost {result['cost']:g}, {len(per_tree)} optimal tree(s), "
          f"consistency {summary['quartet_consistency']:.3f} ({time.time() - t0:.0f}s)")
    for t in summary["optimal_trees"][:3]:
        print(f"  {t['newick']}  {t['quartet_consistency']:.3f}")
    write_json(summary, args.output)


if __name__ == "__main__":
    main()
