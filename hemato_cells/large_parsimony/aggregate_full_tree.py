#!/usr/bin/env python3
"""
aggregate_full_tree.py

Collect the infer_full_tree.py JSONs and report quartet consistency per bin size
and tolerance (each tolerance on its own, no best-tolerance selection).

An instance's consistency is the mean over all its co-optimal trees. For each bin
size x tolerance, over the chromosomes whose solve finished: mean / median / min /
max of that, how many chromosomes give the reference tree exactly (consistency 1),
and the median number of co-optimal trees. A random binary tree scores 1/3.

Outputs (in --outdir):
    full_tree_per_instance.csv    one row per chromosome x bin size x tolerance
    full_tree_summary.csv         one row per bin size x tolerance
    full_tree_summary.txt         the same, as a readable table
"""

import argparse
import glob
import json
import os

import pandas as pd

RES_ORDER = ["2mb", "1mb", "500kb", "250kb", "100kb"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", default="results/jsons")
    ap.add_argument("--outdir", default="results/summary")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.json_dir, "full_chr*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        rows.append({
            "chrom": int(d["chrom"]),
            "resolution": d["resolution"],
            "tolerance": d["tolerance"],
            "status": d["status"],
            "quartet_consistency": d.get("quartet_consistency"),
            "quartet_consistency_min": d.get("quartet_consistency_min"),
            "quartet_consistency_max": d.get("quartet_consistency_max"),
            "n_optimal_trees": d.get("n_optimal_trees"),
            "n_quartets_resolved": d.get("n_quartets_resolved_in_reference"),
            "cost": d.get("cost"),
            "n_candidate_tads": d.get("n_candidate_tads"),
            "seconds_total": d.get("seconds_total"),
            "best_newick": (d.get("optimal_trees") or [{}])[0].get("newick"),
        })
    if not rows:
        raise SystemExit(f"No JSONs in {args.json_dir}")
    df = pd.DataFrame(rows)
    df["resolution"] = pd.Categorical(df["resolution"], [r for r in RES_ORDER if r in set(df["resolution"])])
    df = df.sort_values(["resolution", "tolerance", "chrom"])

    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "full_tree_per_instance.csv"), index=False)

    summary = []
    for (res, tol), g in df.groupby(["resolution", "tolerance"], observed=True):
        ok = g[g["status"] == "optimal"]
        qc = ok["quartet_consistency"]
        summary.append({
            "resolution": res,
            "tolerance": tol,
            "n_chrom": len(g),
            "n_solved": len(ok),
            "mean": qc.mean() if len(ok) else None,
            "median": qc.median() if len(ok) else None,
            "min": qc.min() if len(ok) else None,
            "max": qc.max() if len(ok) else None,
            "n_exact": int((qc == 1.0).sum()),
            "median_n_optimal_trees": ok["n_optimal_trees"].median() if len(ok) else None,
        })
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(args.outdir, "full_tree_summary.csv"), index=False)

    with open(os.path.join(args.outdir, "full_tree_summary.txt"), "w") as fh:
        fh.write("Quartet consistency of the inferred 10-cell tree with the known lineage\n")
        fh.write("(174 resolved reference quartets; a random binary tree scores 0.333;\n")
        fh.write(" per chromosome, the mean over all co-optimal trees)\n\n")
        fh.write(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        fh.write("\n")
    print(open(os.path.join(args.outdir, "full_tree_summary.txt")).read())


if __name__ == "__main__":
    main()
