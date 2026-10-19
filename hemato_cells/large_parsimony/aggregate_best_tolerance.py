#!/usr/bin/env python3
"""
aggregate_best_tolerance.py

Count how often the large parsimony ILP recovers the hematopoietic quartet
{LT, CLP} | {leaf1, leaf2}, treating the harmonization tolerance as a free
parameter. For every chromosome x resolution x leaf pair the tolerances are tried
and the instance counts once:

  correct_best_tol  at least one tolerance makes the correct quartet the unique
                    optimum (ties count as not correct). The best tolerance is the
                    one where the correct quartet wins by the largest relative
                    margin, (best wrong cost - correct cost) / parsimony cost of
                    the correct quartet (the parsimony part only: with an
                    importance term the total cost can be negative). Costs closer
                    than 1e-6 count as tied.
  chance_best_tol   picking the tolerance knowing the answer lifts chance above
                    1/3: with several tries a random winner is more likely to hit
                    the correct quartet at least once. Computed exactly per
                    instance as (number of distinct quartets that are the unique
                    optimum at some tolerance) / 3, i.e. the hit rate if the
                    correct quartet were one of the three at random. It keeps the
                    correlation between tolerances, unlike 1 - (2/3)^n_tol.

The same is done genome-wide, one instance per resolution x leaf pair. With all
chromosomes in one instance the tree topology is shared, but the internal TAD
hierarchies of different chromosomes never interact: no TAD spans two
chromosomes, laminarity only compares TADs of one chromosome, and the objective
is a sum over TADs and boundaries. So for a fixed quartet q

    genome-wide cost(q) = sum over chromosomes c of cost_c(q)

and each JSON already holds cost_c(q) for all three quartets (the pinned solves),
so no new ILP is solved. (Checked against a joint ILP on concatenated chromosomes
with cross-chromosome candidate TADs removed: the pinned costs are identical.)
A genome-wide tolerance is used only when all --n-chrom chromosomes are ranked.

Outputs (in --outdir):
  best_tolerance_per_instance.csv            chromosome x resolution x leaf pair
  best_tolerance_genomewide_per_instance.csv resolution x leaf pair
  best_tolerance_summary.txt

Usage:
    python aggregate_best_tolerance.py --json-dir results/jsons --outdir results/best_tolerance
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

RES_ORDER = ["100kb", "250kb", "500kb", "1mb", "2mb"]
QUARTETS = ["correct", "LT_with_leaf1", "LT_with_leaf2"]
COSTS = ["cost_" + q for q in QUARTETS]
PARS = ["pars_" + q for q in QUARTETS]
TIE_EPS = 1e-6      # weighted costs are fractional; closer than this counts as a tie


def split_kind(split, leaf1, leaf2):
    """Name a quartet by the leaf LT is paired with."""
    partner = split.split("|")[0].split(",")[1]
    return {"CLP": "correct", leaf1: "LT_with_leaf1", leaf2: "LT_with_leaf2"}[partner]


def load_chrom_rows(json_dir):
    """One row per JSON (chromosome x resolution x leaf pair x tolerance)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(json_dir, "quartet_*.json"))):
        with open(path) as fh:
            s = json.load(fh)
        quartets = s.get("quartets", {})
        row = {
            "chrom": int(s["chrom"]),
            "resolution": s["resolution"],
            "tolerance": s["tolerance"],
            "leaf1": s["leaf1"],
            "leaf2": s["leaf2"],
            "leaf_pair": f"{s['leaf1']}+{s['leaf2']}",
            "ranked": len(quartets) == 3 and all(q.get("status") == "optimal" for q in quartets.values()),
        }
        for split, q in quartets.items():
            kind = split_kind(split, s["leaf1"], s["leaf2"])
            row["cost_" + kind] = q.get("cost")
            row["pars_" + kind] = q.get("parsimony_cost", q.get("cost"))
        rows.append(row)
    return pd.DataFrame(rows)


def per_tolerance(costs, pars_correct):
    """costs = [correct, LT_with_leaf1, LT_with_leaf2] at one tolerance."""
    c = np.asarray(costs, float)
    order = np.sort(c)
    winner = QUARTETS[int(np.argmin(c))] if order[1] - order[0] > TIE_EPS else None   # None = tie at the top
    margin = min(c[1], c[2]) - c[0]
    if abs(margin) <= TIE_EPS:
        margin = 0.0
    return winner, margin / max(float(pars_correct), 1.0)


def best_over_tolerance(df, keys):
    """df: one row per (keys, tolerance) with the three quartet costs, ranked rows only."""
    out = []
    for key, g in df.groupby(keys, sort=False):
        g = g.sort_values("tolerance")
        tols = g["tolerance"].tolist()
        per = [per_tolerance(r, pc) for r, pc in zip(g[COSTS].to_numpy(), g["pars_correct"].to_numpy())]
        winners = [w for w, _ in per]
        margins = np.array([m for _, m in per])
        i_best = int(np.argmax(margins))            # first tolerance on ties
        out.append({
            **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
            "n_tol": len(tols),
            "tolerances_correct": ",".join(str(t) for t, w in zip(tols, winners) if w == "correct"),
            "correct_best_tol": bool(margins[i_best] > 0),
            "best_tol": tols[i_best],
            "best_rel_margin": margins[i_best],
            "chance_best_tol": len({w for w in winners if w is not None}) / 3,
        })
    return pd.DataFrame(out)


def genomewide_costs(chrom_df, n_chrom):
    keys = ["resolution", "tolerance", "leaf1", "leaf2", "leaf_pair"]
    ranked = chrom_df[chrom_df["ranked"]]
    gw = ranked.groupby(keys, sort=False).agg(
        n_chrom_ranked=("chrom", "count"), **{c: (c, "sum") for c in COSTS + PARS}).reset_index()
    return gw[gw["n_chrom_ranked"] == n_chrom]


def sort_res(df, keys):
    df = df.copy()
    df["_r"] = df["resolution"].map({r: i for i, r in enumerate(RES_ORDER)})
    return df.sort_values(["_r"] + keys).drop(columns="_r").reset_index(drop=True)


def summarise(best, keys):
    def agg(g):
        return {
            "n": len(g),
            "correct_best_tol": int(g["correct_best_tol"].sum()),
            "frac_correct_best_tol": g["correct_best_tol"].mean(),
            "chance_best_tol": g["chance_best_tol"].mean(),
        }

    if not keys:
        return pd.DataFrame([agg(best)])
    res = pd.DataFrame([{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **agg(g)}
                        for k, g in best.groupby(keys, sort=False)])
    return sort_res(res, [k for k in keys if k != "resolution"]) if "resolution" in keys \
        else res.sort_values(keys).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", default="results/jsons")
    ap.add_argument("--outdir", default="results/best_tolerance")
    ap.add_argument("--n-chrom", type=int, default=19)
    args = ap.parse_args()

    chrom_df = load_chrom_rows(args.json_dir)
    if chrom_df.empty:
        raise SystemExit(f"No quartet_*.json under {args.json_dir}")
    os.makedirs(args.outdir, exist_ok=True)
    fmt = {c: "{:.3f}".format for c in ("frac_correct_best_tol", "chance_best_tol")}
    all_tols = sorted(chrom_df["tolerance"].unique())
    n_unranked = int((~chrom_df["ranked"]).sum())

    lines = [
        "Large parsimony on the hematopoietic quartet, tolerance treated as a free parameter",
        "correct tree = {LT, CLP} | {leaf1, leaf2}",
        f"tolerances tried: {', '.join(map(str, all_tols))}   solves not ranked (not optimal): {n_unranked}",
        "",
        "frac_correct_best_tol  correct quartet is the unique optimum at >= 1 tolerance (the best one counts)",
        "chance_best_tol        what frac_correct_best_tol would be if the correct quartet were random",
        "",
    ]

    # per chromosome
    best = sort_res(best_over_tolerance(chrom_df[chrom_df["ranked"]], ["chrom", "resolution", "leaf_pair"]),
                    ["chrom", "leaf_pair"])
    best.to_csv(os.path.join(args.outdir, "best_tolerance_per_instance.csv"), index=False)
    incomplete = int((best["n_tol"] < len(all_tols)).sum())
    lines += ["#" * 20 + " per chromosome (chromosome x resolution x leaf pair) " + "#" * 20,
              f"instances: {len(best)}   with fewer than {len(all_tols)} ranked tolerances: {incomplete}", ""]
    for name, k in (("overall", []), ("resolution", ["resolution"]),
                    ("leaf_pair", ["leaf_pair"]), ("chrom", ["chrom"])):
        lines += [f"== {name} ==", summarise(best, k).to_string(index=False, formatters=fmt), ""]
    ok = best[best["correct_best_tol"]]
    lines += ["== which tolerance is best, among instances where one is correct ==",
              ok["best_tol"].value_counts().sort_index().rename("n").to_frame().T.to_string(), ""]

    # genome-wide
    gw = genomewide_costs(chrom_df, args.n_chrom)
    gbest = sort_res(best_over_tolerance(gw, ["resolution", "leaf_pair"]), ["leaf_pair"])
    gbest.to_csv(os.path.join(args.outdir, "best_tolerance_genomewide_per_instance.csv"), index=False)
    lines += ["#" * 20 + f" genome-wide (sum over {args.n_chrom} chromosomes; resolution x leaf pair) " + "#" * 20,
              f"instances: {len(gbest)}", ""]
    for name, k in (("overall", []), ("resolution", ["resolution"]), ("leaf_pair", ["leaf_pair"])):
        lines += [f"== {name} ==", summarise(gbest, k).to_string(index=False, formatters=fmt), ""]
    cols = ["resolution", "leaf_pair", "n_tol", "tolerances_correct", "correct_best_tol", "best_tol",
            "best_rel_margin"]
    lines += ["== every genome-wide instance ==",
              gbest[cols].to_string(index=False, formatters={"best_rel_margin": "{:.4f}".format}), ""]

    text = "\n".join(lines)
    with open(os.path.join(args.outdir, "best_tolerance_summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
