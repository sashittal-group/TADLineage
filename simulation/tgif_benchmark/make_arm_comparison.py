#!/usr/bin/env python3
"""Add a TGIF-DB line to an ARM's leaf figure, in that arm's own plots/ folder.

Writes TWO figures per arm, because there is no single honest way to do it:

  f1_vs_noise_with_tgif_interval.png
      The arm's OWN figure, reproduced exactly -- its metric (interval F1), its
      numbers straight out of its results/, its filters -- with TGIF overlaid.
      The caller and tree-informed curves are identical to the arm's existing
      f1_vs_noise.png. TGIF's line is BOUNDARY F1, because TGIF has no interval
      output at all, so this figure carries two metrics on one axis and says so.

  f1_vs_noise_with_tgif_boundary.png
      All three lines on one consistent metric: boundary F1. The axis is honest,
      but the caller and tree-informed curves will NOT match the arm's existing
      figure, because that one is interval F1.

Pick whichever the argument needs; do not mix them up in a paper.

FILTERS ARE THE ARM'S, READ FROM THE ARM
----------------------------------------
``MIN_METRIC_DEFAULT`` and ``EXCLUDE_DEPTHS_DEFAULT`` are imported from each
arm's own plot module rather than restated here, so they cannot drift:

    robustad   min-metric 0.4,  no depths excluded
    ontad      min-metric 0.5,  0.003x excluded entirely

The min-metric filter drops tree-informed points below the threshold and NEVER
touches the baseline -- that is the arm's behaviour, and it is survivorship bias
(the robustad arm's own plot module documents this at length). It is reproduced
here rather than fixed, because the point of these figures is to sit next to the
arm's, and a figure filtered differently would not. **TGIF is treated as a
baseline and is never min-metric filtered**, for the same reason the caller is
not: filtering only the lines you are comparing manufactures the comparison.
Depth exclusions DO apply to every line, TGIF included -- an excluded rung is
simply not on the x-axis.

NOTHING IS RE-SOLVED BY DEFAULT
-------------------------------
The boundary-level numbers are read from ``f1_vs_noise_with_tgif_boundary.csv``
if it exists. They have to be *computed* once, because the arm's results/ store
interval-level METRICS and a boundary set cannot be recovered from an F1 number
-- but only once. Pass --recompute to redo that solve.

Usage
-----
    python make_arm_comparison.py --arm robustad
    python make_arm_comparison.py --arm ontad

Run in the snakemake-tutorial env (matplotlib + pandas; gurobipy only if the
boundary CSV has to be built).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
SIM_ROOT = EXP_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

# Importing run_tgif_bench also puts the repo root, simulation/, the sweep dir
# and BOTH arms on sys.path, which is what makes the arm modules importable.
import run_tgif_bench as rb
import config
from trees import get_tree

ARMS = {
    "ontad": {
        "dir": SIM_ROOT / "tree_informed_tad_inference_ontad",
        "run_module": "run_tree_impute",
        "plot_module": "plot_tree_impute",
        "caller_label": "OnTAD on the leaf (no tree)",
        "headline_cost": 1.0,
    },
    "robustad": {
        "dir": SIM_ROOT / "tree_informed_tad_inference_robustad",
        "run_module": "run_tree_impute_rt",
        "plot_module": "plot_tree_impute_rt",
        "caller_label": "RobusTAD on the leaf (no tree)",
        "headline_cost": 0.5,
    },
}

CALLER, IMPUTE, TGIF = "caller", "impute", "tgif"

# One colour per ROLE, shared by both arms, so the two panels read the same way.
# Grey and blue are the arms' own baseline/headline colours; red is TGIF's.
STYLE = {
    CALLER: dict(color="#4c4c4c", marker="o", lw=2.0, ls="-", zorder=5),
    IMPUTE: dict(color="#2b6cb0", marker="D", lw=2.6, ls="-", zorder=7),
    TGIF:   dict(color="#c0392b", marker="^", lw=2.4, ls="-", zorder=6),
}
XLABEL = "Hi-C sequencing depth (cleanest → noisiest)"
# The ribbon is HALF a standard deviation, not a full one. At 200 regions the
# full-SD bands overlap so heavily that the three lines are hard to separate;
# the label on the legend says which it is, so nothing is implied about
# significance either way.
BAND_SCALE = 0.5
BAND_LABEL = {
    "sd": f"line thickness = ±{BAND_SCALE:g} SD across regions",
    "sem": f"line thickness = ±{BAND_SCALE:g} SEM of the mean",
}
DEPTH_ORDER_HINT = ["1.0x", "0.3x", "0.1x", "0.03x", "0.01x", "0.003x"]


# ---------------------------------------------------------------------------
# Boundary-level scores (computed once, then cached to CSV)
# ---------------------------------------------------------------------------
def sim_params(args) -> config.SimParams:
    return dataclasses.replace(
        config.DEFAULT_PARAMS,
        n_bins=args.n_bins, n_top_tads=args.n_top_tads,
        nesting_depth=args.nesting_depth, min_size=args.min_size,
        min_score_node_tads=args.min_score_node_tads,
        events_per_edge=args.events_per_edge,
        tolerance=args.tolerance, seed=args.seed,
    )


def arm_predictions(arm: str, mod, tree, pt: dict, cost: float, args):
    """(caller-alone, tree-informed) interval predictions for one depth rung."""
    calls = {leaf: {(int(r[0]), int(r[1])) for r in rows}
             for leaf, rows in pt["leaf_calls"].items()}
    raw = mod.method_raw(tree, calls)

    if arm == "ontad":
        pred, _ = mod.method_impute_snapped(
            tree, calls, cost, args.Ct, args.Cb, args.solve_mode, args.tolerance)
    else:
        tad_scores = {leaf: {(int(r[0]), int(r[1])): float(r[2]) for r in rows}
                      for leaf, rows in pt["leaf_calls"].items()}
        boundary = {leaf: mod.BoundaryScores.from_record(rec, args.tolerance)
                    for leaf, rec in pt.get("leaf_boundary", {}).items()}
        use_imp, use_pri, wsnap = mod.METHOD_SWITCHES[args.robustad_prefix]
        pred, _ = mod.method_impute(
            tree, calls, tad_scores, boundary, cost,
            args.Ct, args.Cb, args.solve_mode, args.tolerance,
            use_importance=use_imp, use_price_discount=use_pri,
            weighted_snap=wsnap, importance_weight=args.importance_weight,
            price_discount=args.price_discount, price_floor=args.price_floor,
            calibrate_importance=True)

    if pred is None:
        print(f"  WARNING: LP found no solution at {pt['depth_label']}; "
              f"falling back to raw calls", flush=True)
        pred = raw
    return raw, pred


def compute_boundary_rows(arm: str, args, cost: float) -> pd.DataFrame:
    """Score caller / tree-informed / TGIF on BOUNDARIES, per (region, depth).

    This is the one expensive step, and the only one that needs the solver: the
    arm's results/ hold interval-level metrics, and boundaries cannot be
    recovered from a metric. Cached to CSV so it happens once.
    """
    cfg = ARMS[arm]
    mod = __import__(cfg["run_module"])
    params = sim_params(args)

    tgif_files = {(p.parent.name, p.stem): p
                  for p in sorted(Path(args.tgif_calls).glob("*/region_*.json"))}
    arm_calls = Path(cfg["dir"]) / "calls"
    rows = []
    for (tree_name, region_stem), tgif_path in tgif_files.items():
        arm_path = arm_calls / tree_name / f"{region_stem}.json"
        if not arm_path.exists():
            print(f"skip {tree_name}/{region_stem}: no {arm} calls cached")
            continue
        region = int(region_stem.removeprefix("region_"))
        tree = get_tree(tree_name, args.n_nodes, args.tree_seed)
        hiers = mod.build_hierarchies(tree, region, params)
        truth_bd = {l: rb.intervals_to_boundaries(mod.true_intervals(hiers[l]),
                                                  params.n_bins)
                    for l in tree.leaves}

        tgif_rec = json.loads(tgif_path.read_text())
        arm_by_depth = {p["depth_label"]: p
                        for p in json.loads(arm_path.read_text())["points"]}

        for tpt in tgif_rec["points"]:
            depth = tpt["depth_label"]
            if depth not in arm_by_depth:
                continue
            raw, imp = arm_predictions(arm, mod, tree, arm_by_depth[depth], cost, args)
            preds = {
                TGIF: {l: set(map(int, tpt["summit"].get(l, []))) for l in tree.leaves},
                CALLER: {l: rb.intervals_to_boundaries(raw.get(l, set()), params.n_bins)
                         for l in tree.leaves},
                IMPUTE: {l: rb.intervals_to_boundaries(imp.get(l, set()), params.n_bins)
                         for l in tree.leaves},
            }
            for name, pred in preds.items():
                m = rb.score_boundaries(tree, pred, truth_bd, args.tolerance, None)
                rows.append({
                    "tree": tree_name, "region": region, "depth_label": depth,
                    "depth_value": tpt["depth_value"], "method": name,
                    "n_bins_na": tpt["n_bins_na"],
                    "tgif_status": tpt.get("status", "ok"), **m,
                })
        print(f"{tree_name}/{region_stem} done", flush=True)

    if not rows:
        raise SystemExit(f"no (tree, region) pair had both a TGIF cache and {arm} calls")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Filtering, mirroring the arm exactly
# ---------------------------------------------------------------------------
def apply_arm_filters(df: pd.DataFrame, pm, protected: set[str]) -> pd.DataFrame:
    """The arm's own depth exclusion and min-metric rule, on boundary rows.

    ``protected`` names the lines the min-metric filter must NOT touch. The arm
    exempts its baseline; TGIF is exempted for the same reason -- filtering only
    one side of a comparison manufactures it.
    """
    excl = {e.strip() for e in pm.EXCLUDE_DEPTHS_DEFAULT if e.strip()}
    if excl:
        before = len(df)
        df = df[~df["depth_label"].isin(excl)]
        print(f"  depth exclusion {sorted(excl)}: dropped {before - len(df)} rows")

    mm = pm.MIN_METRIC_DEFAULT
    if mm and mm > 0:
        bad = (~df["method"].isin(protected)) & (
            (df["precision"] < mm) | (df["recall"] < mm) | (df["f1"] < mm))
        print(f"  min-metric {mm:g}: dropped {int(bad.sum())} tree-informed rows "
              f"(baseline and TGIF exempt)")
        df = df[~bad]
    return df


def categorical_depths(df: pd.DataFrame) -> pd.DataFrame:
    order = [d for d in DEPTH_ORDER_HINT if d in set(df["depth_label"])]
    df = df.copy()
    df["depth_label"] = pd.Categorical(df["depth_label"], categories=order, ordered=True)
    return df


def band_series(df: pd.DataFrame, depths, method: str, band: str, metric: str = "f1"):
    grp = df[df["method"] == method].groupby("depth_label", observed=True)[metric]
    mean = grp.mean().reindex(depths)
    spread = (grp.std() if band == "sd" else grp.sem()).reindex(depths)
    xs = [i for i, d in enumerate(depths) if not pd.isna(mean.get(d))]
    ys = [float(mean[depths[i]]) for i in xs]
    es = [0.0 if pd.isna(spread.get(depths[i]))
          else BAND_SCALE * float(spread[depths[i]]) for i in xs]
    return xs, ys, es


def draw(ax, xs, ys, es, style, label):
    ax.plot(xs, ys, label=label, **style)
    ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                    color=style["color"], alpha=0.18, lw=0)


def finish(ax, fig, depths, ylabel, title, band_label, out_path):
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9,
              title=band_label, title_fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: the arm's own figure, with TGIF overlaid
# ---------------------------------------------------------------------------
def plot_interval_overlay(arm: str, pm, bdf: pd.DataFrame, cost: float,
                          out_path: Path, band: str) -> None:
    """The arm's curves, untouched, plus TGIF.

    The arm's own load/filter/aggregate/series functions are called, so the two
    non-TGIF lines are the SAME numbers its f1_vs_noise.png draws -- nothing is
    recomputed and no filter is second-guessed.
    """
    cfg = ARMS[arm]
    rows = pm.load(Path(cfg["dir"]) / "results")
    rows = pm.drop_depths(rows, pm.EXCLUDE_DEPTHS_DEFAULT)
    rows, dropped = pm.filter_impute_rows(rows, pm.MIN_METRIC_DEFAULT)
    print(f"  arm figure: min-metric {pm.MIN_METRIC_DEFAULT:g} dropped {dropped} rows; "
          f"excluded depths {pm.EXCLUDE_DEPTHS_DEFAULT or 'none'}")
    agg = pm.aggregate(rows)
    depth_pairs = pm.ordered_depths(agg)
    labels = [dl for _, dl in depth_pairs]

    headline = (pm.headline_method(agg, cost) if arm == "ontad"
                else pm.named(agg, "wsnap", cost))
    if headline is None:
        raise SystemExit(f"{arm}: no tree-informed method at cost {cost:g} in results/")

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for method, style, label in (
            (pm.RAW, STYLE[CALLER], cfg["caller_label"]),
            (headline, STYLE[IMPUTE], "TADLineage imputation")):
        xs, ys, es = pm.series(agg, depth_pairs, method, pm.PLOT_METRIC, band)
        es = [BAND_SCALE * e for e in es]   # match band_series()
        if xs:
            draw(ax, xs, ys, es, style, label)

    # TGIF, on the depths this arm plots. Boundary F1 -- flagged in the label,
    # because the two lines above are interval F1 and the axis cannot be both.
    tg = categorical_depths(bdf[(bdf["method"] == TGIF)
                                & (bdf["depth_label"].isin(labels))])
    xs, ys, es = band_series(tg, labels, TGIF, band)
    if xs:
        draw(ax, xs, ys, es, STYLE[TGIF],
             "TGIF-DB (tree-regularized NMF) — BOUNDARY F1")

    n_regions = tg[["tree", "region"]].drop_duplicates().shape[0]
    finish(ax, fig, labels, "mean leaf F1",
           "Tree-informed TAD inference at the leaves, vs TGIF-DB",
           BAND_LABEL[band], out_path)


# ---------------------------------------------------------------------------
# Figure 2: all three on boundary F1
# ---------------------------------------------------------------------------
def plot_boundary(arm: str, pm, bdf: pd.DataFrame, cost: float,
                  out_path: Path, band: str) -> None:
    cfg = ARMS[arm]
    print("  boundary figure:")
    df = categorical_depths(apply_arm_filters(bdf, pm, protected={CALLER, TGIF}))
    depths = list(df["depth_label"].cat.categories)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for method, label in ((CALLER, cfg["caller_label"]),
                          (IMPUTE, "TADLineage imputation"),
                          (TGIF, "TGIF-DB (tree-regularized NMF)")):
        xs, ys, es = band_series(df, depths, method, band)
        if xs:
            draw(ax, xs, ys, es, STYLE[method], label)

    n_regions = df[["tree", "region"]].drop_duplicates().shape[0]
    n_trees = df["tree"].nunique()
    finish(ax, fig, depths, "mean leaf boundary F1",
           "Tree-informed TAD inference at the leaves, vs TGIF-DB",
           BAND_LABEL[band], out_path)


def main() -> None:
    d = config.DEFAULT_PARAMS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--tgif-calls", type=Path, default=EXP_DIR / "calls")
    p.add_argument("--outdir", type=Path, help="defaults to <arm>/plots")
    p.add_argument("--band", choices=["sd", "sem"], default="sd")
    p.add_argument("--impute-cost", type=float,
                   help="defaults to that arm's HEADLINE_COST (1.0 ontad, 0.5 robustad)")
    p.add_argument("--robustad-prefix", default="wsnap")
    p.add_argument("--recompute", action="store_true",
                   help="re-solve the boundary-level scores instead of reusing "
                        "the cached CSV (the only step that needs gurobipy)")

    p.add_argument("--n-nodes", type=int, default=16)
    p.add_argument("--tree-seed", type=int, default=20240101)
    p.add_argument("--n-bins", type=int, default=500)
    p.add_argument("--n-top-tads", type=int, default=10)
    p.add_argument("--nesting-depth", type=int, default=2)
    p.add_argument("--min-size", type=int, default=8)
    p.add_argument("--min-score-node-tads", type=int, default=6)
    p.add_argument("--events-per-edge", type=int, default=1)
    p.add_argument("--tolerance", type=int, default=d.tolerance)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--Ct", type=float, default=1.0)
    p.add_argument("--Cb", type=float, default=1.0)
    p.add_argument("--solve-mode", dest="solve_mode", default="convex")
    p.add_argument("--importance-weight", type=float, default=0.1)
    p.add_argument("--price-discount", type=float, default=0.6)
    p.add_argument("--price-floor", type=float, default=0.25)
    args = p.parse_args()

    cfg = ARMS[args.arm]
    cost = args.impute_cost if args.impute_cost is not None else cfg["headline_cost"]
    outdir = args.outdir or (Path(cfg["dir"]) / "plots")
    outdir.mkdir(parents=True, exist_ok=True)
    pm = __import__(cfg["plot_module"])

    csv = outdir / "f1_vs_noise_with_tgif_boundary.csv"
    if csv.exists() and not args.recompute:
        print(f"reusing boundary scores from {csv} (--recompute to redo the solve)")
        bdf = pd.read_csv(csv)
    else:
        bdf = compute_boundary_rows(args.arm, args, cost)
        bdf.to_csv(csv, index=False)
        print(f"wrote {csv}")

    print(f"\n{args.arm}: filters from {cfg['plot_module']} "
          f"(min-metric {pm.MIN_METRIC_DEFAULT:g}, "
          f"excluded {pm.EXCLUDE_DEPTHS_DEFAULT or 'none'})")

    interval_png = outdir / "f1_vs_noise_with_tgif_interval.png"
    boundary_png = outdir / "f1_vs_noise_with_tgif_boundary.png"
    plot_interval_overlay(args.arm, pm, bdf, cost, interval_png, args.band)
    plot_boundary(args.arm, pm, bdf, cost, boundary_png, args.band)

    print(f"\nwrote {interval_png}")
    print(f"wrote {boundary_png}")


if __name__ == "__main__":
    main()
