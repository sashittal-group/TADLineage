#!/usr/bin/env python3
"""Aggregate and plot the TGIF-DB benchmark.

The headline figure is deliberately the SAME figure as
``../tree_informed_tad_inference_robustad/plots/f1_vs_noise.png`` -- same
layout, same styling, same axis labels, same ±1 SD ribbon, same file name --
with TGIF-DB added as an extra line. The two can be put side by side in a talk
or a paper without the reader having to re-learn the axes.

One thing IS different, and it is the whole point of this folder: the metric.

    the arm:   F1 over TAD INTERVALS a leaf recovers
    here:      F1 over BOUNDARIES a leaf recovers from its noisy Hi-C

That substitution is forced, not chosen. TGIF-DB emits per-bin boundary calls
and never assembles nested intervals, so intervals are not a currency it can pay
in; boundaries are the only common ground. Every method on the figure is scored
the same way -- an interval (a, b) contributes the two interface positions
{a-1, b}, and predictions are matched to truth one-to-one within ±tolerance.
Interval-level F1 is still carried through to the CSVs as ``f1_interval`` for
the methods that have intervals, so the arm's own numbers stay reachable.

Figures
-------
  f1_vs_noise.png          HEADLINE: mean leaf boundary F1 down the noise ladder.
                           Mirrors the arm's figure of the same name.
  f1_vs_noise_masked.png   the same, restricted to the bins TGIF evaluated.
  precision_recall.png     P and R separately. Worth drawing because the methods
                           fail in OPPOSITE directions and the F1 mean hides it.
  tgif_coverage.png        fraction of bins TGIF marked N/A per rung. Read this
                           BEFORE the F1 plots: where coverage collapses, a low
                           TGIF F1 is a refusal to call, not a wrong call.

CSVs
----
  leaf_metrics_long.csv    every per-(tree, region, depth, method) point
  leaf_metrics_summary.csv pooled mean / sd / sem / n per (depth, method)
  paired_gain.csv          per-rung paired differences, against BOTH the no-tree
                           baseline (what the tree buys) and TGIF (who wins where)

Run in the snakemake-tutorial env (pandas + matplotlib).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAW = "robustad"
TGIF = "tgif"

# Styling is copied from the arm's plot_tree_impute_rt.py on purpose: robustad,
# snap and wsnap keep the exact colours, markers, line widths and z-orders they
# have there, so a reader who knows that figure recognises these lines
# immediately. TGIF is the one new thing on the plot, so it gets the one colour
# that is not already spoken for.
METHOD_STYLE = {
    RAW:         dict(color="#4c4c4c", marker="o", lw=2.0, ls="-", zorder=5),
    "snap":      dict(color="#7f9bb5", marker="s", lw=2.0, ls="--", zorder=6),
    "wsnap":     dict(color="#2b6cb0", marker="D", lw=2.6, ls="-", zorder=7),
    TGIF:        dict(color="#c0392b", marker="^", lw=2.4, ls="-", zorder=8),
    "tgif_wide": dict(color="#e8a49c", marker="v", lw=1.6, ls=":", zorder=4),
}
METHOD_LABEL = {
    RAW:         "RobusTAD on the leaf (no tree)",
    "snap":      "tree-informed ILP, unweighted",
    "wsnap":     "tree-informed ILP, RobusTAD-weighted",
    TGIF:        "TGIF-DB (tree-regularized NMF)",
    "tgif_wide": "TGIF-DB, full stretches",
}
# Draw order only; anything unrecognised is appended.
METHOD_ORDER = ["tgif_wide", TGIF, RAW, "snap", "wsnap"]

# The ribbon is HALF a standard deviation, not a full one: at 200 regions the
# full-SD bands overlap so heavily that the lines are hard to separate. The
# legend names the multiplier, so nothing is implied about significance.
BAND_SCALE = 0.5
BAND_LABEL = {
    "sd": f"line thickness = ±{BAND_SCALE:g} SD across regions",
    "sem": f"line thickness = ±{BAND_SCALE:g} SEM of the mean",
}

XLABEL = "Hi-C sequencing depth (cleanest → noisiest)"


def load(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*/region_*.json")):
        rec = json.loads(path.read_text())
        for pt in rec["points"]:
            rows.append({
                "tree": rec["tree"], "region": rec["region"],
                "tolerance": rec["tolerance"], "n_leaves": rec["n_leaves"],
                **pt,
            })
    if not rows:
        raise SystemExit(f"no result JSONs under {results_dir}")
    df = pd.DataFrame(rows)
    # The depth ladder is an ordering, not a number to place on a linear axis:
    # the rungs are drawn evenly spaced, cleanest first, exactly as in the arm.
    order = (df[["depth_label", "depth_value"]].drop_duplicates()
             .sort_values("depth_value", ascending=False)["depth_label"].tolist())
    df["depth_label"] = pd.Categorical(df["depth_label"], categories=order, ordered=True)
    return df.sort_values(["tree", "region", "depth_label"])


def present_methods(df: pd.DataFrame, wanted: list[str] | None = None) -> list[str]:
    """Methods to DRAW, in a fixed order.

    ``wanted`` restricts the figures only -- every method stays in the CSVs and
    in the paired-gain report either way. The default (TGIF alone) exists
    because this figure is shown next to the arm's own f1_vs_noise.png rather
    than in place of it.
    """
    have = set(df["method"])
    if wanted:
        missing = [m for m in wanted if m not in have]
        if missing:
            print(f"WARNING: asked to plot {missing}, not present in the results")
        have &= set(wanted)
    return [m for m in METHOD_ORDER if m in have] + sorted(have - set(METHOD_ORDER))


def series(df: pd.DataFrame, depths: list[str], method: str, metric: str,
           band: str) -> tuple[list[int], list[float], list[float]]:
    """x positions, per-depth means, and the half-width of the spread ribbon.

    Rungs where the method has no scorable point (masked scoring returns NaN
    when TGIF evaluated nothing) are dropped rather than plotted as zero.
    """
    grp = df[df["method"] == method].groupby("depth_label", observed=True)[metric]
    mean = grp.mean().reindex(depths)
    spread = (grp.std() if band == "sd" else grp.sem()).reindex(depths)
    xs, ys, es = [], [], []
    for i, d in enumerate(depths):
        if pd.isna(mean.get(d)):
            continue
        xs.append(i)
        ys.append(float(mean[d]))
        es.append(0.0 if pd.isna(spread.get(d)) else BAND_SCALE * float(spread[d]))
    return xs, ys, es


def plot_f1(df: pd.DataFrame, metric: str, title: str, ylabel: str,
            out_path: Path, band: str, draw: list[str]) -> None:
    """The headline figure, laid out exactly like the arm's f1_vs_noise.png."""
    depths = list(df["depth_label"].cat.categories)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))

    for method in draw:
        style = METHOD_STYLE.get(method, {})
        xs, ys, es = series(df, depths, method, metric, band)
        if not xs:
            continue
        ax.plot(xs, ys, label=METHOD_LABEL.get(method, method), **style)
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)],
                        color=style.get("color", "#888888"), alpha=0.18, lw=0)

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9,
              title=BAND_LABEL[band], title_fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_precision_recall(df: pd.DataFrame, out_path: Path, band: str,
                          draw: list[str]) -> None:
    """P and R side by side.

    This is not decoration. The two approaches fail in opposite directions --
    TGIF keeps calling as the data degrades and loses precision, RobusTAD stops
    calling and loses recall -- and a mean F1 curve hides exactly that.
    """
    depths = list(df["depth_label"].cat.categories)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    for ax, metric, name in ((axes[0], "precision_full", "precision"),
                             (axes[1], "recall_full", "recall")):
        for method in draw:
            style = METHOD_STYLE.get(method, {})
            xs, ys, es = series(df, depths, method, metric, band)
            if not xs:
                continue
            ax.plot(xs, ys, label=METHOD_LABEL.get(method, method), **style)
            ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                            [y + e for y, e in zip(ys, es)],
                            color=style.get("color", "#888888"), alpha=0.18, lw=0)
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels(depths)
        ax.set_xlabel(XLABEL)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, ls=":")
        ax.set_title(f"Boundary {name} at the leaves", fontsize=11)
    axes[0].set_ylabel("mean leaf boundary precision / recall")
    axes[1].legend(fontsize=8, loc="lower left", framealpha=0.9,
                   title=BAND_LABEL[band], title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_coverage(df: pd.DataFrame, out_path: Path) -> None:
    """Fraction of bins TGIF refused to evaluate, per rung.

    This figure decides how the F1 figures may be described. TGIF's sparsity
    filter drops a bin if ANY leaf is too sparse there, so on thinned matrices
    it can decline to answer over most of the region -- and past a point it
    cannot run at all. A near-zero F1 under near-total N/A coverage is a refusal
    to call, not a wrong call, and the two must not be reported as the same
    failure.
    """
    one = df[df["method"] == TGIF].copy()
    if one.empty:
        return
    one["na_frac"] = one["n_bins_na"] / one["n_bins"]
    depths = list(df["depth_label"].cat.categories)
    grp = one.groupby("depth_label", observed=True)["na_frac"]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(range(len(depths)), grp.mean().reindex(depths).values,
           yerr=grp.sem().reindex(depths).values, capsize=3,
           color=METHOD_STYLE[TGIF]["color"], alpha=0.85)
    if "tgif_status" in one.columns:
        refused = (one.assign(bad=one["tgif_status"] != "ok")
                   .groupby("depth_label", observed=True)["bad"].sum().reindex(depths))
        for i, d in enumerate(depths):
            n = int(refused.get(d, 0) or 0)
            if n:
                ax.text(i, 1.02, f"{n} refused", ha="center", fontsize=8,
                        color=METHOD_STYLE[TGIF]["color"])
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("fraction of bins TGIF marked N/A")
    ax.grid(alpha=0.25, ls=":", axis="y")
    ax.set_title("TGIF coverage: bins dropped by its sparsity filter\n"
                 "(read before the F1 figures — high N/A means TGIF declined, "
                 "not that it was wrong)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def paired_gain(df: pd.DataFrame, reference: str, metric: str) -> pd.DataFrame:
    """Per-rung mean difference of each method against ``reference``, PAIRED.

    Every method scores the same (tree, region, depth) leaf sets, so the paired
    difference has a far smaller standard error than the gap between two
    independent means. This is the arm's convention and the reason its gains are
    quotable at all; unpaired SEMs would understate every effect here too.
    """
    key = ["tree", "region", "depth_label"]
    wide = df.pivot_table(index=key, columns="method", values=metric, observed=True)
    if reference not in wide:
        return pd.DataFrame()
    blocks = []
    for method in wide.columns:
        if method == reference:
            continue
        diff = (wide[method] - wide[reference]).dropna()
        if diff.empty:
            continue
        g = diff.groupby(level="depth_label", observed=True)
        blocks.append(pd.DataFrame({
            "method": method, "reference": reference, "metric": metric,
            "n": g.size(), "mean_gain": g.mean(), "sem": g.sem(),
        }).reset_index())
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def report(gains: pd.DataFrame, reference: str, metric: str) -> None:
    sub = gains[(gains["reference"] == reference) & (gains["metric"] == metric)]
    if sub.empty:
        return
    print(f"\npaired {metric} gain over {reference}:")
    for method, block in sub.groupby("method"):
        cells = " ".join(f"{r.depth_label}:{r.mean_gain:+.3f}±{r.sem:.3f}"
                         for r in block.itertuples())
        print(f"  {method:11s} {cells}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--outdir", type=Path, default=Path("plots"))
    p.add_argument("--band", choices=["sd", "sem"], default="sd",
                   help="what the ribbon around each line shows; 'sd' (the arm's "
                        "default) is the spread across regions, 'sem' is the "
                        "uncertainty in the mean")
    p.add_argument("--plot-methods", default=TGIF,
                   help="comma-separated methods to DRAW, or 'all'. Defaults to "
                        "TGIF alone, because this figure is meant to be shown "
                        "next to the arm's own f1_vs_noise.png rather than to "
                        "redraw it. Everything is scored and written to the CSVs "
                        "regardless of this flag")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load(args.results_dir)
    n_regions = df[["tree", "region"]].drop_duplicates().shape[0]
    n_trees = df["tree"].nunique()
    wanted = (None if str(args.plot_methods).strip().lower() == "all"
              else [m.strip() for m in str(args.plot_methods).split(",") if m.strip()])
    draw = present_methods(df, wanted)
    print(f"loaded {n_regions} (tree, region) pairs over {n_trees} trees, "
          f"{len(df)} rows")
    print(f"scored: {present_methods(df)}\ndrawn:  {draw}")

    # Sample size and matching tolerance are NOT on the figure any more (they
    # cluttered it); they live in leaf_metrics_summary.csv and this stdout line.
    lead = (METHOD_LABEL.get(draw[0], draw[0]) if len(draw) == 1
            else "Boundary recovery at the leaves")
    plot_f1(df, "f1_full", lead, "mean leaf boundary F1",
            args.outdir / "f1_vs_noise.png", args.band, draw)
    plot_f1(df, "f1_masked", f"{lead}, restricted to bins TGIF evaluated",
            "mean leaf boundary F1",
            args.outdir / "f1_vs_noise_masked.png", args.band, draw)
    plot_precision_recall(df, args.outdir / "precision_recall.png", args.band, draw)
    plot_coverage(df, args.outdir / "tgif_coverage.png")

    df.to_csv(args.outdir / "leaf_metrics_long.csv", index=False)
    metrics = [c for c in df.columns
               if c.endswith(("_full", "_masked")) or c == "f1_interval"]
    (df.groupby(["depth_label", "method"], observed=True)[metrics]
       .agg(["mean", "std", "sem", "count"])
       .to_csv(args.outdir / "leaf_metrics_summary.csv"))

    blocks = [g for g in (paired_gain(df, RAW, "f1_full"),
                          paired_gain(df, TGIF, "f1_full"),
                          paired_gain(df, TGIF, "f1_masked")) if not g.empty]
    if blocks:
        gains = pd.concat(blocks, ignore_index=True)
        gains.to_csv(args.outdir / "paired_gain.csv", index=False)
        report(gains, RAW, "f1_full")     # what the tree buys over the caller
        report(gains, TGIF, "f1_full")    # who wins where, against TGIF

    print(f"\nwrote figures and CSVs to {args.outdir}")


if __name__ == "__main__":
    main()
