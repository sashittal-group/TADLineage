#!/usr/bin/env python3
"""Aggregate the per-(tree, region) results and draw the leaf-imputation figures.

Trees and regions are pure replication axes, so every point is the mean over all
of them (with a standard error band on the headline figure).

Every figure reports F1 only, and line thickness carries the spread across trees
x regions (mean +/- 1 SD by default, ``--band sem`` for the narrower error of
the mean).

Imputation points are filtered before anything is aggregated: a tree-informed
point is kept only if its precision, recall AND F1 are all >= ``--min-metric``
(default 0.5). Points below that are solver/scoring artefacts rather than real
imputations, and they drag the means down out of proportion to how often they
occur. The raw OnTAD baseline is never filtered -- it is the reference the gain
is measured against, so trimming it would bias the comparison. The noisiest rung
(``0.003x``) is dropped outright via ``--exclude-depth``: almost nothing there
survives the filter.

Figures
-------
``f1_vs_noise.png``            headline: two lines -- raw OnTAD and the same
                               leaves after tree-informed imputation -- on a
                               single F1 panel.
``f1_vs_noise_by_cost.png``    the same F1 panel with one line per ``impute_cost``,
                               showing that noisier data wants a cheaper (more
                               trusting) imputation.
``impute_gain.png``            F1 gain over raw OnTAD vs depth -- the "how much
                               does the tree buy you" summary.

CSVs (``leaf_metrics_long.csv`` / ``leaf_metrics_summary.csv``) carry every
surviving per-region point and the aggregated means, so the numbers behind the
figures can be re-checked without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Depth is plotted on an evenly spaced categorical axis (cleanest -> noisiest);
# the values span 3 orders of magnitude, so a linear axis would bunch them up.
METRICS = ["precision", "recall", "f1"]
PLOT_METRIC = "f1"

RAW = "ontad"
MIN_METRIC_DEFAULT = 0.5
# Depth rungs to leave off the figures entirely (matched on depth_label or value).
EXCLUDE_DEPTHS_DEFAULT = ["0.003x"]
# Ribbon thickness = spread across trees x regions. SD shows how variable a single
# region is; SEM shows how well-pinned the mean is and is ~14x narrower at n=200.
BAND_LABEL = {
    "sd": "line thickness = ±1 SD across regions",
    "sem": "line thickness = ±1 SEM of the mean",
}

RAW_STYLE = dict(color="#4c4c4c", marker="o", lw=2.0, ls="-", zorder=5)
HEADLINE_STYLE = dict(color="#2b6cb0", marker="D", lw=2.4, ls="-", zorder=6)
COST_COLORS = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#e6550d", "#31a354"]


def load(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.rglob("region_*.json")):
        rec = json.loads(path.read_text())
        for pt in rec["points"]:
            rows.append({
                "tree": rec["tree"], "region": rec["region"],
                "headline_cost": rec.get("headline_cost", 1.0),
                **pt,
            })
    if not rows:
        raise SystemExit(f"no result JSONs under {results_dir}")
    return rows


def filter_impute_rows(rows: list[dict], min_metric: float) -> tuple[list[dict], int]:
    """Drop tree-informed points whose precision, recall or F1 falls below ``min_metric``.

    The raw OnTAD rows pass through untouched: they are the baseline, so filtering
    them would move the reference line as well as the imputed one.
    """
    if min_metric <= 0:
        return rows, 0
    kept, dropped = [], 0
    for r in rows:
        if r["method"] != RAW and any(
                r.get(m) is None or float(r[m]) < min_metric for m in METRICS):
            dropped += 1
            continue
        kept.append(r)
    if not kept:
        raise SystemExit(f"every point was filtered out at --min-metric {min_metric:g}")
    return kept, dropped


def drop_depths(rows: list[dict], exclude: list[str]) -> list[dict]:
    """Remove whole depth rungs, matched on either the label ("0.003x") or the value."""
    if not exclude:
        return rows
    wanted = {e.strip() for e in exclude if e.strip()}
    kept = [r for r in rows
            if r["depth_label"] not in wanted and f"{r['depth_value']:g}" not in wanted]
    if not kept:
        raise SystemExit(f"every depth was excluded by {sorted(wanted)}")
    return kept


def aggregate(rows: list[dict]) -> dict:
    """(depth, method) -> {metric: (mean, sem, sd), ...} pooled over trees x regions."""
    buckets: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["depth_value"], r["depth_label"], r["method"], r["method_label"])
        for m in METRICS + ["imputed_per_leaf", "n_pred", "solve_seconds"]:
            if r.get(m) is not None:
                buckets[key][m].append(float(r[m]))
    out = {}
    for key, vals in buckets.items():
        out[key] = {
            m: (sum(v) / len(v), statistics_sem(v), statistics_sd(v))
            for m, v in vals.items() if v
        }
        out[key]["n"] = (len(next(iter(vals.values()))), 0.0, 0.0)
    return out


def statistics_sd(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def statistics_sem(values: list[float]) -> float:
    return statistics_sd(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def ordered_depths(agg: dict) -> list[tuple[float, str]]:
    """Cleanest -> noisiest, i.e. descending depth fraction."""
    seen = {(k[0], k[1]) for k in agg}
    return sorted(seen, key=lambda d: -d[0])


def cost_methods(agg: dict, prefix: str = "snap_c") -> list[tuple[float, str]]:
    """(cost, method_name) for every priced imputation run, cheapest first."""
    names = {k[2] for k in agg if k[2].startswith(prefix)}
    return sorted(((float(n.removeprefix(prefix)), n) for n in names))


def paired_gain(rows: list[dict], depths, method: str, baseline: str = RAW):
    """Per-depth mean F1 gain of ``method`` over ``baseline``, paired by region.

    Both methods are scored on the SAME (tree, region, depth) leaf sets, so the
    difference is paired and its standard error is much smaller than the gap
    between two independent means would suggest. Reporting the unpaired SEMs
    would understate a real effect.
    """
    by_key: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_key[(r["tree"], r["region"], r["depth_value"])][r["method"]] = r["f1"]

    out = {}
    for dv, _dl in depths:
        diffs = [v[method] - v[baseline] for k, v in by_key.items()
                 if k[2] == dv and method in v and baseline in v]
        if not diffs:
            continue
        n = len(diffs)
        mean = sum(diffs) / n
        sem = statistics_sem(diffs) if n > 1 else 0.0
        out[dv] = (mean, sem, n)
    return out


def series(agg: dict, depths, method: str, metric: str, band: str = "sd"):
    """x positions, means, and the half-width of the spread ribbon at each point."""
    idx = {"sd": 2, "sem": 1}[band]
    xs, ys, es = [], [], []
    for i, (dv, dl) in enumerate(depths):
        key = next((k for k in agg if k[0] == dv and k[2] == method), None)
        if key is None or metric not in agg[key]:
            continue
        xs.append(i)
        ys.append(agg[key][metric][0])
        es.append(agg[key][metric][idx])
    return xs, ys, es


def headline_method(agg: dict, headline_cost: float) -> str | None:
    name = f"snap_c{headline_cost:g}"
    return name if any(k[2] == name for k in agg) else None


def plot_headline(agg: dict, headline_cost: float, out_path: Path, n_leaves: int,
                  band: str = "sd") -> None:
    """Two F1 lines: raw OnTAD, and the same leaves after tree-informed imputation.

    The line is drawn as a ribbon whose thickness is the spread across regions,
    so how variable a method is reads off the figure directly.
    """
    depths = ordered_depths(agg)
    labels = [dl for _, dl in depths]
    head = headline_method(agg, headline_cost)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for method, style, lbl in ((RAW, RAW_STYLE, "OnTAD on the leaf (no tree)"),
                               (head, HEADLINE_STYLE, "tree-informed imputation")):
        if not method:
            continue
        xs, ys, es = series(agg, depths, method, PLOT_METRIC, band)
        if not xs:
            continue
        ax.plot(xs, ys, label=lbl, **style)
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)],
                        color=style["color"], alpha=0.22, lw=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Hi-C sequencing depth (cleanest → noisiest)")
    ax.set_ylabel("mean leaf F1 (over leaves and trees)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9,
              title=BAND_LABEL[band], title_fontsize=8)
    ax.set_title("Tree-informed TAD inference at the leaves", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_by_cost(agg: dict, headline_cost: float, out_path: Path, band: str = "sd") -> None:
    depths = ordered_depths(agg)
    labels = [dl for _, dl in depths]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    def ribbon(xs, ys, es, colour, alpha=0.12):
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)],
                        color=colour, alpha=alpha, lw=0)

    xs, ys, es = series(agg, depths, RAW, PLOT_METRIC, band)
    ax.plot(xs, ys, label="OnTAD (no tree)", **RAW_STYLE)
    ribbon(xs, ys, es, RAW_STYLE["color"])

    for i, (cost, name) in enumerate(cost_methods(agg)):
        xs, ys, es = series(agg, depths, name, PLOT_METRIC, band)
        if not xs:
            continue
        emph = abs(cost - headline_cost) < 1e-9
        colour = COST_COLORS[i % len(COST_COLORS)]
        ax.plot(xs, ys, color=colour,
                marker="D" if emph else ".", lw=2.4 if emph else 1.4,
                label=f"impute_cost={cost:g}" + (" (headline)" if emph else ""))
        # Only the headline line gets a ribbon: five overlapping bands would be mud.
        if emph:
            ribbon(xs, ys, es, colour)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Hi-C sequencing depth (cleanest → noisiest)")
    ax.set_ylabel("mean leaf F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, ls=":")
    ax.set_title("How much to trust the tree: leaf F1 vs imputation cost\n"
                 "(noisier Hi-C rewards a cheaper, more trusting imputation)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower left",
              title=BAND_LABEL[band] + " (baseline + headline only)",
              title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_gain(rows: list[dict], agg: dict, headline_cost: float, out_path: Path) -> None:
    """F1 gain over raw OnTAD, paired by (tree, region) with paired error bars."""
    depths = ordered_depths(agg)
    labels = [dl for _, dl in depths]
    head = headline_method(agg, headline_cost)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    bars = [(head, HEADLINE_STYLE["color"], 0.0, 0.9,
             f"tree-informed imputation (cost={headline_cost:g})")]
    for name, colour, off, alpha, lbl in bars:
        if not name:
            continue
        g = paired_gain(rows, depths, name)
        xs = [i for i, (dv, _) in enumerate(depths) if dv in g]
        ys = [g[dv][0] for dv, _ in depths if dv in g]
        es = [g[dv][1] for dv, _ in depths if dv in g]
        if not xs:
            continue
        ax.bar([x + off for x in xs], ys, width=0.5, color=colour,
               alpha=alpha, label=lbl,
               yerr=es, capsize=2, error_kw=dict(lw=0.9, ecolor="#333333"))

    ax.axhline(0, color="#4c4c4c", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Hi-C sequencing depth (cleanest → noisiest)")
    ax.set_ylabel("leaf F1 gain over raw OnTAD (paired)")
    ax.grid(alpha=0.25, ls=":", axis="y")
    ax.set_title("What the phylogeny buys you, by Hi-C quality\n"
                 "(paired per region; error bars are paired SEM)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_csvs(rows: list[dict], agg: dict, long_path: Path, summary_path: Path) -> None:
    cols = ["tree", "region", "depth_label", "depth_value", "method", "method_label",
            "precision", "recall", "f1", "imputed_per_leaf", "n_pred", "n_leaves",
            "n_candidates", "projected_candidates", "status", "solve_seconds"]
    with long_path.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else str(r.get(c, "")) for c in cols) + "\n")

    with summary_path.open("w") as fh:
        fh.write("depth_label,depth_value,method,method_label,n,"
                 "precision,recall,f1,f1_sem,f1_sd,imputed_per_leaf,solve_seconds\n")
        for (dv, dl, method, mlabel) in sorted(agg, key=lambda k: (-k[0], k[2])):
            a = agg[(dv, dl, method, mlabel)]
            def g(m, i=0):
                return f"{a[m][i]:.4f}" if m in a else ""
            fh.write(f"{dl},{dv},{method},\"{mlabel}\",{int(a['n'][0])},"
                     f"{g('precision')},{g('recall')},{g('f1')},{g('f1', 1)},{g('f1', 2)},"
                     f"{g('imputed_per_leaf')},{g('solve_seconds')}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--outdir", type=Path, default=Path("plots"))
    p.add_argument("--headline-cost", type=float, default=None,
                   help="override the headline impute_cost (default: from the results)")
    p.add_argument("--min-metric", type=float, default=MIN_METRIC_DEFAULT,
                   help="drop tree-informed points whose precision, recall or F1 is "
                        "below this (default: %(default)g); 0 disables the filter")
    p.add_argument("--exclude-depth", action="append", default=None,
                   help="depth rung to leave out entirely, by label or value; "
                        f"repeatable (default: {' '.join(EXCLUDE_DEPTHS_DEFAULT)}). "
                        "Pass --exclude-depth '' to keep every rung")
    p.add_argument("--band", choices=["sd", "sem"], default="sd",
                   help="what the line thickness shows: spread across regions (sd, "
                        "default) or precision of the mean (sem)")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    excluded = EXCLUDE_DEPTHS_DEFAULT if args.exclude_depth is None else args.exclude_depth
    rows = drop_depths(load(args.results_dir), excluded)
    n_all = len(rows)
    rows, dropped = filter_impute_rows(rows, args.min_metric)
    agg = aggregate(rows)
    headline = args.headline_cost
    if headline is None:
        headline = rows[0].get("headline_cost", 1.0)
    n_leaves = int(rows[0].get("n_leaves") or 0)

    plot_headline(agg, headline, args.outdir / "f1_vs_noise.png", n_leaves, args.band)
    plot_by_cost(agg, headline, args.outdir / "f1_vs_noise_by_cost.png", args.band)
    plot_gain(rows, agg, headline, args.outdir / "impute_gain.png")
    write_csvs(rows, agg, args.outdir / "leaf_metrics_long.csv",
               args.outdir / "leaf_metrics_summary.csv")

    print(f"trees x regions pooled: {len({(r['tree'], r['region']) for r in rows})}")
    if any(e.strip() for e in excluded):
        print(f"depth rungs excluded: {', '.join(e for e in excluded if e.strip())}")
    print(f"tree-informed points dropped by --min-metric {args.min_metric:g}: "
          f"{dropped}/{n_all} ({100.0 * dropped / n_all:.1f}%)")
    for (dv, dl, method, _), a in sorted(agg.items(), key=lambda kv: (-kv[0][0], kv[0][2])):
        print(f"  {dl:>7}  {method:<14} P={a['precision'][0]:.3f} "
              f"R={a['recall'][0]:.3f} F1={a['f1'][0]:.3f}")

    depths = ordered_depths(agg)
    head = headline_method(agg, headline)
    if head:
        print(f"\npaired F1 gain over {RAW} (mean +/- paired SEM, n regions):")
        for _cost, name in cost_methods(agg):
            g = paired_gain(rows, depths, name)
            cells = "  ".join(f"{dl}={g[dv][0]:+.4f}+/-{g[dv][1]:.4f}"
                              for dv, dl in depths if dv in g)
            print(f"  {name:<14} {cells}")
    print(f"wrote figures + CSVs to {args.outdir}")


if __name__ == "__main__":
    main()
