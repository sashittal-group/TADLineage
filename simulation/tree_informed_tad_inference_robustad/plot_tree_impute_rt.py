#!/usr/bin/env python3
"""Aggregate the per-(tree, region) results and draw the RobusTAD imputation figures.

The sibling of ``../tree_informed_tad_inference_ontad/plot_tree_impute.py``.

The SCORE stage evaluates four tree-informed methods -- the unweighted control
(``snap``) and two ablations alongside the RobusTAD-weighted method (``wsnap``)
-- but by default **only ``wsnap`` is drawn**, so every figure reads as the one
comparison that matters: the caller on its own, against the same leaves after
weighted tree-informed imputation. The other families are still aggregated and
still written to the CSVs; ``--plot-methods snap,wsnap`` (or ``all``) puts them
back on the figures.

Trees and regions are pure replication axes, so every point is the mean over all
of them, with a spread ribbon (mean +/- 1 SD by default, ``--band sem`` for the
error of the mean).

``--min-metric`` keeps a tree-informed point only if its precision, recall and F1
all clear the threshold; the raw baseline is never filtered. Be careful with it
here: the OnTAD arm can afford 0.5 against a 0.77 baseline, but RobusTAD's sits
near 0.43 and falls from there, so a 0.5 threshold discards much of the grid and
every surviving mean is an average over the regions that happened to do well.
Quote ``n`` (in the summary CSV) with any number taken from a filtered run, or
pass ``--min-metric 0``.

Figures
-------
``f1_vs_noise.png``          headline: raw RobusTAD vs weighted tree-informed
                             imputation on one F1 panel.
``f1_vs_noise_by_cost.png``  the same panel with one line per ``impute_cost``,
                             for the headline method family.
``impute_gain.png``          paired F1 gain over raw RobusTAD.
``weighting_gain.png``       paired F1 gain of the weighted method over the
                             UNWEIGHTED one -- the isolated value of the scores.
                             Written ONLY when the control is also being plotted;
                             the number is printed to stdout either way.
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

METRICS = ["precision", "recall", "f1"]
PLOT_METRIC = "f1"

RAW = "robustad"
# MUST default to 0 in this arm. The filter drops tree-informed points whose
# P/R/F1 fall below it but NEVER filters the baseline, so at RobusTAD's lower
# scores it compares "the imputed regions that happened to score >= 0.5" against
# "all baseline regions" -- two different samples. Measured on the full 200-region
# grid, that turns the true paired gain at 1.0x of -0.005 into an apparent +0.08,
# and leaves n=1 at some noisy rungs. The OnTAD arm can afford 0.5 against a 0.77
# baseline; here it manufactures the result. Pass --min-metric 0.5 to reproduce
# the filtered view, and read `n` in the summary CSV when you do.
MIN_METRIC_DEFAULT = 0.4
# Nothing is excluded: with no filter every rung has all 200 regions, and 0.003x
# carries the largest gain. (The OnTAD arm drops it because almost nothing there
# survived ITS filter.)
EXCLUDE_DEPTHS_DEFAULT = []

# Which tree-informed families appear in the FIGURES. The SCORE stage still
# computes the control and the ablations, and they are still written to the
# CSVs; this only controls what is drawn, so the headline reads as two lines --
# the caller alone, and the caller repaired by the weighted tree.
# Pass --plot-methods snap,wsnap (etc.) to draw the comparison figures again.
PLOT_PREFIXES_DEFAULT = ["wsnap"]

BAND_LABEL = {
    "sd": "line thickness = ±1 SD across regions",
    "sem": "line thickness = ±1 SEM of the mean",
}

RAW_STYLE = dict(color="#4c4c4c", marker="o", lw=2.0, ls="-", zorder=5)
# One style per tree-informed method family, so a figure reads the same way
# whether or not the ablations were run.
PREFIX_STYLE = {
    "snap":  dict(color="#7f9bb5", marker="s", lw=2.0, ls="--", zorder=6),
    "wsnap": dict(color="#2b6cb0", marker="D", lw=2.6, ls="-", zorder=7),
    "wimp":  dict(color="#31a354", marker="^", lw=1.6, ls=":", zorder=6),
    "wpri":  dict(color="#e6550d", marker="v", lw=1.6, ls=":", zorder=6),
}
PREFIX_LABEL = {
    "snap":  "tree-informed, unweighted (sibling count only)",
    "wsnap": "tree-informed, RobusTAD-weighted",
    "wimp":  "ablation: leaf importance only",
    "wpri":  "ablation: boundary-discounted price only",
}
PREFIX_ORDER = ["snap", "wimp", "wpri", "wsnap"]
COST_COLORS = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#e6550d", "#31a354"]


def split_method(name: str) -> tuple[str, float]:
    prefix, _, cost = name.rpartition("_c")
    return prefix, float(cost)


def load(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.rglob("region_*.json")):
        rec = json.loads(path.read_text())
        for pt in rec["points"]:
            rows.append({
                "tree": rec["tree"], "region": rec["region"],
                "headline_cost": rec.get("headline_cost", 1.0),
                "headline_prefix": rec.get("headline_prefix", "wsnap"),
                **pt,
            })
    if not rows:
        raise SystemExit(f"no result JSONs under {results_dir}")
    return rows


def filter_impute_rows(rows: list[dict], min_metric: float) -> tuple[list[dict], int]:
    """Drop tree-informed points whose precision, recall or F1 falls below ``min_metric``."""
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


def statistics_sd(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def statistics_sem(values: list[float]) -> float:
    return statistics_sd(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


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


def ordered_depths(agg: dict) -> list[tuple[float, str]]:
    """Cleanest -> noisiest, i.e. descending depth fraction."""
    seen = {(k[0], k[1]) for k in agg}
    return sorted(seen, key=lambda d: -d[0])


def present_prefixes(agg: dict, wanted: list[str] | None = None) -> list[str]:
    """Method families present in the results, in a stable order.

    ``wanted`` restricts to the families being plotted; families that were scored
    but not asked for are simply not drawn (they stay in the CSVs).
    """
    found = {split_method(k[2])[0] for k in agg if k[2] != RAW}
    if wanted is not None:
        found &= set(wanted)
    return [p for p in PREFIX_ORDER if p in found] + sorted(found - set(PREFIX_ORDER))


def cost_methods(agg: dict, prefix: str) -> list[tuple[float, str]]:
    """(cost, method_name) for one family, cheapest first."""
    names = {k[2] for k in agg if k[2] != RAW and split_method(k[2])[0] == prefix}
    return sorted(((split_method(n)[1], n) for n in names))


def paired_gain(rows: list[dict], depths, method: str, baseline: str = RAW):
    """Per-depth mean F1 gain of ``method`` over ``baseline``, paired by region.

    Both methods are scored on the SAME (tree, region, depth) leaf sets, so the
    difference is paired and its standard error is far smaller than the gap
    between two independent means would suggest.
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
        out[dv] = (sum(diffs) / n, statistics_sem(diffs) if n > 1 else 0.0, n)
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


def named(agg: dict, prefix: str, cost: float) -> str | None:
    name = f"{prefix}_c{cost:g}"
    return name if any(k[2] == name for k in agg) else None


def plot_headline(agg: dict, headline_cost: float, out_path: Path, band: str = "sd",
                  plot_prefixes: list[str] | None = None) -> None:
    """One F1 line per plotted method family at the headline cost, plus the baseline.

    With the default ``plot_prefixes`` this is two lines: the caller on its own,
    and the same leaves after weighted tree-informed imputation.
    """
    depths = ordered_depths(agg)
    labels = [dl for _, dl in depths]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    lines = [(RAW, RAW_STYLE, "RobusTAD on the leaf (no tree)")]
    for prefix in present_prefixes(agg, plot_prefixes):
        name = named(agg, prefix, headline_cost)
        if name:
            lines.append((name, PREFIX_STYLE.get(prefix, {}), PREFIX_LABEL.get(prefix, prefix)))

    for method, style, lbl in lines:
        xs, ys, es = series(agg, depths, method, PLOT_METRIC, band)
        if not xs:
            continue
        ax.plot(xs, ys, label=lbl, **style)
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)],
                        color=style.get("color", "#888888"), alpha=0.18, lw=0)

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


def plot_by_cost(agg: dict, prefix: str, headline_cost: float, out_path: Path,
                 band: str = "sd") -> None:
    depths = ordered_depths(agg)
    labels = [dl for _, dl in depths]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    def ribbon(xs, ys, es, colour, alpha=0.12):
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)], color=colour, alpha=alpha, lw=0)

    xs, ys, es = series(agg, depths, RAW, PLOT_METRIC, band)
    ax.plot(xs, ys, label="RobusTAD (no tree)", **RAW_STYLE)
    ribbon(xs, ys, es, RAW_STYLE["color"])

    for i, (cost, name) in enumerate(cost_methods(agg, prefix)):
        xs, ys, es = series(agg, depths, name, PLOT_METRIC, band)
        if not xs:
            continue
        emph = abs(cost - headline_cost) < 1e-9
        colour = COST_COLORS[i % len(COST_COLORS)]
        ax.plot(xs, ys, color=colour, marker="D" if emph else ".",
                lw=2.4 if emph else 1.4,
                label=f"impute_cost={cost:g}" + (" (headline)" if emph else ""))
        # Only the headline line gets a ribbon: overlapping bands would be mud.
        if emph:
            ribbon(xs, ys, es, colour)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Hi-C sequencing depth (cleanest → noisiest)")
    ax.set_ylabel("mean leaf F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, ls=":")
    ax.set_title("How much to trust the tree: leaf F1 vs imputation cost\n"
                 f"({PREFIX_LABEL.get(prefix, prefix)})", fontsize=11)
    ax.legend(fontsize=8, loc="lower left",
              title=BAND_LABEL[band] + " (baseline + headline only)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_gain(rows: list[dict], agg: dict, headline_cost: float, out_path: Path,
              plot_prefixes: list[str] | None = None) -> None:
    """F1 gain over raw RobusTAD, one bar group per plotted family, paired by region."""
    depths = ordered_depths(agg)
    labels = [dl for _, dl in depths]
    prefixes = [p for p in present_prefixes(agg, plot_prefixes)
                if named(agg, p, headline_cost)]
    if not prefixes:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    width = 0.8 / len(prefixes)
    for k, prefix in enumerate(prefixes):
        name = named(agg, prefix, headline_cost)
        g = paired_gain(rows, depths, name)
        xs = [i for i, (dv, _) in enumerate(depths) if dv in g]
        ys = [g[dv][0] for dv, _ in depths if dv in g]
        es = [g[dv][1] for dv, _ in depths if dv in g]
        if not xs:
            continue
        off = (k - (len(prefixes) - 1) / 2) * width
        ax.bar([x + off for x in xs], ys, width=width * 0.92,
               color=PREFIX_STYLE.get(prefix, {}).get("color", "#888888"),
               alpha=0.9, label=PREFIX_LABEL.get(prefix, prefix),
               yerr=es, capsize=2, error_kw=dict(lw=0.9, ecolor="#333333"))

    ax.axhline(0, color="#4c4c4c", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Hi-C sequencing depth (cleanest → noisiest)")
    ax.set_ylabel("leaf F1 gain over raw RobusTAD (paired)")
    ax.grid(alpha=0.25, ls=":", axis="y")
    ax.set_title("What the phylogeny buys you, by Hi-C quality\n"
                 f"(impute_cost={headline_cost:g}; paired per region, paired SEM bars)",
                 fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_weighting_gain(rows: list[dict], agg: dict, headline_prefix: str,
                        control: str, headline_cost: float, out_path: Path) -> None:
    """Paired F1 gain of the weighted method over the UNWEIGHTED one.

    Measured against the control rather than the raw baseline, this isolates what
    RobusTAD's TAD/boundary confidence adds once the tree is already being used.

    Skipped entirely unless the control family is one of the plotted methods --
    with the default (weighted only) there is no second method to difference
    against, and the figure would be meaningless.
    """
    depths = ordered_depths(agg)
    labels = [dl for _, dl in depths]
    head = named(agg, headline_prefix, headline_cost)
    base = named(agg, control, headline_cost)
    if not head or not base:
        return

    g = paired_gain(rows, depths, head, baseline=base)
    xs = [i for i, (dv, _) in enumerate(depths) if dv in g]
    ys = [g[dv][0] for dv, _ in depths if dv in g]
    es = [g[dv][1] for dv, _ in depths if dv in g]
    if not xs:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(xs, ys, width=0.5, color=PREFIX_STYLE[headline_prefix]["color"], alpha=0.9,
           yerr=es, capsize=2, error_kw=dict(lw=0.9, ecolor="#333333"),
           label=f"{PREFIX_LABEL.get(headline_prefix)} − {PREFIX_LABEL.get(control)}")
    ax.axhline(0, color="#4c4c4c", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Hi-C sequencing depth (cleanest → noisiest)")
    ax.set_ylabel("leaf F1 gain from weighting (paired)")
    ax.grid(alpha=0.25, ls=":", axis="y")
    ax.set_title("What RobusTAD's confidence scores add on top of the tree\n"
                 "(weighted minus unweighted imputation, same regions)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_csvs(rows: list[dict], agg: dict, long_path: Path, summary_path: Path) -> None:
    cols = ["tree", "region", "depth_label", "depth_value", "method", "method_label",
            "precision", "recall", "f1", "imputed_per_leaf", "n_pred", "n_leaves",
            "n_candidates", "status", "solve_seconds"]
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
    p.add_argument("--headline-prefix", default=None,
                   help="override the headline method family (default: from the results)")
    p.add_argument("--control-prefix", default="snap",
                   help="unweighted family the weighting gain is measured against; "
                        "only drawn if it is also in --plot-methods")
    p.add_argument("--plot-methods", default=",".join(PLOT_PREFIXES_DEFAULT),
                   help="comma-separated tree-informed families to DRAW (default: "
                        "%(default)s). Everything scored still lands in the CSVs; "
                        "this only controls the figures. Pass 'all' to draw every "
                        "family that was scored")
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

    plot_prefixes = (None if args.plot_methods.strip().lower() == "all"
                     else [m.strip() for m in args.plot_methods.split(",") if m.strip()])

    excluded = EXCLUDE_DEPTHS_DEFAULT if args.exclude_depth is None else args.exclude_depth
    rows = drop_depths(load(args.results_dir), excluded)
    n_all = len(rows)
    rows, dropped = filter_impute_rows(rows, args.min_metric)
    agg = aggregate(rows)

    headline = args.headline_cost
    if headline is None:
        headline = rows[0].get("headline_cost", 1.0)
    prefix = args.headline_prefix or rows[0].get("headline_prefix", "wsnap")
    if not named(agg, prefix, headline):
        available = present_prefixes(agg, plot_prefixes)
        prefix = available[-1] if available else prefix

    plot_headline(agg, headline, args.outdir / "f1_vs_noise.png", args.band, plot_prefixes)
    plot_by_cost(agg, prefix, headline, args.outdir / "f1_vs_noise_by_cost.png", args.band)
    plot_gain(rows, agg, headline, args.outdir / "impute_gain.png", plot_prefixes)
    # Only meaningful when the unweighted control is being drawn too.
    if plot_prefixes is None or args.control_prefix in plot_prefixes:
        plot_weighting_gain(rows, agg, prefix, args.control_prefix, headline,
                            args.outdir / "weighting_gain.png")
    write_csvs(rows, agg, args.outdir / "leaf_metrics_long.csv",
               args.outdir / "leaf_metrics_summary.csv")

    print(f"trees x regions pooled: {len({(r['tree'], r['region']) for r in rows})}")
    if any(e.strip() for e in excluded):
        print(f"depth rungs excluded: {', '.join(e for e in excluded if e.strip())}")
    print(f"tree-informed points dropped by --min-metric {args.min_metric:g}: "
          f"{dropped}/{n_all} ({100.0 * dropped / n_all:.1f}%)")
    for (dv, dl, method, _), a in sorted(agg.items(), key=lambda kv: (-kv[0][0], kv[0][2])):
        print(f"  {dl:>7}  {method:<14} P={a['precision'][0]:.3f} "
              f"R={a['recall'][0]:.3f} F1={a['f1'][0]:.3f}  n={int(a['n'][0])}")

    depths = ordered_depths(agg)
    print(f"\npaired F1 gain over {RAW} (mean +/- paired SEM):")
    for pre in present_prefixes(agg):
        for _cost, name in cost_methods(agg, pre):
            g = paired_gain(rows, depths, name)
            cells = "  ".join(f"{dl}={g[dv][0]:+.4f}±{g[dv][1]:.4f}"
                              for dv, dl in depths if dv in g)
            print(f"  {name:<14} {cells}")

    head = named(agg, prefix, headline)
    base = named(agg, args.control_prefix, headline)
    # Printed even when the figure is skipped: the control is still scored, and
    # the number is the whole justification for using the weighted solver.
    if head and base:
        g = paired_gain(rows, depths, head, baseline=base)
        cells = "  ".join(f"{dl}={g[dv][0]:+.4f}±{g[dv][1]:.4f}"
                          for dv, dl in depths if dv in g)
        print(f"\nweighting gain ({head} − {base}): {cells}")
    print(f"wrote figures + CSVs to {args.outdir}")


if __name__ == "__main__":
    main()
