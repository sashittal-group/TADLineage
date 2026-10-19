#!/usr/bin/env python3
"""Aggregate the data-quality sweep and draw F1-vs-depth-by-bin-size figures.

Pools every (tree, region) together -- trees and regions are just replication
axes -- and reports the *average* precision/recall/F1 over the scored internal
nodes. All three metrics go to CSV; only F1 is plotted, per the brief.

The sweep runs a single degradation axis (sequencing depth) at several bin sizes
(resolutions). Bin size is a counts-density knob: at fixed region depth, finer
bins pool fewer reads (counts/pixel ~ bin_kb^2), so coarse bins stay robust to
down-sampling while fine bins go count-poor and lose F1 as depth drops. Random
ligation noise was removed (Hi-C is itself noisy).

Two figures:

  f1_vs_depth_by_binsize.png             headline -- one panel per caller,
                                         given_root (honest reconstruction),
                                         one line per bin size across depth.
  f1_vs_depth_by_binsize_allmethods.png  full grid rows=caller x cols=method,
                                         one line per bin size.

Every line starts at the shared caller-free "perfect TADs" anchor (x = 0, the
reconstruction ceiling, bin-size independent) and then degrades across the depth
ladder; a dotted separator sets the perfect anchor apart from the Hi-C points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CACHE = SCRIPT_DIR / ".matplotlib_cache"
XDG_CACHE = SCRIPT_DIR / ".cache"
MPL_CACHE.mkdir(exist_ok=True)
XDG_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from run_sweep import DEPTH_LEVELS, METHOD_LABELS, METHODS

CALLER_LABELS = {"ontad": "OnTAD", "robustad": "RobusTAD"}
CALLER_ORDER = ["ontad", "robustad"]
HEADLINE_METHOD = "given_root"          # the honest reconstruction baseline
ROBUSTAD_GIVEN_ROOT_MIN_METRIC = 0.4
ROBUSTAD_WEIGHTED_MIN_METRIC = 0.5
METHODS_FIGURE_BIN_KB = 2.5             # fixed resolution for the by-method figure
METHODS_BAND_STD_SCALE = 0.5            # band half-width = this * across-replicate std


def load_rows(results_dir: Path) -> pd.DataFrame:
    """Flatten every region JSON into one long dataframe (one row per point)."""
    rows: list[dict] = []
    for json_path in sorted(results_dir.rglob("region_*.json")):
        record = json.loads(json_path.read_text())
        for point in record["points"]:
            rows.append(
                {
                    "tree": record["tree"],
                    "caller": record["caller"],
                    "region": record["region"],
                    "method": point.get("method"),
                    "method_label": point.get("method_label", point.get("method")),
                    "bin_label": point.get("bin_label"),
                    "bin_kb": point.get("bin_kb"),
                    "depth_label": point["depth_label"],
                    "depth_value": point.get("depth_value"),
                    "precision": point["precision"],
                    "recall": point["recall"],
                    "f1": point["f1"],
                    "n_scored": point.get("n_scored"),
                }
            )
    if not rows:
        raise SystemExit(f"No region_*.json found under {results_dir}")
    return pd.DataFrame(rows)


def filter_robustad_given_root_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only adequately recovered RobusTAD given-root replicates.

    This affects *only* real RobusTAD ``given_root`` and
    ``given_root_weighted`` points. The caller-free ``perfect`` anchor remains
    untouched. A replicate contributes to its plotted/summary mean only when
    all three of its internal-node-average metrics meet that method's floor.
    """
    thresholds = {
        "given_root": ROBUSTAD_GIVEN_ROOT_MIN_METRIC,
        "given_root_weighted": ROBUSTAD_WEIGHTED_MIN_METRIC,
    }
    target = (df["caller"] == "robustad") & df["method"].isin(thresholds)
    floors = df["method"].map(thresholds)
    passes = (
        df[["precision", "recall", "f1"]]
        .ge(floors, axis="index")
        .all(axis=1)
    )
    return df.loc[~target | passes].copy()


def _bin_sizes(df: pd.DataFrame) -> list[float]:
    """Resolutions present in the data, ordered coarse -> fine."""
    present = df.loc[df["bin_kb"].notna(), "bin_kb"].unique().tolist()
    return sorted(present, reverse=True)


def _depth_positions() -> tuple[list[int], list[str], dict[str, int]]:
    """Integer x positions and tick labels: perfect at 0, then the depth ladder."""
    labels = ["perfect"] + [lab for lab, _ in DEPTH_LEVELS]
    pos = list(range(len(labels)))
    depth_pos = {lab: i + 1 for i, (lab, _) in enumerate(DEPTH_LEVELS)}
    return pos, labels, depth_pos


def _line_df(df: pd.DataFrame, caller: str, method: str, bin_kb: float,
             depth_pos: dict[str, int]) -> pd.DataFrame:
    """Rows for one (caller, method, bin size) line, with the perfect anchor.

    The perfect anchor is caller- and bin-size-independent, so the same
    per-method anchors (x = 0) are attached to every resolution line -- each
    line starts at its own ceiling and then diverges across the depth ladder.
    """
    perfect = df[(df["depth_label"] == "perfect") & (df["method"] == method)][
        ["tree", "region", "f1"]
    ].copy()
    perfect["pos"] = 0

    sub = df[
        (df["caller"] == caller)
        & (df["method"] == method)
        & (df["bin_kb"] == bin_kb)
        & (df["depth_label"] != "perfect")
    ].copy()
    sub["pos"] = sub["depth_label"].map(depth_pos)

    cols = ["tree", "region", "pos", "f1"]
    return pd.concat([perfect[cols], sub[cols]], ignore_index=True)


def _bin_palette(bin_sizes: list[float]) -> dict[float, tuple]:
    """Coarse (robust) -> dark, fine (fragile) -> light, on a viridis ramp."""
    colors = sns.color_palette("viridis", len(bin_sizes))
    return dict(zip(bin_sizes, colors))


def _draw_panel(ax, df, caller, method, bin_sizes, palette, depth_pos,
                positions, labels) -> None:
    for bin_kb in bin_sizes:
        ldf = _line_df(df, caller, method, bin_kb, depth_pos)
        if ldf.empty:
            continue
        sns.lineplot(
            data=ldf, x="pos", y="f1", ax=ax, color=palette[bin_kb],
            marker="o", markersize=5, linewidth=1.7,
            estimator="mean", errorbar=("ci", 95), legend=False,
        )
    ax.axvline(0.5, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.3, positions[-1] + 0.3)


def _bin_legend(bin_sizes: list[float], palette: dict[float, tuple]):
    return [
        Line2D([0], [0], color=palette[b], marker="o", linewidth=1.7,
               label=f"{b:g} kb")
        for b in bin_sizes
    ]


def plot_headline(df: pd.DataFrame, output_path: Path) -> None:
    """One panel per caller: given_root F1 vs depth, one line per bin size."""
    sns.set_theme(style="whitegrid", context="paper")
    bin_sizes = _bin_sizes(df)
    palette = _bin_palette(bin_sizes)
    positions, labels, depth_pos = _depth_positions()

    n = len(CALLER_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.3), sharey=True, squeeze=False)
    for c, caller in enumerate(CALLER_ORDER):
        ax = axes[0][c]
        _draw_panel(ax, df, caller, HEADLINE_METHOD, bin_sizes, palette,
                    depth_pos, positions, labels)
        ax.set_title(CALLER_LABELS[caller])
        ax.set_xlabel("sequencing depth  (best → worst)")
        if c == 0:
            ax.set_ylabel(f"mean internal-node F1\n({METHOD_LABELS[HEADLINE_METHOD]})")

    fig.legend(handles=_bin_legend(bin_sizes, palette), loc="upper center",
               ncol=len(bin_sizes), frameon=False, title="bin size (resolution)",
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Internal-node reconstruction F1 vs. sequencing depth, by bin size",
                 y=1.11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods(df: pd.DataFrame, output_path: Path) -> None:
    """Full grid: rows = caller, cols = method, one line per bin size."""
    sns.set_theme(style="whitegrid", context="paper")
    bin_sizes = _bin_sizes(df)
    palette = _bin_palette(bin_sizes)
    positions, labels, depth_pos = _depth_positions()

    n_rows, n_cols = len(CALLER_ORDER), len(METHODS)
    fig, axgrid = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows),
        sharey=True, squeeze=False,
    )
    for r, caller in enumerate(CALLER_ORDER):
        for c, method in enumerate(METHODS):
            ax = axgrid[r][c]
            _draw_panel(ax, df, caller, method, bin_sizes, palette,
                        depth_pos, positions, labels)
            if r == 0:
                ax.set_title(PLOT_METHOD_LABELS[method])
            ax.set_xlabel("depth (best → worst)" if r == n_rows - 1 else "")
            ax.set_ylabel(f"{CALLER_LABELS[caller]}\nmean internal-node F1" if c == 0 else "")
            ax.tick_params(axis="x", labelrotation=90)

    fig.legend(handles=_bin_legend(bin_sizes, palette), loc="upper center",
               ncol=len(bin_sizes), frameon=False, title="bin size (resolution)",
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Internal-node reconstruction F1 vs. depth, by bin size (all methods)",
                 y=1.07)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Display names for the figures; overrides the internal method keys.
PLOT_METHOD_LABELS = {
    **METHOD_LABELS,
    "given_root": "TADLineage",
    "given_root_weighted": "TADLineage (weighted)",
}


def _method_palette(methods: list[str]) -> dict[str, tuple]:
    """One distinct qualitative color per reconstruction method."""
    colors = sns.color_palette("tab10", len(methods))
    return dict(zip(methods, colors))


def _method_legend(methods: list[str], palette: dict[str, tuple]):
    return [
        Line2D([0], [0], color=palette[m], marker="o", linewidth=2.0,
               label=PLOT_METHOD_LABELS[m])
        for m in methods
    ]


def _draw_method_panel(ax, df, caller, methods, palette, depth_pos,
                       positions, labels) -> None:
    """One line per method at the fixed bin size; ``±std`` shaded band.

    The band half-width is the across-replicate standard deviation of F1 at each
    depth, so a curve's *thickness* reads as its spread -- fat where replicates
    disagree, thin where they agree (the boxplot-style variance cue).
    """
    for method in methods:
        ldf = _line_df(df, caller, method, METHODS_FIGURE_BIN_KB, depth_pos)
        if ldf.empty:
            continue
        stats = (
            ldf.groupby("pos")["f1"]
            .agg(mean="mean", std="std")
            .reindex(positions)
            .dropna(subset=["mean"])
            .reset_index()
        )
        if stats.empty:
            continue
        half = stats["std"].fillna(0.0) * METHODS_BAND_STD_SCALE
        color = palette[method]
        ax.fill_between(
            stats["pos"],
            (stats["mean"] - half).clip(lower=0.0),
            (stats["mean"] + half).clip(upper=1.0),
            color=color, alpha=0.18, linewidth=0,
        )
        ax.plot(stats["pos"], stats["mean"], color=color, marker="o",
                markersize=5, linewidth=2.0)
    ax.axvline(0.5, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.3, positions[-1] + 0.3)


def plot_methods_at_binsize(df: pd.DataFrame, output_path: Path) -> None:
    """One panel per caller: F1 vs depth at 2.5 kb, one line per method.

    Six method curves per panel, colored distinctly; each curve carries a
    ``±std`` band so its width shows the across-replicate variance.
    """
    sns.set_theme(style="whitegrid", context="paper")
    positions, labels, depth_pos = _depth_positions()
    palette = _method_palette(METHODS)

    n = len(CALLER_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.3), sharey=True, squeeze=False)
    for c, caller in enumerate(CALLER_ORDER):
        ax = axes[0][c]
        _draw_method_panel(ax, df, caller, METHODS, palette, depth_pos,
                           positions, labels)
        ax.set_title(CALLER_LABELS[caller])
        ax.set_xlabel("sequencing depth  (best → worst)")
        if c == 0:
            ax.set_ylabel("mean internal-node F1")

    fig.legend(handles=_method_legend(METHODS, palette), loc="upper center",
               ncol=len(METHODS), frameon=False, title="method",
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Internal-node reconstruction F1 vs. sequencing depth, by method",
        y=1.11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summaries(df: pd.DataFrame, outdir: Path) -> None:
    df.to_csv(outdir / "sweep_metrics_long.csv", index=False)
    summary = (
        df.groupby(["caller", "method", "bin_label", "bin_kb", "depth_label", "depth_value"],
                   observed=True, dropna=False)[["precision", "recall", "f1"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(str(x) for x in c).rstrip("_") for c in summary.columns]
    summary.to_csv(outdir / "sweep_metrics_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "plots")
    args = parser.parse_args()

    raw_df = load_rows(args.results_dir)
    df = filter_robustad_given_root_methods(raw_df)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_summaries(df, args.outdir)
    plot_headline(df, args.outdir / "f1_vs_depth_by_binsize.png")
    plot_all_methods(df, args.outdir / "f1_vs_depth_by_binsize_allmethods.png")
    plot_methods_at_binsize(df, args.outdir / "f1_vs_depth_methods_2p5kb.png")

    n_tr = df[["tree", "region"]].drop_duplicates().shape[0]
    n_bins = df.loc[df["bin_kb"].notna(), "bin_kb"].nunique()
    n_filtered = len(raw_df) - len(df)
    print(f"Aggregated {n_tr} (tree, region) replicates across "
          f"{df['tree'].nunique()} trees, {df['method'].nunique()} methods, "
          f"and {n_bins} bin sizes; wrote plots + CSVs under {args.outdir}")
    print(f"Excluded {n_filtered} RobusTAD given-root rows with at least one "
          "metric below their method-specific threshold.")


if __name__ == "__main__":
    main()
