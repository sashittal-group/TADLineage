"""
Per-resolution comparison of the six methods.

One figure per (node, metric), laid out like the per-leaf-configuration
figures: the bin size on the x-axis (2 Mb -> 100 kb), six boxes per bin size
(one per method), and the metric on the y-axis. Each box pools the
autosomes, the leaf configurations and the boundary-matching tolerances.

Applies the same noise-aware display pipeline as plot_comparison.py
(filter_for_metric, then expand_collapsed_boxes), grouping by resolution
instead of by chromosome or by leaf configuration, so this figure is
consistent with the other comparison_figures/ panels.

Reuses the dataset table and JSON loader of plot_comparison.py in this folder,
and writes only into comparison_figures/{CMP,MPP}_per_resolution/ (it never
deletes the other comparison_figures/ subfolders). Note that re-running
plot_comparison.py wipes comparison_figures/, so re-run this script after it.
"""

import sys
from pathlib import Path

import pandas as pd
import seaborn as sns

import plot_comparison as pc


def load_raw() -> pd.DataFrame:
    rows = []
    for dataset_name, config in pc.DATASETS.items():
        rows.extend(pc.load_summary_rows(dataset_name, config))
    if not rows:
        raise RuntimeError("No summary JSON rows found for per-resolution plotting.")
    df = pd.DataFrame(rows)
    df["resolution"] = pd.Categorical(df["resolution"], categories=pc.RESOLUTION_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=pc.METHOD_ORDER, ordered=True)
    return df


def plot_node_metric(df: pd.DataFrame, node: str, metric: str, out_path: Path, raw: bool = False) -> None:
    df_plot = df[(df["node"] == node) & df[metric].notna()].copy()
    if not raw:
        df_plot = pc.filter_for_metric(df_plot, metric)
        if df_plot.empty:
            return
        df_plot = pc.expand_collapsed_boxes(
            df_plot,
            x_column="resolution",
            x_order=pc.RESOLUTION_ORDER,
            metric=metric,
        )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 6))
    ax = sns.boxplot(
        data=df_plot,
        x="resolution",
        y=metric,
        hue="method",
        order=pc.RESOLUTION_ORDER,
        hue_order=pc.METHOD_ORDER,
        palette=pc.PALETTE,
        showfliers=False,
        width=pc.BOX_WIDTH,
    )
    pc.set_common_axis_style(
        ax,
        f"{node} {metric.capitalize()} by Bin Size",
        "Bin size",
        metric.capitalize(),
    )
    pc.save_plot(out_path)


def main() -> None:
    # --raw: skip the filtering and display adjustments, write to *_per_resolution_raw/
    raw = "--raw" in sys.argv[1:]
    df = load_raw()
    sns.set_theme(style="whitegrid")
    for node in pc.NODES:
        node_dir = pc.OUTPUT_DIR / f"{node}_per_resolution{'_raw' if raw else ''}"
        node_dir.mkdir(parents=True, exist_ok=True)
        for metric in pc.METRICS:
            plot_node_metric(df, node, metric, node_dir / f"{metric}_per_resolution.png", raw=raw)


if __name__ == "__main__":
    main()
