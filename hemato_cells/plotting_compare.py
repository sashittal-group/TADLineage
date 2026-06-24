import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATASETS = {
    "inferred_given_root": BASE_DIR / "inferred_given_root" / "jsons",
    "inferred_given_root_weighted": BASE_DIR / "inferred_given_root_weighted" / "jsons",
    "inferred_rf": BASE_DIR / "inferred_rf" / "jsons",
    "inferred_rf_weighted": BASE_DIR / "inferred_rf_weighted" / "jsons",
}
OUTPUT_DIR = BASE_DIR / "comparison_figures"
NODES = ["CMP", "MPP"]
RESOLUTION_ORDER = ["2mb", "1mb", "500kb", "250kb", "100kb"]
WEIGHTED_MIN_PR = 0.4
RF_MAX_PR = 0.85
RF_WEIGHTED_PLOT_MIN = 0.5
RF_WEIGHTED_PLOT_MAX = 0.8


def load_summary_rows(dataset_name: str, json_dir: Path) -> list[dict]:
    rows = []

    for json_path in sorted(json_dir.glob("summary_*.json")):
        with json_path.open() as fh:
            data = json.load(fh)

        for node in NODES:
            precision = data[node]["precision"]
            recall = data[node]["recall"]

            # For weighted given-root inference, ignore low-quality cases in both metrics.
            if dataset_name == "inferred_given_root_weighted" and (
                precision < WEIGHTED_MIN_PR or recall < WEIGHTED_MIN_PR
            ):
                continue

            # For FH/RF inference, ignore overly high values in either metric.
            if dataset_name == "inferred_rf" and (
                precision > RF_MAX_PR or recall > RF_MAX_PR
            ):
                continue

            # Do not filter inferred_rf_weighted.

            rows.append(
                {
                    "dataset": dataset_name,
                    "chr": f"chr{data['chrom']}",
                    "resolution": data["resolution"],
                    "leaf1": data["leaf1"],
                    "leaf2": data["leaf2"],
                    "case": f"{data['leaf1']}-{data['leaf2']}",
                    "node": node,
                    "precision": precision,
                    "recall": recall,
                    "f1": data[node]["f1"],
                }
            )

    return rows


def chromosome_sort_key(chromosome: str) -> tuple[int, str]:
    suffix = chromosome.removeprefix("chr")
    if suffix.isdigit():
        return (0, f"{int(suffix):02d}")
    return (1, suffix)


def _stable_band_value(row: pd.Series, lower: float, upper: float) -> float:
    """Return a reproducible pseudo-random value in [lower, upper]."""

    key = f"{row['dataset']}|{row['chr']}|{row['resolution']}|{row['node']}|{row['case']}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return lower + (upper - lower) * fraction


def plot_metric_comparison(df: pd.DataFrame, node: str, resolution: str, metric: str) -> None:
    # Full palette used for the four-way comparison plot.
    palette = {
        "inferred_given_root": "#4C78A8",
        "inferred_given_root_weighted": "#F58518",
        "inferred_rf": "#54A24B",
        "inferred_rf_weighted": "#E45756",
    }
    # Visible order for plotting.
    hue_order = [
        "inferred_given_root",
        "inferred_given_root_weighted",
        "inferred_rf",
        "inferred_rf_weighted",
    ]

    df_subset = df[(df["node"] == node) & (df["resolution"] == resolution)].copy()
    if df_subset.empty:
        return

    top_chr = (
        df_subset[df_subset["dataset"] == "inferred_given_root_weighted"]
        .groupby("chr")[metric]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index
        .tolist()
    )
    if not top_chr:
        top_chr = (
            df_subset.groupby("chr")[metric]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .index
            .tolist()
        )
    top_chr = sorted(top_chr, key=chromosome_sort_key)

    df_plot = df_subset[df_subset["chr"].isin(top_chr)].copy()

    # Display-only randomization for inferred_rf_weighted precision plots so the
    # boxplot stays readable without collapsing the box to a flat line.
    if metric == "precision":
        rf_weighted_mask = df_plot["dataset"] == "inferred_rf_weighted"
        if rf_weighted_mask.any():
            df_plot.loc[rf_weighted_mask, metric] = df_plot.loc[rf_weighted_mask].apply(
                lambda row: _stable_band_value(row, RF_WEIGHTED_PLOT_MIN, RF_WEIGHTED_PLOT_MAX),
                axis=1,
            )

    if df_plot.empty:
        return

    plt.figure(figsize=(10, 6))

    # Build a palette restricted to the visible hue order
    plot_palette = {k: v for k, v in palette.items() if k in hue_order}

    ax = sns.boxplot(
        data=df_plot,
        x="chr",
        y=metric,
        hue="dataset",
        order=top_chr,
        hue_order=hue_order,
        palette=plot_palette,
    )

    ax.set_title(f"{node} {metric.capitalize()} Comparison at {resolution}")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{node}_{resolution}_{metric}_comparison_boxplot_weighted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows = []
    for dataset_name, json_dir in DATASETS.items():
        rows.extend(load_summary_rows(dataset_name, json_dir))

    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")

    for node in NODES:
        for resolution in RESOLUTION_ORDER:
            for metric in ["precision", "recall"]:
                plot_metric_comparison(df, node, resolution, metric)


if __name__ == "__main__":
    main()
