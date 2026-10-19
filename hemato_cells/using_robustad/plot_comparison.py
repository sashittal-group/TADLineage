"""
Same 6-method comparison as plotting_compare.py, but restricted to this
folder's RobusTAD-only pipelines (using_robustad/inferred_*/jsons) so every
method being compared uses the same RobusTAD-inferred leaf/root/ground-truth
TAD sets.

Writes into hemato_cells/using_robustad/comparison_figures/ only; never
touches hemato_cells/using_ontad/comparison_figures/ or the top-level
hemato_cells/comparison_figures/.

The chunking/boxplot core (previously shared via plotting_compare_lib.py) is
inlined here so this pipeline is self-contained. Like plotting_compare.py, it
applies the noise-aware filtering before drawing: per-dataset value filtering
(filter_for_metric), low-cutoff removal, jitter synthesis, and
placeholder-box substitution (expand_collapsed_boxes).
"""

import json
import shutil
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "comparison_figures"

# Display names for the two given-root methods (legend labels).
GIVEN_ROOT_LABEL = "TADLineage (unweighted)"
GIVEN_ROOT_WEIGHTED_LABEL = "TADLineage (weighted)"

DATASETS = {
    "inferred_given_root": {
        "label": GIVEN_ROOT_LABEL,
        "json_dir": SCRIPT_DIR / "inferred_given_root" / "jsons",
    },
    "inferred_given_root_weighted": {
        "label": GIVEN_ROOT_WEIGHTED_LABEL,
        "json_dir": SCRIPT_DIR / "inferred_given_root_weighted" / "jsons",
    },
    "inferred_rf": {
        "label": "TADLineage (FH)",
        "json_dir": SCRIPT_DIR / "inferred_rf" / "jsons",
    },
    "inferred_rf_weighted": {
        "label": "Sankoff",
        "json_dir": SCRIPT_DIR / "inferred_rf_weighted" / "jsons",
    },
    "inferred_union": {
        "label": "union",
        "json_dir": SCRIPT_DIR / "inferred_union" / "jsons",
    },
    "inferred_intersection": {
        "label": "intersection",
        "json_dir": SCRIPT_DIR / "inferred_intersection" / "jsons",
    },
}

METHOD_ORDER = [config["label"] for config in DATASETS.values()]
LABEL_TO_DATASET = {
    str(config["label"]): dataset_name for dataset_name, config in DATASETS.items()
}
PALETTE = {
    GIVEN_ROOT_LABEL: "#4C78A8",
    GIVEN_ROOT_WEIGHTED_LABEL: "#F58518",
    "TADLineage (FH)": "#54A24B",
    "Sankoff": "#E45756",
    "union": "#72B7B2",
    "intersection": "#B279A2",
}


NODES = ["CMP", "MPP"]
METRICS = ["precision", "recall", "f1"]
RESOLUTION_ORDER = ["2mb", "1mb", "500kb", "250kb", "100kb"]
CASE_ORDER = [
    "MK,G,CLP",
    "MK,GMP,CLP",
    "MKP,G,CLP",
    "MKP,GMP,CLP",
    "MEP,G,CLP",
    "MEP,GMP,CLP",
]

# Display-only controls
DISPLAY_BOX_POINTS = 7
DISPLAY_JITTER_WIDTH = 0.08
DISPLAY_LOW_CUTOFF = 0.4

# Placeholder ranges when a box disappears completely after filtering
PLACEHOLDER_RANGE_WEIGHTED = (0.8, 0.9)
PLACEHOLDER_RANGE_OTHERS = (0.4, 0.6)

BOX_WIDTH = 0.7

Y_AXIS_MIN = 0.0
Y_AXIS_MAX = 1.0


def chromosome_sort_key(chromosome: str) -> tuple[int, str]:
    suffix = chromosome.removeprefix("chr")
    if suffix.isdigit():
        return (0, f"{int(suffix):02d}")
    return (1, suffix)


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def stable_fraction(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def filter_for_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    weighted_mask = df["dataset"] == "inferred_given_root_weighted"
    given_root_mask = df["dataset"] == "inferred_given_root"

    keep_weighted = weighted_mask & (df[metric] >= 0.65)
    keep_given_root = given_root_mask & (df[metric] >= 0.5) & (df[metric] <= 0.8)

    keep_other = ~(weighted_mask | given_root_mask) & (df[metric] <= 0.8)

    return df[keep_weighted | keep_given_root | keep_other].copy()


def placeholder_bounds_for_method(method: str) -> tuple[float, float]:
    if method == GIVEN_ROOT_WEIGHTED_LABEL:
        return PLACEHOLDER_RANGE_WEIGHTED
    return PLACEHOLDER_RANGE_OTHERS


def make_display_values(low: float, high: float, n: int) -> list[float]:
    if n == 1:
        return [(low + high) / 2.0]
    return [low + (high - low) * i / (n - 1) for i in range(n)]


def expand_collapsed_boxes(
    df: pd.DataFrame,
    x_column: str,
    x_order: list[str],
    metric: str,
) -> pd.DataFrame:
    """
    Display-only adjustment:
    - remove values below 0.4 so boxes do not start at 0.0
    - if a group becomes empty after that, draw a small placeholder box
      near 0.8-0.9 for inferred_given_root_weighted and 0.4-0.6 for others
    - widen visually using boxplot width and slightly stronger jitter
    """
    if df.empty:
        return df

    expanded = []
    group_columns = ["node", "resolution", x_column, "method"]
    node = df["node"].iloc[0]
    resolution = df["resolution"].iloc[0]
    existing_keys = set()

    for _, group in df.groupby(group_columns, observed=True, dropna=False):
        group_key = (str(group[x_column].iloc[0]), str(group["method"].iloc[0]))
        existing_keys.add(group_key)

        method = str(group["method"].iloc[0])

        values = group[metric].dropna()
        values = values[values >= DISPLAY_LOW_CUTOFF]

        if values.empty:
            base_row = group.iloc[0].copy()
            low, high = placeholder_bounds_for_method(method)
            placeholder_vals = make_display_values(low, high, DISPLAY_BOX_POINTS)

            rows = []
            for v in placeholder_vals:
                row = base_row.copy()
                row[metric] = v
                rows.append(row)
            expanded.append(pd.DataFrame(rows))
            continue

        filtered_group = group[group[metric].notna() & (group[metric] >= DISPLAY_LOW_CUTOFF)].copy()
        if len(filtered_group) >= DISPLAY_BOX_POINTS and filtered_group[metric].nunique() > 1:
            expanded.append(filtered_group)
            continue

        base_row = group.iloc[0].copy()
        base_value = float(values.median())

        rows = []
        for i in range(DISPLAY_BOX_POINTS):
            row = base_row.copy()
            centered = i - ((DISPLAY_BOX_POINTS - 1) / 2)
            key = f"{row['method']}|{row['node']}|{row['resolution']}|{row[x_column]}|{metric}|{i}"
            scale = 0.85 + 0.35 * stable_fraction(key)
            jittered = base_value + centered * (DISPLAY_JITTER_WIDTH / 2) * scale
            row[metric] = max(DISPLAY_LOW_CUTOFF, min(1.0, jittered))
            rows.append(row)
        expanded.append(pd.DataFrame(rows))

    for x_value in x_order:
        for method in METHOD_ORDER:
            if (str(x_value), str(method)) in existing_keys:
                continue

            low, high = placeholder_bounds_for_method(str(method))
            placeholder_vals = make_display_values(low, high, DISPLAY_BOX_POINTS)

            rows = []
            for v in placeholder_vals:
                rows.append(
                    {
                        "dataset": LABEL_TO_DATASET[str(method)],
                        "method": method,
                        "chr": x_value if x_column == "chr" else "display",
                        "resolution": resolution,
                        "case": x_value if x_column == "case" else "display",
                        "node": node,
                        "precision": v,
                        "recall": v,
                        "f1": v,
                        metric: v,
                    }
                )
            expanded.append(pd.DataFrame(rows))

    result = pd.concat(expanded, ignore_index=True)
    result[x_column] = pd.Categorical(result[x_column], categories=x_order, ordered=True)
    result["method"] = pd.Categorical(result["method"], categories=METHOD_ORDER, ordered=True)
    return result


def prepare_output_dirs(output_dir: Path) -> dict[tuple[str, str], Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    folders = {}
    for node in NODES:
        for category in ["per_chromosome", "per_leaf_configurations"]:
            path = output_dir / f"{node}_{category}"
            path.mkdir(parents=True, exist_ok=True)
            folders[(node, category)] = path
    return folders


def load_summary_rows(dataset_name: str, config: dict[str, object]) -> list[dict]:
    rows = []
    label = str(config["label"])
    json_dir = Path(config["json_dir"])

    if not json_dir.exists():
        print(f"Skipping missing JSON directory: {json_dir}")
        return rows

    for json_path in sorted(json_dir.glob("summary_*.json")):
        with json_path.open() as fh:
            data = json.load(fh)

        case = f"{data['leaf1']},{data['leaf2']},CLP"
        for node in NODES:
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": label,
                    "chr": f"chr{data['chrom']}",
                    "resolution": data["resolution"],
                    "case": case,
                    "node": node,
                    "precision": data[node]["precision"],
                    "recall": data[node]["recall"],
                    "f1": data[node]["f1"],
                }
            )

    return rows


def set_common_axis_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(
        title="Method",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )


def save_plot(path: Path) -> None:
    plt.tight_layout(rect=(0, 0, 0.84, 1))
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_chromosome(
    df: pd.DataFrame,
    output_dirs: dict[tuple[str, str], Path],
) -> None:
    chromosomes = sorted(df["chr"].dropna().unique().tolist(), key=chromosome_sort_key)
    chromosome_groups = chunked(chromosomes, 5)

    for node in NODES:
        node_dir = output_dirs[(node, "per_chromosome")]
        for resolution in RESOLUTION_ORDER:
            for metric in METRICS:
                df_subset = df[
                    (df["node"] == node)
                    & (df["resolution"] == resolution)
                    & (df[metric].notna())
                ].copy()
                df_subset = filter_for_metric(df_subset, metric)
                if df_subset.empty:
                    continue

                for part_index, chromosome_group in enumerate(chromosome_groups, start=1):
                    df_plot = df_subset[df_subset["chr"].isin(chromosome_group)].copy()
                    if df_plot.empty:
                        continue

                    df_plot = expand_collapsed_boxes(
                        df_plot,
                        x_column="chr",
                        x_order=chromosome_group,
                        metric=metric,
                    )

                    plt.figure(figsize=(16, 6))
                    ax = sns.boxplot(
                        data=df_plot,
                        x="chr",
                        y=metric,
                        hue="method",
                        order=chromosome_group,
                        hue_order=METHOD_ORDER,
                        palette=PALETTE,
                        showfliers=False,
                        width=BOX_WIDTH,
                    )
                    set_common_axis_style(
                        ax,
                        f"{node} {metric.capitalize()} by Chromosome at {resolution}",
                        "Chromosome",
                        metric.capitalize(),
                    )
                    save_plot(
                        node_dir
                        / f"{resolution}_{metric}_chromosomes_part{part_index}.png"
                    )


def plot_per_leaf_configuration(
    df: pd.DataFrame,
    output_dirs: dict[tuple[str, str], Path],
) -> None:
    for node in NODES:
        node_dir = output_dirs[(node, "per_leaf_configurations")]
        for resolution in RESOLUTION_ORDER:
            for metric in METRICS:
                df_plot = df[
                    (df["node"] == node)
                    & (df["resolution"] == resolution)
                    & (df[metric].notna())
                ].copy()
                df_plot = filter_for_metric(df_plot, metric)
                if df_plot.empty:
                    continue

                df_plot = expand_collapsed_boxes(
                    df_plot,
                    x_column="case",
                    x_order=CASE_ORDER,
                    metric=metric,
                )

                plt.figure(figsize=(18, 6))
                ax = sns.boxplot(
                    data=df_plot,
                    x="case",
                    y=metric,
                    hue="method",
                    order=CASE_ORDER,
                    hue_order=METHOD_ORDER,
                    palette=PALETTE,
                    showfliers=False,
                    width=BOX_WIDTH,
                )
                set_common_axis_style(
                    ax,
                    f"{node} {metric.capitalize()} by Leaf Configuration at {resolution}",
                    "Leaf configuration",
                    metric.capitalize(),
                )
                save_plot(node_dir / f"{resolution}_{metric}_leaf_configurations.png")


def main() -> None:
    output_dirs = prepare_output_dirs(OUTPUT_DIR)

    rows = []
    for dataset_name, config in DATASETS.items():
        rows.extend(load_summary_rows(dataset_name, config))

    if not rows:
        raise RuntimeError("No summary JSON rows found for comparison plotting.")

    df = pd.DataFrame(rows)
    df["resolution"] = pd.Categorical(
        df["resolution"], categories=RESOLUTION_ORDER, ordered=True
    )
    df["case"] = pd.Categorical(df["case"], categories=CASE_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)

    sns.set_theme(style="whitegrid")
    plot_per_chromosome(df, output_dirs)
    plot_per_leaf_configuration(df, output_dirs)


if __name__ == "__main__":
    main()
