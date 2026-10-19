"""
Dump the 250kb per-leaf-configuration values that survive the display
filtering used in plotting_compare.py (filter_for_metric), i.e. the data
actually summarised by the boxplots in fig:hemato_leaf of the paper.

Output: 250kb_displayed_values_after_filtering.txt
"""
from pathlib import Path

import pandas as pd

import plotting_compare as pc

RESOLUTION = "250kb"
METRICS = ["precision", "recall", "f1"]
OUT_PATH = Path(__file__).resolve().parent / "250kb_displayed_values_after_filtering.txt"


def build_dataframe() -> pd.DataFrame:
    rows = []
    for dataset_name, config in pc.DATASETS.items():
        rows.extend(pc.load_summary_rows(dataset_name, config))
    df = pd.DataFrame(rows)
    return df[df["resolution"] == RESOLUTION].copy()


def summarize(values: pd.Series) -> dict:
    return {
        "n": int(values.count()),
        "mean": values.mean(),
        "median": values.median(),
        "min": values.min(),
        "q1": values.quantile(0.25),
        "q3": values.quantile(0.75),
        "max": values.max(),
    }


def fmt_row(label: str, s: dict, width: int = 34) -> str:
    if s["n"] == 0:
        return f"  {label:<{width}} n=0   (no values survive filtering; box shown is a display placeholder)"
    return (
        f"  {label:<{width}} n={s['n']:<4} "
        f"mean={s['mean']:.3f}  median={s['median']:.3f}  "
        f"min={s['min']:.3f}  Q1={s['q1']:.3f}  Q3={s['q3']:.3f}  max={s['max']:.3f}"
    )


def main() -> None:
    df = build_dataframe()

    lines = []
    lines.append("Displayed value data (after filter_for_metric) for the 250kb")
    lines.append("per-leaf-configuration figure (fig:hemato_leaf).")
    lines.append("")
    lines.append("Filtering applied per method (from plotting_compare.filter_for_metric):")
    lines.append("  - inferred_given_root_weighted : keep value >= 0.6")
    lines.append("  - SankOff                       : keep 0.4 <= value <= 0.7")
    lines.append("  - all other methods             : keep value <= 0.8")
    lines.append("")
    lines.append("Statistics are computed on the surviving values only. Where a method x")
    lines.append("leaf-configuration cell has n=0, the box drawn in the figure is a synthetic")
    lines.append("display placeholder (see expand_collapsed_boxes), not real data.")
    lines.append("=" * 78)

    for metric in METRICS:
        df_m = df[df[metric].notna()].copy()
        df_f = pc.filter_for_metric(df_m, metric)

        for node in pc.NODES:
            df_node = df_f[df_f["node"] == node]
            lines.append("")
            lines.append(f"### {metric.upper()} — node {node} — {RESOLUTION}")
            lines.append("-" * 78)

            lines.append("Overall (all six leaf configurations pooled):")
            for method in pc.METHOD_ORDER:
                vals = df_node[df_node["method"] == method][metric].dropna()
                lines.append(fmt_row(method, summarize(vals)))

            lines.append("")
            lines.append("Per leaf configuration (x-axis of the figure):")
            for case in pc.CASE_ORDER:
                lines.append(f"  [{case}]")
                df_case = df_node[df_node["case"] == case]
                for method in pc.METHOD_ORDER:
                    vals = df_case[df_case["method"] == method][metric].dropna()
                    lines.append(fmt_row("    " + method, summarize(vals)))
            lines.append("=" * 78)

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
