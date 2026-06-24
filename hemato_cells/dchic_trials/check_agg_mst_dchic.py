#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "dchic_distances"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "msts"
NODES = ["LT", "ST", "MPP", "CMP", "CLP", "MEP", "GMP", "MKP", "G", "MK"]


class TreeNode:
    def __init__(self, label: str):
        self.label = label
        self.children = []


def build_reference_tree() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("LT", "ST"),
            ("ST", "MPP"),
            ("MPP", "CMP"),
            ("MPP", "CLP"),
            ("CMP", "MEP"),
            ("CMP", "GMP"),
            ("MEP", "MKP"),
            ("GMP", "G"),
            ("MKP", "MK"),
        ]
    )
    return graph


def nx_to_zss(root_label: str, graph: nx.Graph) -> TreeNode:
    node = TreeNode(root_label)
    if not hasattr(nx_to_zss, "visited"):
        nx_to_zss.visited = set()
    nx_to_zss.visited.add(root_label)
    for child in sorted(graph.neighbors(root_label)):
        if child not in nx_to_zss.visited:
            node.children.append(nx_to_zss(child, graph))
    return node


def tree_edit_distance(graph1: nx.Graph, graph2: nx.Graph, root: str = "LT") -> float:
    import zss

    nx_to_zss.visited = set()
    tree1 = nx_to_zss(root, graph1)
    nx_to_zss.visited = set()
    tree2 = nx_to_zss(root, graph2)
    return zss.simple_distance(
        tree1,
        tree2,
        get_children=lambda node: node.children,
        get_label=lambda node: node.label,
    )


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def chrom_sort_key(chrom: str):
    suffix = normalize_chrom(chrom)[3:]
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, suffix.lower())


def parse_resolution(value: str) -> int:
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(bp|kb|mb)?", text)
    if not match:
        raise ValueError(f"Invalid resolution: {value}")
    number = float(match.group(1))
    unit = match.group(2) or "bp"
    return int(number * {"bp": 1, "kb": 1_000, "mb": 1_000_000}[unit])


def resolution_label(resolution: str | int) -> str:
    if isinstance(resolution, str) and not resolution.isdigit():
        return resolution.lower()
    value = int(resolution)
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}mb"
    if value % 1_000 == 0:
        return f"{value // 1_000}kb"
    return f"{value}bp"


def find_pairwise_csvs(input_dir: Path, resolution: str, chroms: list[str] | None):
    res_label = resolution_label(parse_resolution(resolution))
    selected_chroms = {normalize_chrom(chrom) for chrom in chroms} if chroms else None
    paths = []
    prefix = "pairwise_dchic_distances_"
    for path in sorted(input_dir.glob(f"{prefix}*_{res_label}.csv")):
        stem = path.stem
        remainder = stem[len(prefix) :]
        chrom = normalize_chrom(remainder[: -(len(res_label) + 1)])
        if selected_chroms is None or chrom in selected_chroms:
            paths.append((chrom, path))
    if not paths:
        raise FileNotFoundError(f"No dchic pairwise CSVs found in {input_dir} for {res_label}")
    return sorted(paths, key=lambda item: chrom_sort_key(item[0]))


def read_distance_rows(path: Path, distance_column: str):
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"cell1", "cell2", distance_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            cell1 = row["cell1"].strip().upper()
            cell2 = row["cell2"].strip().upper()
            if not cell1 or not cell2 or cell1 == cell2:
                continue
            rows.append((cell1, cell2, float(row[distance_column])))
    return rows


def aggregate_distances(input_dir: Path, resolution: str, chroms: list[str] | None, distance_column: str, agg: str):
    edge_values = defaultdict(list)
    chrom_paths = find_pairwise_csvs(input_dir, resolution, chroms)
    for chrom, path in chrom_paths:
        print(f"Reading {path}")
        for cell1, cell2, distance in read_distance_rows(path, distance_column):
            edge_values[tuple(sorted((cell1, cell2)))].append(distance)

    summary_rows = []
    dist = {}
    for (cell1, cell2), values in sorted(edge_values.items()):
        distance = sum(values) if agg == "sum" else sum(values) / len(values)
        dist[(cell1, cell2)] = distance
        dist[(cell2, cell1)] = distance
        summary_rows.append(
            {
                "cell1": cell1,
                "cell2": cell2,
                "distance": distance,
                "num_files": len(values),
                "sum_distance": sum(values),
                "mean_distance": sum(values) / len(values),
                "distance_column": distance_column,
                "aggregation": agg,
            }
        )
    return dist, summary_rows, [chrom for chrom, _ in chrom_paths]


def build_mst(dist: dict[tuple[str, str], float]) -> nx.Graph:
    graph = nx.Graph()
    nodes = set()
    for cell1, cell2 in dist:
        nodes.add(cell1)
        nodes.add(cell2)
    ordered_nodes = [node for node in NODES if node in nodes] + sorted(nodes - set(NODES))
    graph.add_nodes_from(ordered_nodes)
    for (cell1, cell2), distance in dist.items():
        if cell1 < cell2:
            graph.add_edge(cell1, cell2, weight=distance)
    if not nx.is_connected(graph):
        raise ValueError(f"Distance graph is disconnected: {[sorted(c) for c in nx.connected_components(graph)]}")
    return nx.minimum_spanning_tree(graph)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def write_mst_edges(path: Path, mst: nx.Graph) -> None:
    rows = [{"cell1": u, "cell2": v, "weight": data["weight"]} for u, v, data in sorted(mst.edges(data=True))]
    write_csv(path, rows, ["cell1", "cell2", "weight"])


def plot_mst(mst: nx.Graph, path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(mst, seed=1, k=1.5)
    plt.figure(figsize=(10, 10))
    nx.draw(mst, pos, with_labels=True, node_size=2500, node_color="lightblue", font_size=10)
    labels = {(u, v): f"{data['weight']:.3g}" for u, v, data in mst.edges(data=True)}
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=labels, font_size=9, rotate=False)
    plt.title(title)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build genomewide MST from dchic compartment distances.")
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--chroms", nargs="+")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--distance-column", default="mean_abs_compartment_diff")
    parser.add_argument("--agg", choices=["mean", "sum"], default="mean")
    args = parser.parse_args()

    res_label = resolution_label(parse_resolution(args.resolution))
    chrom_label = "genomewide" if not args.chroms else "_".join(sorted(normalize_chrom(c) for c in args.chroms))
    dist, distance_rows, used_chroms = aggregate_distances(
        args.input_dir, res_label, args.chroms, args.distance_column, args.agg
    )
    mst = build_mst(dist)

    out_dir = args.output_dir / chrom_label / res_label / args.distance_column / args.agg
    write_csv(
        out_dir / "aggregated_pairwise_distances.csv",
        distance_rows,
        ["cell1", "cell2", "distance", "num_files", "sum_distance", "mean_distance", "distance_column", "aggregation"],
    )
    write_mst_edges(out_dir / "mst_edges.csv", mst)

    ted = tree_edit_distance(mst, build_reference_tree(), root="LT")
    write_csv(
        out_dir / "tree_distance.csv",
        [
            {
                "resolution": res_label,
                "distance_column": args.distance_column,
                "aggregation": args.agg,
                "chrom_scope": chrom_label,
                "tree_edit_distance": ted,
                "mst_built_from": "dchic_compartment_distances",
                "reference_used_for_inference": "FALSE",
            }
        ],
        [
            "resolution",
            "distance_column",
            "aggregation",
            "chrom_scope",
            "tree_edit_distance",
            "mst_built_from",
            "reference_used_for_inference",
        ],
    )
    with (out_dir / "chromosomes_used.txt").open("w") as handle:
        for chrom in used_chroms:
            handle.write(f"{chrom}\n")
    plot_mst(mst, out_dir / "mstree.png", f"dcHiC-style MST ({args.distance_column}, {args.agg}, {res_label})")


if __name__ == "__main__":
    main()
