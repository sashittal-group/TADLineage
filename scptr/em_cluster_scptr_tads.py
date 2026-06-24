import argparse
import itertools
import json
import random
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import zss

# Allow imports from repository root when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from filtering import harmonize_leaf_boundaries
from nussinov import infer_tads_with_dp
from nussinov_with_imputation import infer_tads_with_dp as infer_tads_with_imputation
from tad_utilities import boundaries_from_tads, parse_interval, relaxed_boundary_rf, relaxed_tad_rf

STAGE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
STAGE_TO_ID = {
    "1-cell": "1",
    "2-cell": "2",
    "4-cell": "4",
    "8-cell": "8",
    "64-cell": "64",
}
STAGE_RANK = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}


class TreeNode:
    def __init__(self, label):
        self.label = label
        self.children = []


def build_reference_chain_tree():
    root = TreeNode("1")
    node2 = TreeNode("2")
    node4 = TreeNode("4")
    node8 = TreeNode("8")
    node64 = TreeNode("64")
    root.children = [node2]
    node2.children = [node4]
    node4.children = [node8]
    node8.children = [node64]
    return root


def normalize_tree_label(label):
    return STAGE_TO_ID.get(label, str(label))


def build_rooted_tree_from_mst(mst, root, node_labels):
    def build(node, parent=None):
        tree_node = TreeNode(normalize_tree_label(node_labels.get(node, str(node))))
        for neighbor in sorted(mst.neighbors(node), key=lambda value: str(value)):
            if neighbor == parent:
                continue
            tree_node.children.append(build(neighbor, node))
        return tree_node

    return build(root)


def compute_majority_labels(cluster_assignments, true_labels):
    cluster_to_labels = {}
    for cluster_id, label in zip(cluster_assignments, true_labels):
        cluster_to_labels.setdefault(cluster_id, []).append(label)

    majority_map = {}
    cluster_summary = {}
    for cluster_id, labels in cluster_to_labels.items():
        counts = Counter(labels)
        majority_label = max(
            counts.items(),
            key=lambda item: (item[1], -STAGE_RANK.get(item[0], len(STAGE_ORDER)))
        )[0]
        majority_map[cluster_id] = majority_label
        cluster_summary[cluster_id] = {
            "size": len(labels),
            "counts": dict(counts),
            "majority": majority_label,
            "majority_id": STAGE_TO_ID.get(majority_label, majority_label),
        }

    return majority_map, cluster_summary


def choose_mst_root(cluster_majorities):
    def rank(cluster_idx):
        label = cluster_majorities.get(cluster_idx, "")
        return (STAGE_RANK.get(label, len(STAGE_ORDER)), cluster_idx)

    return min(cluster_majorities.keys(), key=rank)


def compute_tree_distance_from_mst(mst, cluster_majorities):
    if mst.number_of_nodes() == 0:
        return float("nan")

    root = choose_mst_root(cluster_majorities)
    inferred_tree = build_rooted_tree_from_mst(mst, root, cluster_majorities)
    reference_tree = build_reference_chain_tree()

    return zss.simple_distance(
        inferred_tree,
        reference_tree,
        get_children=lambda node: node.children,
        get_label=lambda node: node.label,
    )


def write_run_json(output_path, row):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)


def read_tad_file_as_strings(path, start_col=0, end_col=1, sep=None):
    """Read a TAD file and return intervals as {'[start,end]', ...}."""
    intervals = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cols = line.split() if sep is None else line.split(sep)
            try:
                start = int(cols[start_col])
                end = int(cols[end_col])
            except Exception:
                # Skip headers or malformed lines.
                continue
            intervals.add(f"[{start},{end}]")
    return intervals


def tad_distance(intervals_a, intervals_b, ct=1.0, cb=1.0, tolerance=0):
    """Repo-wide relaxed distance: Ct * TAD RF + Cb * boundary RF."""

    def to_pairs(intervals):
        pairs = set()
        for item in intervals:
            if isinstance(item, tuple):
                pairs.add((int(item[0]), int(item[1])))
            else:
                try:
                    pairs.add(parse_interval(str(item)))
                except Exception:
                    nums = re.findall(r"-?\d+", str(item))
                    if len(nums) >= 2:
                        pairs.add((int(nums[0]), int(nums[1])))
        return pairs

    a = to_pairs(intervals_a)
    b = to_pairs(intervals_b)

    a_starts, a_ends = boundaries_from_tads(a)
    b_starts, b_ends = boundaries_from_tads(b)

    tad_rf = relaxed_tad_rf(a, b, tol=tolerance)
    boundary_rf = relaxed_boundary_rf(a_starts, b_starts, a_ends, b_ends, tol=tolerance)

    return ct * tad_rf + cb * boundary_rf


def compute_median_for_cluster(list_of_interval_sets, ct=1.0, cb=1.0):
    """Compute cluster median with nussinov DP."""
    m = len(list_of_interval_sets)
    if m == 0:
        return set(), 0.0

    edges = [(0, i + 1) for i in range(m)]
    leaf_intervals = {i + 1: deepcopy(list_of_interval_sets[i]) for i in range(m)}

    starts = set()
    ends = set()
    for interval_set in list_of_interval_sets:
        for item in interval_set:
            nums = re.findall(r"-?\d+", str(item))
            if len(nums) >= 2:
                starts.add(int(nums[0]))
                starts.add(int(nums[1]))
                ends.add(int(nums[0]))
                ends.add(int(nums[1]))

    med_intervals, med_score = infer_tads_with_dp(
        edges,
        leaf_intervals,
        starts,
        ends,
        Ct=ct,
        Cb=cb,
        trace=False,
    )

    return med_intervals, med_score


def em_cluster_tads(cells, k=2, ct=1.0, cb=1.0, max_iter=50, tol=1e-6, init="kmeans++", random_state=None, distance_tolerance=0):
    """EM-style clustering for TAD interval sets."""
    if random_state is not None:
        random.seed(random_state)

    n = len(cells)
    if n == 0:
        return [], [], {"obj": [], "labels": []}

    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, n)

    pairwise = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = tad_distance(cells[i], cells[j], ct=ct, cb=cb, tolerance=distance_tolerance)
            pairwise[i][j] = d
            pairwise[j][i] = d

    if init == "random":
        med_idx = random.sample(range(n), k)
    elif init == "kmeans++":
        med_idx = [random.randrange(n)]
        while len(med_idx) < k:
            d2 = []
            for i in range(n):
                if i in med_idx:
                    d2.append(0.0)
                else:
                    nearest = min(pairwise[i][c] for c in med_idx)
                    d2.append(nearest * nearest)
            total = sum(d2)
            if total == 0:
                choices = [i for i in range(n) if i not in med_idx]
                med_idx.append(random.choice(choices))
            else:
                r = random.random() * total
                cumulative = 0.0
                for i, val in enumerate(d2):
                    cumulative += val
                    if cumulative >= r and i not in med_idx:
                        med_idx.append(i)
                        break
    else:
        med_idx = list(range(k))

    medians = [deepcopy(cells[i]) for i in med_idx]
    labels = [-1] * n
    history = {"obj": [], "labels": []}

    print(f"Starting EM clustering: n={n}, k={k}, max_iter={max_iter}, init={init}", flush=True)

    for iteration in range(1, max_iter + 1):
        new_labels = [None] * n
        objective = 0.0

        for i in range(n):
            best_cluster = None
            best_dist = float("inf")
            for c_idx in range(k):
                d = tad_distance(cells[i], medians[c_idx], ct=ct, cb=cb, tolerance=distance_tolerance)
                if d < best_dist:
                    best_dist = d
                    best_cluster = c_idx
            new_labels[i] = best_cluster
            objective += best_dist

        new_medians = []
        for c_idx in range(k):
            members = [cells[i] for i in range(n) if new_labels[i] == c_idx]
            if not members:
                new_medians.append(deepcopy(cells[random.randrange(n)]))
                continue

            median_set, _ = compute_median_for_cluster(members, ct=ct, cb=cb)
            median_strs = set(f"[{a},{b}]" for (a, b) in median_set)
            new_medians.append(median_strs)

        history["obj"].append(objective)
        history["labels"].append(new_labels)

        cluster_counts = dict(Counter(new_labels))
        print(
            f"Iteration {iteration:03d}: objective={objective:.6f}, cluster_counts={cluster_counts}",
            flush=True,
        )

        if new_labels == labels:
            labels = new_labels
            medians = new_medians
            print(f"Converged at iteration {iteration:03d}: labels unchanged.", flush=True)
            break

        if len(history["obj"]) > 1 and abs(history["obj"][-2] - history["obj"][-1]) < tol:
            labels = new_labels
            medians = new_medians
            print(
                f"Converged at iteration {iteration:03d}: objective change < tol ({tol}).",
                flush=True,
            )
            break

        labels = new_labels
        medians = new_medians

    else:
        print(f"Reached max_iter={max_iter} without convergence.", flush=True)

    return labels, medians, history


def collect_scptr_files(base_dir, chromosome, resolution, stage_dirs):
    """Collect all tads.tad files for one chromosome/resolution across stage dirs."""
    base = Path(base_dir)
    records = []

    for stage in stage_dirs:
        stage_path = base / stage
        if not stage_path.exists():
            continue

        for sample_dir in sorted(stage_path.iterdir()):
            if not sample_dir.is_dir():
                continue
            tad_path = sample_dir / chromosome / resolution / "tads.tad"
            if tad_path.exists():
                records.append(
                    {
                        "path": str(tad_path),
                        "stage": stage,
                        "sample": sample_dir.name,
                    }
                )

    return records


def sample_records_per_stage(records, per_stage, random_state=None):
    """Sample up to per_stage records from each stage label."""
    rng = random.Random(random_state)
    grouped = {}
    for record in records:
        grouped.setdefault(record["stage"], []).append(record)

    sampled = []
    for stage in sorted(grouped):
        stage_records = grouped[stage][:]
        if len(stage_records) > per_stage:
            sampled.extend(rng.sample(stage_records, per_stage))
        else:
            sampled.extend(stage_records)

    return sampled


def filtered_path_for(path):
    path = Path(path)
    return path.with_name(path.name.replace(".tad", "_filtered.tad"))


def imputed_path_for(path):
    path = Path(path)
    return path.with_name(path.name.replace(".tad", "_imputed.tad"))


def write_intervals_to_tad_file(intervals, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        for a, b in sorted(intervals):
            handle.write(f"{a}\t{b}\n")


def impute_children_for_stage(file_paths, ct=1.0, cb=1.0):
    """Run nussinov_with_imputation on one stage and save imputed child TADs."""
    if not file_paths:
        return []

    cells = [read_tad_file_as_strings(path) for path in file_paths]
    m = len(cells)
    edges = [(0, i + 1) for i in range(m)]
    leaf_intervals = {i + 1: deepcopy(cells[i]) for i in range(m)}

    starts = set()
    ends = set()
    for interval_set in cells:
        for item in interval_set:
            nums = re.findall(r"-?\d+", str(item))
            if len(nums) >= 2:
                a = int(nums[0])
                b = int(nums[1])
                starts.add(a)
                starts.add(b)
                ends.add(a)
                ends.add(b)

    _, _, children_map = infer_tads_with_imputation(
        edges,
        leaf_intervals,
        starts,
        ends,
        Ct=ct,
        Cb=cb,
        trace=False,
    )

    imputed_paths = []
    for idx, path in enumerate(file_paths, start=1):
        out_path = imputed_path_for(path)
        child_intervals = set(children_map.get(idx, frozenset()))
        write_intervals_to_tad_file(child_intervals, out_path)
        imputed_paths.append(str(out_path))

    return imputed_paths


def resolve_base_dir(base_dir_arg):
    """Resolve base-dir whether script is run from repo root or scptr directory."""
    base = Path(base_dir_arg)
    if base.is_absolute():
        return base

    candidates = [
        Path.cwd() / base,
        Path(__file__).resolve().parent / base,
        REPO_ROOT / base,
    ]

    # If user passes "scptr/cell_differentiation" while already in scptr/.
    if len(base.parts) > 1 and base.parts[0] == "scptr":
        trimmed = Path(*base.parts[1:])
        candidates.extend([
            Path.cwd() / trimmed,
            Path(__file__).resolve().parent / trimmed,
            REPO_ROOT / trimmed,
        ])

    for cand in candidates:
        if cand.exists():
            return cand

    return candidates[0]


def compute_pairwise_distance_matrix(cells, ct=1.0, cb=1.0, tolerance=0):
    n = len(cells)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = tad_distance(cells[i], cells[j], ct=ct, cb=cb, tolerance=tolerance)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def compute_joint_distance_matrix(cells, medians, ct=1.0, cb=1.0, tolerance=0):
    """Distance matrix for cells + medians to co-embed them in 2D."""
    n = len(cells)
    k = len(medians)
    total = n + k
    dist = np.zeros((total, total), dtype=float)

    for i in range(total):
        for j in range(i + 1, total):
            if i < n and j < n:
                d = tad_distance(cells[i], cells[j], ct=ct, cb=cb, tolerance=tolerance)
            elif i >= n and j >= n:
                d = tad_distance(medians[i - n], medians[j - n], ct=ct, cb=cb, tolerance=tolerance)
            elif i < n and j >= n:
                d = tad_distance(cells[i], medians[j - n], ct=ct, cb=cb, tolerance=tolerance)
            else:
                d = tad_distance(medians[i - n], cells[j], ct=ct, cb=cb, tolerance=tolerance)
            dist[i, j] = d
            dist[j, i] = d

    return dist


def compute_mst_edges_for_medians(medians, ct=1.0, cb=1.0, tolerance=0):
    """Return MST edges (u,v,w) on median indices using Prim's algorithm."""
    k = len(medians)
    if k <= 1:
        return []

    mdist = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            d = tad_distance(medians[i], medians[j], ct=ct, cb=cb, tolerance=tolerance)
            mdist[i, j] = d
            mdist[j, i] = d

    in_tree = {0}
    mst_edges = []
    while len(in_tree) < k:
        best = None
        best_w = float("inf")
        for u in in_tree:
            for v in range(k):
                if v in in_tree:
                    continue
                w = mdist[u, v]
                if w < best_w:
                    best_w = w
                    best = (u, v, w)
        if best is None:
            break
        mst_edges.append(best)
        in_tree.add(best[1])

    return mst_edges


def make_side_by_side_plot(cell_coords, true_labels, em_labels, output_png, title_suffix="", median_coords=None, mst_edges=None, median_labels=None):
    unique_true = sorted(set(true_labels))
    cmap_true = plt.cm.get_cmap("tab10", len(unique_true))
    true_to_id = {label: idx for idx, label in enumerate(unique_true)}
    true_ids = [true_to_id[x] for x in true_labels]

    plt.figure(figsize=(13, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(cell_coords[:, 0], cell_coords[:, 1], c=true_ids, cmap="tab10", s=44, alpha=0.9)
    ax1.set_title(f"True Clustering{title_suffix}")
    ax1.set_xlabel("MDS-1")
    ax1.set_ylabel("MDS-2")

    handles = []
    for label in unique_true:
        idx = true_to_id[label]
        handles.append(ax1.scatter([], [], c=[cmap_true(idx)], label=label, s=36))
    ax1.legend(handles=handles, loc="best", fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    scatter_em = ax2.scatter(cell_coords[:, 0], cell_coords[:, 1], c=em_labels, cmap="tab20", s=44, alpha=0.9)

    if median_coords is not None and len(median_coords) > 0:
        ax2.scatter(median_coords[:, 0], median_coords[:, 1], c="black", s=120, marker="X", label="Cluster median")
        for midx in range(len(median_coords)):
            label = f"M{midx}"
            if median_labels is not None and midx < len(median_labels):
                label = str(median_labels[midx])
            ax2.text(median_coords[midx, 0], median_coords[midx, 1], label, fontsize=8, color="black")

    if mst_edges:
        for u, v, _ in mst_edges:
            x1, y1 = median_coords[u, 0], median_coords[u, 1]
            x2, y2 = median_coords[v, 0], median_coords[v, 1]
            ax2.plot([x1, x2], [y1, y2], color="black", linewidth=1.5, alpha=0.85)
    ax2.set_title(f"EM Clustering (Imputed Children){title_suffix}")
    ax2.set_xlabel("MDS-1")
    ax2.set_ylabel("MDS-2")
    plt.colorbar(scatter_em, ax=ax2, label="Cluster ID")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Select files, impute children per stage, then run EM clustering on imputed TADs."
    )
    parser.add_argument("--base-dir", default="cell_differentiation", help="Base directory containing 1-cell/2-cell/4-cell/8-cell/64-cell")
    parser.add_argument("--chromosome", required=True, help="Chromosome name, e.g. chr1")
    parser.add_argument("--resolution", required=True, help="Resolution folder, e.g. 100kb")
    parser.add_argument("--k", type=int, default=5, help="Number of EM clusters")
    parser.add_argument("--ct", type=float, default=1.0, help="Interval mismatch weight")
    parser.add_argument("--cb", type=float, default=1.0, help="Boundary mismatch weight")
    parser.add_argument("--tolerance", type=int, default=10, help="Boundary harmonization tolerance")
    parser.add_argument("--per-stage", type=int, default=10, help="Number of files to sample from each stage")
    parser.add_argument("--max-iter", type=int, default=30, help="Maximum EM iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Objective convergence tolerance")
    parser.add_argument("--init", choices=["random", "kmeans++", "first"], default="kmeans++", help="EM initialization mode")
    parser.add_argument("--random-state", type=int, default=100, help="Random seed")
    parser.add_argument("--json-out", default=None, help="Output JSON file for this run")
    parser.add_argument("--output", default=None, help="Output png path")
    args = parser.parse_args()

    stage_dirs = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
    resolved_base_dir = resolve_base_dir(args.base_dir)
    records = collect_scptr_files(resolved_base_dir, args.chromosome, args.resolution, stage_dirs)

    if not records:
        raise SystemExit(
            "No tads.tad files found for "
            f"chromosome={args.chromosome}, resolution={args.resolution} "
            f"under {resolved_base_dir}"
        )

    print("Sampling files per stage...", flush=True)
    sampled_records = sample_records_per_stage(records, args.per_stage, random_state=args.random_state)
    sampled_records = sorted(sampled_records, key=lambda record: (record["stage"], record["sample"], record["path"]))

    file_paths = [r["path"] for r in sampled_records]
    true_labels = [r["stage"] for r in sampled_records]

    print(
        "Sampled files per stage: "
        + ", ".join(
            f"{stage}={sum(1 for r in sampled_records if r['stage'] == stage)}"
            for stage in stage_dirs
        ),
        flush=True,
    )

    print(
        f"Harmonizing leaf boundaries with tolerance={args.tolerance} for {len(file_paths)} files...",
        flush=True,
    )
    harmonize_leaf_boundaries(file_paths, args.tolerance)

    filtered_paths = [str(filtered_path_for(path)) for path in file_paths]
    missing_filtered = [path for path in filtered_paths if not Path(path).exists()]
    if missing_filtered:
        raise FileNotFoundError(
            "Expected filtered TAD files were not created: " + ", ".join(missing_filtered)
        )

    grouped_filtered = {}
    for record, filtered_path in zip(sampled_records, filtered_paths):
        grouped_filtered.setdefault(record["stage"], []).append(filtered_path)

    imputed_paths_by_stage = {}
    for stage in stage_dirs:
        stage_paths = grouped_filtered.get(stage, [])
        if not stage_paths:
            continue
        print(
            f"Running nussinov_with_imputation for stage={stage} on {len(stage_paths)} files...",
            flush=True,
        )
        imputed_paths_by_stage[stage] = impute_children_for_stage(stage_paths, ct=args.ct, cb=args.cb)

    imputed_paths = []
    ordered_true_labels = []
    for stage in stage_dirs:
        for path in imputed_paths_by_stage.get(stage, []):
            imputed_paths.append(path)
            ordered_true_labels.append(stage)

    if not imputed_paths:
        raise SystemExit("No imputed files were produced.")

    cells = [read_tad_file_as_strings(path) for path in imputed_paths]

    labels, medians, history = em_cluster_tads(
        cells,
        k=args.k,
        ct=args.ct,
        cb=args.cb,
        max_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
        random_state=args.random_state,
        distance_tolerance=args.tolerance,
    )

    cluster_majorities, cluster_summary = compute_majority_labels(labels, ordered_true_labels)

    mst_edges = compute_mst_edges_for_medians(medians, ct=args.ct, cb=args.cb, tolerance=args.tolerance)

    mst_graph = nx.Graph()
    mst_graph.add_nodes_from(range(len(medians)))
    mst_graph.add_edges_from([(u, v) for u, v, _ in mst_edges])

    tree_distance = compute_tree_distance_from_mst(mst_graph, cluster_majorities)

    dist = compute_joint_distance_matrix(cells, medians, ct=args.ct, cb=args.cb, tolerance=args.tolerance)

    positive = dist[dist > 0]
    if len(positive) > 0:
        scale = np.median(positive)
        if scale > 0:
            dist = dist / scale

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        metric=True,
        n_init=20,
        max_iter=5000,
        eps=1e-9,
        random_state=args.random_state,
    )
    coords = mds.fit_transform(dist)
    n_cells = len(cells)
    cell_coords = coords[:n_cells]
    median_coords = coords[n_cells:]

    if args.output:
        output_png = args.output
    else:
        output_png = (
            f"scptr_imputed_em_{args.chromosome}_{args.resolution}"
            f"_perstage{args.per_stage}_k{args.k}.png"
        )

    title_suffix = f" ({args.chromosome}, {args.resolution})"
    make_side_by_side_plot(
        cell_coords,
        ordered_true_labels,
        labels,
        output_png,
        title_suffix=title_suffix,
        median_coords=median_coords,
        mst_edges=mst_edges,
        median_labels=[cluster_majorities.get(i, f"M{i}") for i in range(len(medians))],
    )

    if args.json_out:
        json_path = args.json_out
    else:
        json_path = f"results/scptr_tree_distance/{args.chromosome}/tol_{args.tolerance}.json"

    # Save MST details in both machine-readable and compact edge-string forms.
    spanning_tree = []
    spanning_tree_edges = []
    for u, v, w in mst_edges:
        u_majority = cluster_majorities.get(u, f"M{u}")
        v_majority = cluster_majorities.get(v, f"M{v}")
        u_id = STAGE_TO_ID.get(u_majority, str(u_majority))
        v_id = STAGE_TO_ID.get(v_majority, str(v_majority))
        edge_label = f"{u_id}-{v_id}"
        weight = float(w)
        spanning_tree_edges.append(edge_label)
        spanning_tree.append(
            {
                "edge": edge_label,
                "source": u_id,
                "target": v_id,
                "weight": weight,
            }
        )

    write_run_json(
        json_path,
        {
            "chromosome": args.chromosome,
            "resolution": args.resolution,
            "tolerance": args.tolerance,
            "per_stage": args.per_stage,
            "k": args.k,
            "random_state": args.random_state,
            "tree_distance": tree_distance,
            "cluster_majorities": {str(k): v for k, v in sorted(cluster_majorities.items())},
            "cluster_summary": {str(k): v for k, v in sorted(cluster_summary.items())},
            "spanning_tree_edges": spanning_tree_edges,
            "spanning_tree": spanning_tree,
            "output_png": output_png,
            "filtered_paths": filtered_paths,
            "imputed_paths": imputed_paths,
        },
    )

    print("Input files:", len(file_paths))
    print("Filtered files:", len(filtered_paths))
    print("Imputed files:", len(imputed_paths))
    print("True group counts:", dict(Counter(ordered_true_labels)))
    print("EM cluster counts:", dict(Counter(labels)))
    print("Median MST edges:", len(mst_edges))
    print("Tree distance:", tree_distance)
    print("Run JSON:", json_path)
    print("EM iterations:", len(history["obj"]))
    if history["obj"]:
        print("Final objective:", history["obj"][-1])
    print("Saved figure:", output_png)


if __name__ == "__main__":
    main()