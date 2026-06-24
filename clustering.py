import os
import glob
import random
import re
from collections import defaultdict
from copy import deepcopy
from nussinov import infer_tads_with_dp
from filtering import harmonize_leaf_boundaries
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# increase recursion limit for DP
sys.setrecursionlimit(50000)

# ---- helper: read .tad file ----
def read_tad_file_as_strings(path, start_col=0, end_col=1, sep=None):
    """
    Read a .tad file where first two columns are integer indices (start, end).
    Return a set of strings like "[start,end]" (compatible with parse_interval in your DP).
    """
    s = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split() if sep is None else line.split(sep)
            try:
                a = int(cols[start_col])
                b = int(cols[end_col])
            except Exception:
                # skip header or malformed lines
                continue
            s.add(f"[{a},{b}]")
    return s

# ---- tad distance (RF-like) ----
def tad_distance(intervalsA, intervalsB, Ct=1.0, Cb=1.0):
    """
    intervalsA/B are sets of (string "[a,b]" or tuple) or mixture.
    We'll coerce to tuples of ints then compute start/end sets.
    """
    def to_pairs(intervals):
        pairs = set()
        for it in intervals:
            if isinstance(it, tuple):
                pairs.add((int(it[0]), int(it[1])))
            else:
                # expect string like "[a,b]" or "a,b"
                nums = re.findall(r"-?\d+", str(it))
                if len(nums) >= 2:
                    pairs.add((int(nums[0]), int(nums[1])))
        return pairs

    A = to_pairs(intervalsA)
    B = to_pairs(intervalsB)

    A_starts = {a for (a, _) in A}
    B_starts = {a for (a, _) in B}
    A_ends = {b for (_, b) in A}
    B_ends = {b for (_, b) in B}

    interval_symdiff = len(A.symmetric_difference(B))/(len(A) + len(B))  # avoid div by zero
    start_symdiff = len(A_starts.symmetric_difference(B_starts))/(len(A_starts) + len(B_starts)) 
    end_symdiff = len(A_ends.symmetric_difference(B_ends))/ (len(A_ends) + len(B_ends))

    return (Ct * interval_symdiff + Cb * (start_symdiff + end_symdiff))

# ---- wrapper that calls your DP as a median solver for a given list of cells ----
def compute_median_for_cluster(list_of_interval_sets, Ct=1.0, Cb=1.0):
    """
    list_of_interval_sets: list of sets of strings like "[a,b]" (one per cell).
    Use infer_tads_with_dp to compute the median TAD set for this cluster.
    Returns median_intervals (set of tuples) and score.
    """
    # Build star topology edges like (0,1),(0,2),...
    m = len(list_of_interval_sets)
    if m == 0:
        return set(), 0.0
    edges = [(0, i+1) for i in range(m)]
    # leaf_intervals mapping: children numbered 1..m
    leaf_intervals = {i+1: deepcopy(list_of_interval_sets[i]) for i in range(m)}

    # build start/end boundary universe
    starts = set()
    ends = set()
    for s in list_of_interval_sets:
        for it in s:
            nums = re.findall(r"-?\d+", str(it))
            if len(nums) >= 2:
                starts.add(int(nums[0]))
                ends.add(int(nums[1]))

    # call your infer_tads_with_dp (must be in the same module / imported)
    med_intervals, med_score = infer_tads_with_dp(edges, leaf_intervals, starts, ends, Ct=Ct, Cb=Cb, trace=False)
    # med_intervals already returned as set of (a,b) tuples in your DP
    return med_intervals, med_score

# ---- EM clustering ----
def em_cluster_tads_from_files(file_paths, k=2, Ct=1.0, Cb=1.0,
                               max_iter=50, tol=1e-6, init='kmeans++', random_state=None):
    """
    file_paths: list of file paths to .tad files (one per single cell).
    Returns: labels list (len = n_cells), medians list (k sets), history dict.
    """

    if random_state is not None:
        random.seed(random_state)

    # read all .tad files
    cells = [read_tad_file_as_strings(p) for p in file_paths]
    n = len(cells)
    if n == 0:
        return [], [], {}

    # precompute pairwise distances (optional but helpful)
    pairwise = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = tad_distance(cells[i], cells[j], Ct=Ct, Cb=Cb)
            pairwise[i][j] = d
            pairwise[j][i] = d

    # initialization
    if init == 'random':
        med_idx = random.sample(range(n), k)
    elif init == 'kmeans++':
        med_idx = []
        # kmeans++ style: pick first uniformly, then weighted by squared distance
        med_idx.append(random.randrange(n))
        while len(med_idx) < k:
            d2 = []
            for i in range(n):
                if i in med_idx:
                    d2.append(0.0)
                else:
                    mind = min(pairwise[i][c] for c in med_idx)
                    d2.append(mind**2)
            tot = sum(d2)
            if tot == 0:
                # fallback random
                choices = [i for i in range(n) if i not in med_idx]
                med_idx.append(random.choice(choices))
            else:
                r = random.random()*tot
                cum = 0.0
                for i,val in enumerate(d2):
                    cum += val
                    if cum >= r and i not in med_idx:
                        med_idx.append(i)
                        break
        # med_idx length k
    else:
        # deterministic: first k items
        med_idx = list(range(min(k, n)))

    # build initial medians as the actual cell intervals at med_idx
    medians = [deepcopy(cells[i]) for i in med_idx]

    labels = [-1]*n
    history = {'obj': [], 'labels': []}

    for it in range(max_iter):
        # E-step: assign labels to nearest median (compute distances)
        new_labels = [None]*n
        total_obj = 0.0
        for i in range(n):
            best_d = float('inf'); best_k = None
            for cind in range(k):
                d = tad_distance(cells[i], medians[cind], Ct=Ct, Cb=Cb)
                if d < best_d:
                    best_d = d; best_k = cind
            new_labels[i] = best_k
            total_obj += best_d

        # M-step: for each cluster recompute median via your DP median solver
        new_medians = []
        for cind in range(k):
            members = [cells[i] for i in range(n) if new_labels[i] == cind]
            if len(members) == 0:
                # empty cluster -> reinitialize to a random point
                new_medians.append(deepcopy(cells[random.randrange(n)]))
                continue
            med_set, med_score = compute_median_for_cluster(members, Ct=Ct, Cb=Cb)
            # med_set is set of (a,b) tuples; convert to the same string-set format for consistency of tad_distance
            med_strs = set(f"[{a},{b}]" for (a,b) in med_set)
            new_medians.append(med_strs)

        history['obj'].append(total_obj)
        history['labels'].append(new_labels)

        # check convergence: labels unchanged or objective change small
        if new_labels == labels:
            break
        if len(history['obj']) > 1 and abs(history['obj'][-2] - history['obj'][-1]) < tol:
            labels = new_labels
            medians = new_medians
            break

        labels = new_labels
        medians = new_medians

    return labels, medians, history

# ---- Example usage: cluster two directories into 2 clusters ----
def cluster_dirs(dir1, dir2, dir3, k=2, **kwargs):
    """
    dir1, dir2, dir3: directories with *.tad files for three cell types.
    Merge the files from all directories and cluster them.
    """
    # files1 = sorted(glob.glob(os.path.join(dir1, "*.tad")))
    # files2 = sorted(glob.glob(os.path.join(dir2, "*.tad")))
    # files3 = sorted(glob.glob(os.path.join(dir3, "*.tad")))
    # harmonize_leaf_boundaries(files3, 10)  # optional: harmonize boundaries across all files before clustering
    # harmonize_leaf_boundaries(files1, 10)  # optional: harmonize boundaries across all files before clustering
    # harmonize_leaf_boundaries(files2, 10)
    files1= sorted(glob.glob(os.path.join(dir1, "*_filtered.tad")))  # re-read after harmonization
    files2= sorted(glob.glob(os.path.join(dir2, "*_filtered.tad")))
    files3= sorted(glob.glob(os.path.join(dir3, "*_filtered.tad")))
    file_list = files1 + files2 + files3

    labels, medians, history = em_cluster_tads_from_files(file_list, k=k, **kwargs)
    return file_list, labels, medians, history



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EM clustering with optional random state.")
    parser.add_argument("--random-state", type=int, default=100, help="Random seed for clustering")
    parser.add_argument("--max_iter", type=int, default=30, help="Maximum iterations for EM")
    args = parser.parse_args()
    random_state = args.random_state

    file_list, labels, medians, history = cluster_dirs(
        "single_cells/h1esc/tads",
        "single_cells/imr90/tads",
        "single_cells/hff-gm/tads",
        k=6,
        Ct=1.0, Cb=1.0,
        max_iter=args.max_iter,
        init='random',
        random_state=random_state
    )

    # print assignment counts
    from collections import Counter
    print("cluster counts:", Counter(labels))

    # print which files belong to which cluster
    print("\n" + "="*70)
    print("File assignments by cluster:")
    print("="*70)
    for cluster_id in range(max(labels) + 1):
        cluster_file_paths = [file_list[i] for i in range(len(file_list)) if labels[i] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_file_paths)} files):")
        for fpath in sorted(cluster_file_paths):
            print(f"  {fpath}")

    # --- Step 1: Compute full pairwise distance matrix ---
    n = len(file_list)
    dist_matrix = np.zeros((n, n))

    cells = [read_tad_file_as_strings(p) for p in file_list]

    for i in range(n):
        for j in range(i+1, n):
            d = tad_distance(cells[i], cells[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # --- Step 2: 2D embedding using MDS ---
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_matrix)

    # --- Step 3: Build TRUE labels based on directory names ---
    true_labels = []
    for f in file_list:
        if "h1esc" in f.lower():
            true_labels.append("H1ESC")
        elif "imr90" in f.lower():
            true_labels.append("IMR90")
        elif "hff" in f.lower():
            true_labels.append("HFF-GM")
        else:
            true_labels.append("Unknown")

    # --- Convert true labels to integers for plotting ---
    unique_true = list(sorted(set(true_labels)))
    true_label_ids = [unique_true.index(x) for x in true_labels]

    # =============================
    # Plot 1: TRUE clustering
    # =============================
    plt.figure(figsize=(12,5))

    # Manual color mapping: red, green, blue for the three cell types
    color_map = {'H1ESC': 'red', 'IMR90': 'green', 'HFF-GM': 'blue'}
    true_colors = [color_map.get(label, 'gray') for label in true_labels]

    plt.subplot(1,2,1)
    plt.scatter(coords[:,0], coords[:,1], c=true_colors, s=50, alpha=0.7)
    plt.title("Original Cell Types")
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")

    # legend with manual colors
    for name in unique_true:
        plt.scatter([], [], c=color_map.get(name, 'gray'), label=name, s=50)
    plt.legend()

    # =============================
    # Plot 2: EM clustering result
    # =============================
    plt.subplot(1,2,2)
    plt.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab10', s=50, alpha=0.7)
    plt.title("Your EM Clustering")
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.colorbar(label='Cluster ID')

    plt.tight_layout()
    plt.savefig(f"clustering_result_{random_state}_{args.max_iter}.png", dpi=300)
    plt.close()