import os
import random
import re
from collections import defaultdict, deque
import networkx as nx
import itertools
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from check_lp_constrained import solve_lp_from_edges_and_leaves
from filtering import *
from tad_utilities import *








def compute_parsimony_score(edges, node_tads, Ct=1.0, Cb=1.0, tolerance=0):
    total = 0
    per_edge = {}

    for u, v in edges:
        Tu = node_tads.get(u, set())
        Tv = node_tads.get(v, set())

        # --- TAD RF ---
        rf_tad = relaxed_tad_rf(Tu, Tv, tol=tolerance)
        tad_cost = Ct * rf_tad

        # --- Boundary RF (derived) ---
        Su, Eu = boundaries_from_tads(Tu)
        Sv, Ev = boundaries_from_tads(Tv)

        rf_boundary = relaxed_boundary_rf(Su, Sv, Eu, Ev, tol=tolerance)
        boundary_cost = Cb * rf_boundary

        edge_cost = tad_cost + boundary_cost
        total += edge_cost
        per_edge[(u, v)] = {
            "tad_rf": rf_tad,
            "boundary_rf": rf_boundary,
            "cost": edge_cost
        }

    return total, per_edge



def brute_force(edges, leaf_intervals, Ct=1.0, Cb=1.0, tolerance=0):
    """
    Exact brute-force Sankoff parsimony with laminar TAD states.

    Returns:
        best_cost: optimal parsimony score
        assignment: dict node -> set of (a,b) intervals
    """

    # -------------------------
    # Utilities
    # -------------------------


    def is_laminar(intervals):
        I = list(intervals)
        for i in range(len(I)):
            for j in range(i + 1, len(I)):
                a, b = I[i]
                c, d = I[j]
                if (a < c < b < d) or (c < a < d < b):
                    return False
        return True


    @lru_cache(None)
    def state_cost(frozen_u, frozen_v):
        Tu = set(frozen_u)
        Tv = set(frozen_v)

        # RF on TADs
        rf_tad = relaxed_tad_rf(Tu, Tv, tol=tolerance)

        # RF on boundaries (start/end distinct)
        Su, Eu = boundaries_from_tads(Tu)
        Sv, Ev = boundaries_from_tads(Tv)

        rf_boundary = relaxed_boundary_rf(Su, Sv, Eu, Ev, tol=tolerance)

        return Ct * rf_tad + Cb * rf_boundary

    # -------------------------
    # Tree structure
    # -------------------------

    children = defaultdict(list)
    parent = {}
    nodes = set()

    for u, v in edges:
        children[u].append(v)
        parent[v] = u
        nodes.add(u)
        nodes.add(v)

    root = next(n for n in nodes if n not in parent)

    # -------------------------
    # Candidate boundaries (from leaves only)
    # -------------------------

    positions = set()
    for ints in leaf_intervals.values():
        for s in ints:
            a, b = parse_interval(s)
            positions.add(a)
            positions.add(b)
    positions = sorted(positions)

    # candidate intervals (a < b)
    all_intervals = [(a, b) for a in positions for b in positions if a < b]

    # -------------------------
    # Generate laminar states
    # -------------------------

    laminar_states = []
    for r in range(len(all_intervals) + 1):
        for subset in itertools.combinations(all_intervals, r):
            if is_laminar(subset):
                laminar_states.append(frozenset(subset))

    # -------------------------
    # Leaf states (fixed)
    # -------------------------

    leaf_state = {
        v: frozenset(parse_interval(s) for s in leaf_intervals[v])
        for v in leaf_intervals
    }

    # -------------------------
    # Postorder traversal
    # -------------------------

    postorder = []

    def dfs(u):
        for c in children.get(u, []):
            dfs(c)
        postorder.append(u)

    dfs(root)

    # -------------------------
    # Sankoff DP
    # -------------------------

    dp = {u: {S: float("inf") for S in laminar_states} for u in nodes}
    back = {u: {} for u in nodes}

    for u in postorder:
        for Su in laminar_states:

            # Leaf constraint
            if u in leaf_state:
                if Su == leaf_state[u]:
                    dp[u][Su] = 0.0
                continue

            total = 0.0
            choice = []

            for c in children.get(u, []):
                best = float("inf")
                best_Sc = None

                for Sc in laminar_states:
                    val = dp[c][Sc] + state_cost(Su, Sc)
                    if val < best:
                        best = val
                        best_Sc = Sc

                total += best
                choice.append(best_Sc)

            dp[u][Su] = total
            back[u][Su] = choice

    # -------------------------
    # Recover optimal solution
    # -------------------------

    best_root_state = min(dp[root], key=lambda S: dp[root][S])
    best_cost = dp[root][best_root_state]

    assignment = {}

    def traceback(u, Su):
        assignment[u] = set(Su)
        for i, c in enumerate(children.get(u, [])):
            traceback(c, back[u][Su][i])

    traceback(root, best_root_state)

    return best_cost, assignment

def read_bedpe_intervals(
    bedpe_file,
    chrom=None,
    min_size=None,
    max_size=None
):
    """
    Read a BEDPE file and return a set of interval strings: {"[start,end]"}.
    
    Parameters
    ----------
    chrom : str or None
        If given, only keep rows on this chromosome (e.g. "1" or "chr1")
    min_size, max_size : int or None
        Optional TAD size filtering
    """
    intervals = set()

    with open(bedpe_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            fields = line.rstrip().split("\t")
            chr1, x1, x2, chr2, y1, y2 = fields[:6]

            # intra-chromosomal only
            if chr1 != chr2:
                continue

            if chrom is not None and chr1 not in {chrom, f"chr{chrom}"}:
                continue

            x1, x2 = int(x1), int(x2)

            size = x2 - x1
            if min_size is not None and size < min_size:
                continue
            if max_size is not None and size > max_size:
                continue

            intervals.add(f"[{x1},{x2}]")

    return intervals


def build_interval_index(file_paths, chrom="1", resolution=None):
    """
    Build an index mapping interval tuples to files that contain them.
    """
    interval_to_files = defaultdict(set)
    for path in file_paths:
        if path.lower().endswith(".tad"):
            intervals = read_tad_intervals(path, resolution=resolution)
        else:
            intervals = read_bedpe_intervals(path, chrom=chrom)
        for s in intervals:
            a, b = parse_interval(s)
            interval_to_files[(a, b)].add(path)
    return interval_to_files





def get_internal_nodes(edges):
    """Return sorted list of internal nodes (nodes with at least one child)."""
    return sorted({u for u, _ in edges})






def _normalize_interval_list(intervals):
    normalized = set()
    for item in intervals:
        if isinstance(item, str):
            normalized.add(parse_interval(item))
        else:
            a, b = item
            normalized.add((int(a), int(b)))
    return normalized


def _tad_in_set_with_tolerance(tad, tad_set, tolerance=0):
    for other in tad_set:
        if tads_match_with_tolerance(tad, other, tolerance):
            return True
    return False


def print_internal_unique_intervals(node_assignments, leaf_intervals, edges, tolerance=0):
    """
    Print intervals inferred at internal nodes that are not present in any leaf node.
    """
    internal_nodes = get_internal_nodes(edges)

    leaf_tads = set()
    for intervals in leaf_intervals.values():
        leaf_tads |= _normalize_interval_list(intervals)

    print("\nInternal-node intervals not present in any leaf (tolerance=%d):" % tolerance)
    for n in internal_nodes:
        tads = _normalize_interval_list(node_assignments.get(n, set()))
        unique_tads = [t for t in sorted(tads) if not _tad_in_set_with_tolerance(t, leaf_tads, tolerance)]
        print(f"  Node {n}: {len(unique_tads)} intervals")
        if unique_tads:
            for a, b in unique_tads:
                print(f"    [{a},{b}]")


def print_internal_unique_intervals_from_files(internal_files, leaf_files, edges, tolerance=0, chrom="1", resolution=None):
    leaf_intervals = {}
    for node_id, path in leaf_files.items():
        if path.lower().endswith(".tad"):
            intervals = read_tad_intervals(path, resolution=resolution)
        else:
            intervals = read_bedpe_intervals(path, chrom=chrom)
        leaf_intervals[node_id] = intervals

    internal_assignments = {}
    for node_id, path in internal_files.items():
        if path.lower().endswith(".tad"):
            intervals = read_tad_intervals(path, resolution=resolution)
        else:
            intervals = read_bedpe_intervals(path, chrom=chrom)
        internal_assignments[node_id] = {parse_interval(s) for s in intervals}

    print_internal_unique_intervals(internal_assignments, leaf_intervals, edges, tolerance=tolerance)


def fitch_hartigan_tads(edges, leaf_intervals):
    """
    Fitch-Hartigan-style majority rule on TAD presence/absence.
    Bottom-up: majority of children -> {1} or {0}, tie -> {0,1}.
    Top-down: resolve ties using parent assignment (root ties default to 0).
    """
    children = defaultdict(list)
    parent = {}
    nodes = set()

    for u, v in edges:
        children[u].append(v)
        parent[v] = u
        nodes.add(u)
        nodes.add(v)

    root = next(n for n in nodes if n not in parent)

    leaf_sets = {node: _normalize_interval_list(tads) for node, tads in leaf_intervals.items()}
    interval_universe = set()
    for tads in leaf_sets.values():
        interval_universe |= tads

    postorder = []

    def dfs(u):
        for c in children.get(u, []):
            dfs(c)
        postorder.append(u)

    dfs(root)

    ZERO = frozenset((0,))
    ONE = frozenset((1,))
    BOTH = frozenset((0, 1))

    states = {u: {} for u in nodes}
    for u in postorder:
        if u in leaf_sets:
            leaf_tads = leaf_sets[u]
            for interval in interval_universe:
                states[u][interval] = ONE if interval in leaf_tads else ZERO
            continue

        for interval in interval_universe:
            present = 0
            absent = 0
            for c in children.get(u, []):
                child_state = states[c][interval]
                if child_state == ONE:
                    present += 1
                elif child_state == ZERO:
                    absent += 1
            if present > absent:
                states[u][interval] = ONE
            elif absent > present:
                states[u][interval] = ZERO
            else:
                states[u][interval] = BOTH

    resolved = {u: {} for u in nodes}

    def resolve_value(state, parent_value=None, is_root=False):
        if state == BOTH:
            return 0 if is_root else parent_value
        return 1 if state == ONE else 0

    def dfs_resolve(u, parent_values=None):
        for interval in interval_universe:
            state = states[u][interval]
            resolved[u][interval] = resolve_value(
                state,
                parent_value=None if parent_values is None else parent_values[interval],
                is_root=(u == root),
            )
        for c in children.get(u, []):
            dfs_resolve(c, resolved[u])

    dfs_resolve(root)

    node_tads = {}
    for u in nodes:
        node_tads[u] = {interval for interval, val in resolved[u].items() if val == 1}

    return node_tads


def write_fitch_hartigan_phylogeny(node_tads, total_score, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Total parsimony score (tads only): {total_score:.4f}\n")
        f.write("\nNode assignments:\n")
        for node in sorted(node_tads.keys()):
            tads = sorted(node_tads[node])
            f.write(f"Node {node}: {len(tads)} TADs\n")
            for a, b in tads:
                f.write(f"  [{a},{b}]\n")







def merge_common_tads_into_third(tad_file_a, tad_file_b, tad_file_c,
                                 tolerance=0, resolution=None, min_size=None, max_size=None):
    """
    Find TADs common to A and B, union them with TADs in C, and save into C.

    Returns a dict with counts of common/union results.
    """
    intervals_a = read_tad_intervals(tad_file_a, resolution=resolution, min_size=min_size, max_size=max_size)
    intervals_b = read_tad_intervals(tad_file_b, resolution=resolution, min_size=min_size, max_size=max_size)
    intervals_c = read_tad_intervals(tad_file_c, resolution=resolution, min_size=min_size, max_size=max_size)

    tads_a = {parse_interval(s) for s in intervals_a}
    tads_b = {parse_interval(s) for s in intervals_b}
    tads_c = {parse_interval(s) for s in intervals_c}

    common_ab = set()
    for a_start, a_end in tads_a:
        for b_start, b_end in tads_b:
            if tads_match_with_tolerance((a_start, a_end), (b_start, b_end), tolerance):
                common_ab.add((a_start, a_end))
                break
    merged = tads_c | common_ab

    write_tads_to_file(merged, tad_file_c)

    return {
        "tads_in_a": len(tads_a),
        "tads_in_b": len(tads_b),
        "tads_in_c": len(tads_c),
        "common_ab": len(common_ab),
        "merged_total": len(merged),
        "tolerance": tolerance,
    }



def visualize_phylogeny_with_tads(edges, node_tads, per_edge_scores=None, total_score=None, output_file="phylogeny_tads.png", figsize=(20, 15)):
    """
    Visualize the phylogenetic tree with TAD assignments at each node and parsimony scores on edges.
    
    Parameters:
    -----------
    edges : list of tuples
        List of (parent, child) edges
    node_tads : dict
        Dictionary mapping node -> set of (a, b) interval tuples
    per_edge_scores : dict or None
        Dictionary with edge scores from compute_parsimony_score
    output_file : str
        Output filename for the figure
    figsize : tuple
        Figure size (width, height)
    """
    
    # Build the tree structure
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # Find root (node with no incoming edges)
    all_nodes = set()
    children_nodes = set()
    for u, v in edges:
        all_nodes.add(u)
        all_nodes.add(v)
        children_nodes.add(v)
    roots = all_nodes - children_nodes
    root = list(roots)[0] if roots else 0
    
    # Create hierarchical layout
    def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """
        Create a hierarchical tree layout for the graph.
        """
        def _hierarchy_pos(G, root, leftmost, width, leafdx=0.2, vert_gap=0.2, vert_loc=0, xcenter=0.5, rootpos=None, leafpos=None, parent=None):
            if rootpos is None:
                rootpos = {root: (xcenter, vert_loc)}
            else:
                rootpos[root] = (xcenter, vert_loc)
            if leafpos is None:
                leafpos = {}
            children = list(G.successors(root))
            if not children:
                leafpos[root] = (leftmost, vert_loc)
                rootpos[root] = (leftmost, vert_loc)
                return rootpos, leafpos, leftmost + leafdx
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                rootpos, leafpos, nextx = _hierarchy_pos(G, child, nextx, dx, leafdx, vert_gap, vert_loc - vert_gap, nextx, rootpos, leafpos, root)
            midpoint = (rootpos[children[0]][0] + rootpos[children[-1]][0]) / 2
            rootpos[root] = (midpoint, vert_loc)
            return rootpos, leafpos, nextx
        
        rootpos, leafpos, _ = _hierarchy_pos(G, root, 0, width, vert_gap=vert_gap, vert_loc=vert_loc, xcenter=xcenter)
        # Merge positions
        pos = {**rootpos, **leafpos}
        return pos
    
    pos = hierarchy_pos(G, root, width=2.0, vert_gap=0.3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges with parsimony scores
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Draw edge
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.6, zorder=1)
        
        # Add edge label (parsimony score)
        if per_edge_scores and (u, v) in per_edge_scores:
            score_info = per_edge_scores[(u, v)]
            cost = score_info['cost']
            tad_rf = score_info['tad_rf']
            boundary_rf = score_info['boundary_rf']
            
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            label = f"Cost: {cost:.1f}\n(TAD:{tad_rf}, Bnd:{boundary_rf})"
            ax.text(mid_x + 0.05, mid_y, label, fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   ha='left', va='center', zorder=5)
    
    # Draw nodes with TAD information
    for node in G.nodes():
        x, y = pos[node]
        
        # Get TADs for this node
        tads = node_tads.get(node, set())
        tad_list = sorted(tads)
        
        # Create node label
        if tads:
            tad_strings = [f"[{a},{b}]" for a, b in tad_list[:5]]  # Show max 5 TADs
            if len(tad_list) > 5:
                tad_strings.append(f"... +{len(tad_list)-5} more")
            tad_text = "\n".join(tad_strings)
            node_label = f"Node {node}\n{tad_text}"
        else:
            node_label = f"Node {node}\n(no TADs)"
        
        # Determine node color based on whether it's a leaf
        is_leaf = len(list(G.successors(node))) == 0
        node_color = 'lightblue' if is_leaf else 'lightgreen'
        
        # Draw node as a box
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor=node_color, 
                         edgecolor='black', linewidth=2)
        ax.text(x, y, node_label, fontsize=10, ha='center', va='center',
               bbox=bbox_props, zorder=10)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title with total score
    if total_score is not None:
        title = f'Phylogenetic Tree with TAD Assignments and Parsimony Scores\nTotal Parsimony Score: {total_score:.2f}'
    else:
        title = 'Phylogenetic Tree with TAD Assignments and Parsimony Scores'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=40)
    
    # Add legend
    leaf_patch = mpatches.Patch(color='lightblue', label='Leaf nodes')
    internal_patch = mpatches.Patch(color='lightgreen', label='Internal nodes')
    ax.legend(handles=[leaf_patch, internal_patch], loc='upper right', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Phylogeny visualization saved to: {output_file}")
    plt.close()







# -----------------------------
# Example 2: Use provided TADs for all nodes
# -----------------------------

def run_provided_mode(edges, node_files, chrom="1", resolution=None, Ct=1.0, Cb=1.0, tolerance=0, output_file="phylogeny_provided.png"):
    """
    Use provided TAD assignments for all nodes (no inference).
    
    Parameters:
    -----------
    edges : list of tuples
        Tree edges as (parent, child) pairs
    node_files : dict
        Mapping of node ID -> bedpe file path (for ALL nodes, including internal)
    chrom : str
        Chromosome to extract (e.g., "1" or "chr1") for BEDPE inputs
    resolution : int or None
        Resolution (bp) for .tad inputs; if None, keep bin indices
    Ct, Cb : float
        Cost parameters for parsimony
    tolerance : int
        Boundary/tad matching tolerance in bins (0 = exact)
    output_file : str
        Output filename for visualization
    """
    # Read TAD intervals for all nodes from bedpe or .tad files
    node_tads = {}
    for node_id, bedpe_file in node_files.items():
        if bedpe_file.lower().endswith(".tad"):
            intervals = read_tad_intervals(bedpe_file, resolution=resolution)
        else:
            intervals = read_bedpe_intervals(bedpe_file, chrom=chrom)
        # Convert string intervals to tuples
        node_tads[node_id] = {parse_interval(s) for s in intervals}
    
    # Compute parsimony score
    total_score, per_edge_scores = compute_parsimony_score(edges, node_tads, Ct=Ct, Cb=Cb, tolerance=tolerance)
    print("Total parsimony score (provided):", total_score)
    
    # Print TAD assignments for each node
    print("\nTAD assignments per node:")
    for n in sorted(node_tads.keys()):
        laminar = is_laminar_set(node_tads[n], tolerance=tolerance)
        print(f"  Node {n}: {len(node_tads[n])} TADs | Laminar: {laminar}")
    
    # Visualize
    visualize_phylogeny_with_tads(edges, node_tads, per_edge_scores, total_score, output_file=output_file)
    
    return node_tads, total_score, per_edge_scores


# -----------------------------
# Example 3: Infer internal nodes with LP solver
# -----------------------------

def run_inference_mode(edges, leaf_files, chrom="1", resolution=None, Ct=1.0, Cb=1.0, tolerance=0, mode="convex", output_file="phylogeny_inferred.png", internal_output_dir="inferred_internal_tads"):
    """
    Infer internal node TAD assignments using LP solver, then score and plot.

    Parameters:
    -----------
    edges : list of tuples
        Tree edges as (parent, child) pairs
    leaf_files : dict
        Mapping of leaf node ID -> bedpe/.tad file path
    chrom : str
        Chromosome to extract (e.g., "1" or "chr1") for BEDPE inputs
    resolution : int or None
        Resolution (bp) for .tad inputs; if None, keep bin indices
    Ct, Cb : float
        Cost parameters for parsimony and LP
    tolerance : int
        Boundary/tad matching tolerance in bins (0 = exact)
    mode : str
        LP objective mode: 'convex' | 'lex_boundary_first' | 'lex_tad_first' | 'lex_boundary_tad_count'
    output_file : str
        Output filename for visualization
    internal_output_dir : str
        Directory to save inferred internal node TADs
    """
    leaf_intervals = {}
    for node_id, path in leaf_files.items():
        if path.lower().endswith(".tad"):
            intervals = read_tad_intervals(path, resolution=resolution)
        else:
            intervals = read_bedpe_intervals(path, chrom=chrom)
        leaf_intervals[node_id] = intervals

    interval_index = build_interval_index(leaf_files.values(), chrom=chrom, resolution=resolution)

    lp_cost, is_integral, node_assignments, x_vars, nodes, intervals = solve_lp_from_edges_and_leaves(
        edges,
        leaf_intervals,
        Ct=Ct,
        Cb=Cb,
        mode=mode,
        require_laminar_internal=True,
        use_fitch_hartigan=False,
        verbose=True,
    )

    if lp_cost is None:
        print("LP solver failed or returned no solution.")
        return None, None, None

    print(f"LP cost: {lp_cost:.4f} | Integral: {is_integral}")

    # Compute parsimony score from inferred assignments
    total_score, per_edge_scores = compute_parsimony_score(edges, node_assignments, Ct=Ct, Cb=Cb, tolerance=tolerance)
    print("Total parsimony score (inferred):", total_score)

    # Save inferred TADs for internal nodes
    internal_nodes = get_internal_nodes(edges)
    if internal_output_dir:
        for n in internal_nodes:
            tads = node_assignments.get(n, set())
            output_path = os.path.join(internal_output_dir, f"node_{n}_250kb.tad")
            write_tads_to_file(tads, output_path)
        print(f"Saved inferred internal node TADs to: {internal_output_dir}")

    # Print laminarity per node
    print("\nLaminarity per node:")
    for n in sorted(node_assignments.keys()):
        laminar = is_laminar_set(node_assignments[n], interval_index=interval_index, tolerance=tolerance)
        print(f"  Node {n}: Laminar: {laminar}")


    # Visualize
    visualize_phylogeny_with_tads(edges, node_assignments, per_edge_scores, total_score, output_file=output_file)

    return node_assignments, total_score, per_edge_scores

def tad_to_bedpe(tad_file, chromosome="chr1", resolution=10000):
    """
    Convert .tad (bin coordinates) to BEDPE format (base pair coordinates).

    Parameters:
        tad_file (str): path to input .tad file
        chromosome (str): chromosome name (e.g., "chr1")
        resolution (int): bin size in base pairs (e.g., 10000 for 10Kb)
    """

    base, _ = os.path.splitext(tad_file)
    output_file = f"{base}.bedpe"

    with open(tad_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            start_bin = int(parts[0])
            end_bin = int(parts[1])

            start_bp = start_bin * resolution
            end_bp = end_bin * resolution

            fout.write(
                f"{chromosome}\t{start_bp}\t{end_bp}\t"
                f"{chromosome}\t{start_bp}\t{end_bp}\n"
            )

    print(f"Saved BEDPE file: {output_file}")


# -----------------------------
# Main: Choose which mode to run
# -----------------------------

if __name__ == "__main__":
    # Define tree structure
    # edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(4,7),(4,8),(6,9),(6,10),(10,11),(10,12)]
    # edges = [(0,1),(0,2),(1,3),(1,4)] 
    edges = [(0,1),(0,2),(2,3),(2,4)]
    # Set cost parameters
    Ct = 1.0    
    Cb = 1.0
    tolerance = 2
    resolution = None  # set to 10000 for 10Kb bins if using .tad in bp
    mode = "lex_boundary_first"  # LP objective mode for inference
    run_mode = "comparison"  # "provided", "inference", or "comparison"

    
    # # # Mode 2: Use provided TADs for all nodes
    node_files = {
        # 0: "hic_matrix/tads/LT.bedpe",  # Root
        # 1: "hic_matrix/tads/ST.bedpe",  # Internal node
        # 2: "hic_matrix/tads/MPP.bedpe",    
        # 3: "hic_matrix/tads/CMP.bedpe",    
        # 4: "hic_matrix/tads/CLP.bedpe",    
        # 5: "hic_matrix/tads/MEP.bedpe",     
        # 6: "hic_matrix/tads/GMP.bedpe",      
        # 7: "hic_matrix/tads/MKP.bedpe",     
        # 8: "hic_matrix/tads/MK.bedpe",    
        # 9: "hic_matrix/tads/G.bedpe",    
        0: "hemato_cells/tads/MPP_chr1_test_250kb.tad",  
        1: "hemato_cells/tads/CMP_chr1_test_250kb.tad",  
        2: "hemato_cells/tads/CLP_chr1_test_250kb.tad",    
        3: "hemato_cells/tads/MEP_chr1_test_250kb.tad",    
        4: "hemato_cells/tads/GMP_chr1_test_250kb.tad",
    }

    leaf_files = {
        1: "t_cells/tads/CD8_chr2_250kb.tad",    
        3: "t_cells/tads/CD4_Q1_chr2_250kb.tad",    
        4: "t_cells/tads/CD4_Q4_chr2_250kb.tad",
        # 2: "hemato_cells/tads/CLP_chr1_test_250kb.tad",    
        # 3: "hemato_cells/tads/MEP_chr1_test_250kb.tad",    
        # 4: "hemato_cells/tads/GMP_chr1_test_250kb.tad",

    }

    #     # ---- FILTER LEAF FILES ----
    leaf_paths = list(leaf_files.values())

    harmonize_leaf_boundaries(leaf_paths, tolerance)

    # ---- Update dictionary to use filtered files ----
    for node, path in leaf_files.items():
        base, _ = os.path.splitext(path)
        filtered_path = f"{base}_filtered.tad"
        leaf_files[node] = filtered_path
        node_files[node] = filtered_path  # if internal computation uses node_files

    print("Leaf filtering complete.\n")

    # # # merge_common_tads_into_third("hemato_cells/tads/MPP_chr1.tad", "hemato_cells/tads/inferred_internal_tads/node_0.tad", "hemato_cells/tads/MPP_chr1_test.tad", tolerance=tolerance, resolution=resolution)
    # # # merge_common_tads_into_third("hemato_cells/tads/CMP_chr1.tad", "hemato_cells/tads/inferred_internal_tads/node_1.tad", "hemato_cells/tads/CMP_chr1_test.tad", tolerance=tolerance, resolution=resolution)
    if run_mode == "provided":
        node_tads, total_score, per_edge_scores = run_provided_mode(
            edges, node_files, chrom="1", resolution=resolution, Ct=Ct, Cb=Cb, tolerance=tolerance,
            output_file="hemato_cells/phylogeny_provided_relaxed_test_250kb.png"
        )
    elif run_mode == "inference":
        node_tads, total_score, per_edge_scores = run_inference_mode(
            edges, leaf_files, chrom="1", resolution=resolution, Ct=Ct, Cb=Cb, tolerance=tolerance,
            mode=mode,
            output_file="hemato_cells/phylogeny_inferred_test_250kb.png",
           
            internal_output_dir="hemato_cells/tads/inferred_internal_tads"
        )
    elif run_mode == "comparison":
        # tad_to_bedpe("t_cells/tads/MPP_chr1_test_250kb.tad", chromosome="chr1", resolution=250000)
        # tad_to_bedpe("t_cells/tads/inferred_internal_tads/node_0_250kb.tad", chromosome="chr1", resolution=250000)
        # tad_to_bedpe("t_cells/tads/CMP_chr1_test_250kb.tad", chromosome="chr1", resolution=250000)
        # tad_to_bedpe("t_cells/tads/inferred_internal_tads/node_1_250kb.tad", chromosome="chr1", resolution=250000)
        
        comparison = compare_tad_files(
            "t_cells/tads/inferred_internal_tads/node_0_250kb.tad",
            "t_cells/tads/DP_chr2_250kb.tad",
            # "hemato_cells/tads/inferred_internal_tads/node_0_250kb.tad",
            # "hemato_cells/tads/MPP_chr1_test_250kb.tad",
            tolerance=tolerance,
            resolution=resolution,
            min_size=None,
            max_size=None
        )
        print("\nComparison of inferred internal TAD DP vs DP TADs:")
        for key, value in comparison.items():
            print(f"  {key}: {value}")

        # comparison = compare_tad_files(
        #     "hemato_cells/tads/inferred_internal_tads/node_1_250kb.tad",
        #     "hemato_cells/tads/CMP_chr1_test_250kb.tad",
        #     tolerance=tolerance,
        #     resolution=resolution,
        #     min_size=None,
        #     max_size=None
        # )
        # print("\nComparison of inferred internal TAD CMP vs CMP TADs:")
        # for key, value in comparison.items():
        #     print(f"  {key}: {value}")

        internal_nodes = get_internal_nodes(edges)
        internal_files = {}
        for n in internal_nodes:
            # path = os.path.join("hemato_cells/tads/inferred_internal_tads", f"node_{n}_250kb.tad")
            path = os.path.join("t_cells/tads/inferred_internal_tads", f"node_{n}_250kb.tad")
            if os.path.exists(path):
                internal_files[n] = path

        print_internal_unique_intervals_from_files(
            internal_files,
            leaf_files,
            edges,
            tolerance=tolerance,
            chrom="1",
            resolution=resolution,
        )

        leaf_intervals = {}
        for node_id, path in leaf_files.items():
            if path.lower().endswith(".tad"):
                intervals = read_tad_intervals(path, resolution=resolution)
            else:
                intervals = read_bedpe_intervals(path, chrom="1")
            leaf_intervals[node_id] = intervals

        fitch_assignments = fitch_hartigan_tads(edges, leaf_intervals)
        fitch_score, fitch_per_edge_scores = compute_parsimony_score(
            edges,
            fitch_assignments,
            Ct=Ct,
            Cb=Cb,
            tolerance=tolerance,
        )
        # fitch_output_path = "hemato_cells/fitch_hartigan_phylogeny_tads.txt"
        fitch_output_path = "t_cells/fitch_hartigan_phylogeny_tads.txt"
        write_fitch_hartigan_phylogeny(fitch_assignments, fitch_score, fitch_output_path)
        print(f"Saved Fitch-Hartigan phylogeny to: {fitch_output_path}")

        # fitch_plot_path = "hemato_cells/phylogeny_fitch_hartigan_tads.png"
        fitch_plot_path = "t_cells/phylogeny_fitch_hartigan_tads.png"
        visualize_phylogeny_with_tads(
            edges,
            fitch_assignments,
            fitch_per_edge_scores,
            fitch_score,
            output_file=fitch_plot_path,
        )
        print(f"Saved Fitch-Hartigan visualization to: {fitch_plot_path}")



