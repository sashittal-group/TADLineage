import random
import re
from collections import defaultdict, deque
import networkx as nx
import itertools
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

def parse_interval(s):
    a, b = map(int, re.findall(r"\d+", s))
    return (a, b)

def infer_boundaries_by_majority(edges, leaf_intervals,
                                 tie_break_root='prefer1', random_seed=None):
    """
    edges: list of (parent, child) pairs (root is the node without parent)
    leaf_intervals: dict node -> set of interval strings like "{'[1,5]', '[6,10]'}"
    tie_break_root: 'prefer1' (default), 'prefer0', or 'random'
    random_seed: optional int for reproducible random ties when tie_break_root='random'
    
    Returns:
      node_boundaries: dict node -> {'starts': set(positions), 'ends': set(positions)}
      (positions are integers)
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Build node set and children/parent maps
    nodes = set()
    for u, v in edges:
        nodes.add(u); nodes.add(v)
    # include leaves from leaf_intervals in case isolated nodes
    nodes |= set(leaf_intervals.keys())

    children = defaultdict(list)
    parent = {}
    for u, v in edges:
        children[u].append(v)
        parent[v] = u

    # find root(s) (node with no parent). If multiple, treat each as root (forest)
    roots = [n for n in nodes if n not in parent]
    if not roots:
        raise ValueError("No root found (cycle?)")

    # Collect all positions (boundary coordinates) seen at leaves
    positions = set()
    for v, ints in leaf_intervals.items():
        for s in ints:
            a, b = parse_interval(s)
            positions.add(a)
            positions.add(b)
    positions = sorted(positions)

    # Helper: get leaf char for a given position and side
    # side = 's' for start, 'e' for end
    def leaf_char(v, pos, side):
        """Return 1 if leaf v has a boundary of given side at pos, else 0."""
        if v not in leaf_intervals:
            return 0
        for s in leaf_intervals[v]:
            a, b = parse_interval(s)
            if side == 's' and a == pos:
                return 1
            if side == 'e' and b == pos:
                return 1
        return 0

    # We'll produce final assignments: node -> sets
    node_starts = {n: set() for n in nodes}
    node_ends = {n: set() for n in nodes}

    # For each directed character (pos, side) run majority bottom-up then top-down
    for pos in positions:
        for side, out_sets in (('s', node_starts), ('e', node_ends)):
            # bottom-up: compute candidate sets S_u ∈ {{0},{1},{0,1}} for every node
            candidate = {n: None for n in nodes}

            # postorder traversal: compute order
            postorder = []
            visited = set()
            def dfs(u):
                for w in children.get(u, []):
                    if w not in visited:
                        dfs(w)
                postorder.append(u)
                visited.add(u)
            for r in roots:
                if r not in visited:
                    dfs(r)

            # leaves: fixed sets
            for n in nodes:
                if n in leaf_intervals:
                    v = leaf_char(n, pos, side)
                    candidate[n] = {v}
            # for nodes that are not leaves, candidate computed bottom-up
            for u in postorder:
                if candidate[u] is not None:
                    continue
                # gather children sets (each is a subset of {0,1})
                child_sets = [candidate[c] for c in children.get(u, [])]
                # If node has no children, treat as 0 (safe fallback)
                if not child_sets:
                    candidate[u] = {0}
                    continue
                # count how many children can be 1 vs 0
                count1 = sum(1 for S in child_sets if 1 in S)
                count0 = sum(1 for S in child_sets if 0 in S)
                if count1 > count0:
                    candidate[u] = {1}
                elif count0 > count1:
                    candidate[u] = {0}
                else:
                    candidate[u] = {0,1}  # tie -> ambiguous

            # top-down: break ties using parent assignment
            # we will record final single value val[u] in {0,1}
            final_val = {}

            # process forest roots
            for r in roots:
                S = candidate[r]
                if S == {0}:
                    val_r = 0
                elif S == {1}:
                    val_r = 1
                else:  # {0,1} -> tie at root
                    if tie_break_root == 'prefer1':
                        val_r = 1
                    elif tie_break_root == 'prefer0':
                        val_r = 0
                    elif tie_break_root == 'random':
                        val_r = random.choice([0,1])
                    else:
                        # default fallback
                        val_r = 1
                # BFS/DFS top-down propagate
                stack = [(r, val_r)]
                while stack:
                    node, pval = stack.pop()
                    final_val[node] = pval
                    for c in children.get(node, []):
                        S_c = candidate[c]
                        if pval in S_c:
                            cval = pval
                        else:
                            # pval not allowed for child (child S_c is single opposite value)
                            # choose the single value in S_c
                            # (S_c cannot be empty)
                            cval = next(iter(S_c))
                        stack.append((c, cval))

            # Now final_val contains 0/1 for every node for this (pos,side)
            for n, v in final_val.items():
                if v == 1:
                    out_sets[n].add(pos)

    # Assemble results
    node_boundaries = {}
    for n in nodes:
        node_boundaries[n] = {
            'starts': node_starts[n],
            'ends': node_ends[n]
        }
    return node_boundaries


def min_weight_edge_cover(weighted_edges):
    """
    weighted_edges: list of (u, v, w)
    returns: set of edges forming a minimum-weight edge cover
             as (u, v, w)
    """

    G = nx.Graph()
    for u, v, w in weighted_edges:
        G.add_edge(u, v, weight=w)

    # 1. Minimum-weight maximum matching
    matching = nx.algorithms.matching.min_weight_matching(
        G, weight="weight"
    )

    edge_cover = set()
    covered = set()

    # add matching edges
    for u, v in matching:
        w = G[u][v]["weight"]
        edge_cover.add((u, v, w))
        covered.add(u)
        covered.add(v)

    # 2. For uncovered vertices, add cheapest incident edge
    for v in G.nodes:
        if v not in covered:
            # choose cheapest incident edge
            u = min(G.neighbors(v), key=lambda x: G[v][x]["weight"])
            w = G[v][u]["weight"]
            edge_cover.add((v, u, w))
            covered.add(v)
            covered.add(u)

    return edge_cover

def infer_tads_bottom_up(edges, leaf_intervals, node_boundaries, Ct=1.0):
    """
    Infer TADs at every node bottom-up using min-weight edge cover.

    Returns:
      node_tads: dict node -> set of (a,b) intervals
    """

    # build children + parent maps
    children = defaultdict(list)
    parent = {}
    nodes = set()

    for u, v in edges:
        children[u].append(v)
        parent[v] = u
        nodes.add(u); nodes.add(v)

    nodes |= set(leaf_intervals.keys())

    # find roots
    roots = [n for n in nodes if n not in parent]

    # initialize TAD sets
    node_tads = {n: set() for n in nodes}

    # initialize leaves
    for v, ints in leaf_intervals.items():
        for s in ints:
            node_tads[v].add(parse_interval(s))

    # postorder traversal
    postorder = []
    visited = set()

    def dfs(u):
        for c in children.get(u, []):
            if c not in visited:
                dfs(c)
        postorder.append(u)
        visited.add(u)

    for r in roots:
        dfs(r)

    # process internal nodes bottom-up
    for u in postorder:
        if u in leaf_intervals:
            continue  # already assigned

        S = sorted(node_boundaries[u]['starts'])
        E = sorted(node_boundaries[u]['ends'])
        ch = children.get(u, [])

        if not S or not E or not ch:
            continue

        weighted_edges = []

        # build candidate intervals
        for a in S:
            for b in E:
                if a >= b:
                    continue

                # compute x_v for each child
                present = 0
                for v in ch:
                    if (a, b) in node_tads[v]:
                        present += 1

                k = len(ch)
                w = Ct * (k - 2 * present)

                weighted_edges.append((('s', a), ('e', b), w))

        if not weighted_edges:
            continue

        # compute edge cover
        edge_cover = min_weight_edge_cover(weighted_edges)

        # add all negative edges
        final_edges = set(edge_cover)
        for u0, v0, w in weighted_edges:
            if w < 0:
                final_edges.add((u0, v0, w))

        # decode intervals
        for x, y, w in final_edges:
            if x[0] == 's' and y[0] == 'e':
                a, b = x[1], y[1]
            elif x[0] == 'e' and y[0] == 's':
                a, b = y[1], x[1]
            else:
                continue
            if a < b:
                node_tads[u].add((a, b))

    return node_tads
def boundaries_from_tads(tads):
    starts = {a for (a, b) in tads}
    ends   = {b for (a, b) in tads}
    return starts, ends


def compute_parsimony_score(edges, node_tads, Ct=1.0, Cb=1.0):
    total = 0
    per_edge = {}

    for u, v in edges:
        Tu = node_tads.get(u, set())
        Tv = node_tads.get(v, set())

        # --- TAD RF ---
        rf_tad = len(Tu - Tv) + len(Tv - Tu)
        tad_cost = Ct * rf_tad

        # --- Boundary RF (derived) ---
        Su, Eu = boundaries_from_tads(Tu)
        Sv, Ev = boundaries_from_tads(Tv)

        rf_boundary = (
            len(Su - Sv) + len(Sv - Su)
            + len(Eu - Ev) + len(Ev - Eu)
        )
        boundary_cost = Cb * rf_boundary

        edge_cost = tad_cost + boundary_cost
        total += edge_cost
        per_edge[(u, v)] = {
            "tad_rf": rf_tad,
            "boundary_rf": rf_boundary,
            "cost": edge_cost
        }

    return total, per_edge

def brute_force(edges, leaf_intervals, Ct=1.0, Cb=1.0):
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
                if (a < c <= b < d) or (c < a <= d < b):
                    return False
        return True


    @lru_cache(None)
    def state_cost(frozen_u, frozen_v):
        Tu = set(frozen_u)
        Tv = set(frozen_v)

        # RF on TADs
        rf_tad = len(Tu - Tv) + len(Tv - Tu)

        # RF on boundaries (start/end distinct)
        Su, Eu = boundaries_from_tads(Tu)
        Sv, Ev = boundaries_from_tads(Tv)

        rf_boundary = (
            len(Su - Sv) + len(Sv - Su)
            + len(Eu - Ev) + len(Ev - Eu)
        )

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




edges = [(0,1),(0,2),(1,3),(1,4)]
# leaf_intervals = {
#     2: {"[1,5]", "[6,10]"},
#     3: {"[4,5]", "[6,10]"},
#     4: {"[1,10]"},
# }

# -----------------------------
# Example 1: Infer internal node TADs from leaf TADs only
# -----------------------------

def run_inference_mode(edges, leaf_files, chrom="1", Ct=1.5, Cb=2.0, output_file="phylogeny_inferred.png"):
    """
    Infer TADs at internal nodes from leaf TADs using bottom-up inference.
    
    Parameters:
    -----------
    edges : list of tuples
        Tree edges as (parent, child) pairs
    leaf_files : dict
        Mapping of leaf node ID -> bedpe file path
    chrom : str
        Chromosome to extract (e.g., "1" or "chr1")
    Ct, Cb : float
        Cost parameters for parsimony
    output_file : str
        Output filename for visualization
    """
    # Read leaf intervals from bedpe files
    leaf_intervals = {
        leaf: read_bedpe_intervals(bedpe, chrom=chrom)
        for leaf, bedpe in leaf_files.items()
    }
    
    # Infer boundaries and TADs at internal nodes
    node_boundaries = infer_boundaries_by_majority(edges, leaf_intervals, tie_break_root='prefer1')
    node_tads = infer_tads_bottom_up(edges, leaf_intervals, node_boundaries)
    
    # Compute parsimony score
    total_score, per_edge_scores = compute_parsimony_score(edges, node_tads, Ct=Ct, Cb=Cb)
    print("Total parsimony score (inferred):", total_score)
    
    # Visualize
    visualize_phylogeny_with_tads(edges, node_tads, per_edge_scores, total_score, output_file=output_file)
    
    return node_tads, total_score, per_edge_scores


# -----------------------------
# Example 2: Use provided TADs for all nodes
# -----------------------------

def run_provided_mode(edges, node_files, chrom="1", Ct=1.5, Cb=2.0, output_file="phylogeny_provided.png"):
    """
    Use provided TAD assignments for all nodes (no inference).
    
    Parameters:
    -----------
    edges : list of tuples
        Tree edges as (parent, child) pairs
    node_files : dict
        Mapping of node ID -> bedpe file path (for ALL nodes, including internal)
    chrom : str
        Chromosome to extract (e.g., "1" or "chr1")
    Ct, Cb : float
        Cost parameters for parsimony
    output_file : str
        Output filename for visualization
    """
    # Read TAD intervals for all nodes from bedpe files
    node_tads = {}
    for node_id, bedpe_file in node_files.items():
        intervals = read_bedpe_intervals(bedpe_file, chrom=chrom)
        # Convert string intervals to tuples
        node_tads[node_id] = {parse_interval(s) for s in intervals}
    
    # Compute parsimony score
    total_score, per_edge_scores = compute_parsimony_score(edges, node_tads, Ct=Ct, Cb=Cb)
    print("Total parsimony score (provided):", total_score)
    
    # Print TAD assignments for each node
    print("\nTAD assignments per node:")
    for n in sorted(node_tads.keys()):
        print(f"  Node {n}: {len(node_tads[n])} TADs")
    
    # Visualize
    visualize_phylogeny_with_tads(edges, node_tads, per_edge_scores, total_score, output_file=output_file)
    
    return node_tads, total_score, per_edge_scores


# -----------------------------
# Main: Choose which mode to run
# -----------------------------

if __name__ == "__main__":
    # Define tree structure
    # edges = [(0,1),(1,2),(2,3),(2,4),(3,5),(3,6),(5,7),(7,8),(6,9)]
    edges = [(0,1),(0,2),(1,3),(1,4)]
    # Set cost parameters
    Ct = 1.5
    Cb = 2.0
    
    # Choose mode: 'inference' or 'provided'
    MODE = 'inference'  # Change to 'provided' to use bedpe files for all nodes
    
    if MODE == 'inference':
        # Mode 1: Infer internal nodes from leaf TADs
        leaf_files = {
            # 8: "hic_matrix/tads/MK.bedpe",
            # 9: "hic_matrix/tads/G.bedpe",
            # 4: "hic_matrix/tads/CLP.bedpe",
            2: "hic_matrix/tads/CLP.bedpe",
            3: "hic_matrix/tads/MK.bedpe",
            4: "hic_matrix/tads/G.bedpe",
        }
        run_inference_mode(edges, leaf_files, chrom="1", Ct=Ct, Cb=Cb, 
                          output_file="phylogeny_inferred_terminal.png")
    
    elif MODE == 'provided':
        # Mode 2: Use provided TADs for all nodes
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
            0: "hic_matrix/tads/MPP.bedpe",  
            1: "hic_matrix/tads/CMP.bedpe",  
            2: "hic_matrix/tads/CLP.bedpe",    
            3: "hic_matrix/tads/MK.bedpe",    
            4: "hic_matrix/tads/G.bedpe",
        }
        run_provided_mode(edges, node_files, chrom="1", Ct=Ct, Cb=Cb,
                         output_file="phylogeny_provided_terminal.png")
    
    else:
        print(f"Unknown MODE: {MODE}. Use 'inference' or 'provided'.")
