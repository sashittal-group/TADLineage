"""Tree topology and default simulation parameters.

The tree is the simple branching topology fixed for the first experiment:

    root
     └── A
          ├── B
          │    ├── D
          │    └── E
          └── C
               ├── F
               └── G

Given nodes for reconstruction: root + the four leaves (D, E, F, G).
Reconstructed / scored nodes: the three internal branch points A, B, C.

The evolved (ground-truth) hierarchy is generated at *every* node and is used
only to synthesize that node's Hi-C matrix. Evaluation is defined by OnTAD run
on the simulated matrices, exactly as in the real-data protocol:
  * OnTAD on root + leaf matrices  -> reconstruction inputs
  * OnTAD on A / B / C matrices    -> evaluation ground truth
"""

from __future__ import annotations

from dataclasses import dataclass


ROOT = "root"
EDGES = {
    "root": ["A"],
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
}
LEAVES = ["D", "E", "F", "G"]
INTERNAL = ["A", "B", "C"]          # non-root internal nodes
SCORE_NODES = ["A", "B", "C"]       # every internal node is a branch point -> informative
GIVEN_NODES = ["root"] + LEAVES     # supplied to the reconstruction step

# Deterministic top-down traversal order (parents always precede children).
NODE_ORDER = ["root", "A", "B", "C", "D", "E", "F", "G"]


def parent_of() -> dict[str, str]:
    """Map every non-root node to its parent."""
    parent = {}
    for p, kids in EDGES.items():
        for k in kids:
            parent[k] = p
    return parent


@dataclass
class SimParams:
    """All knobs for one generation run.

    Geometry / hierarchy
    --------------------
    n_bins        : matrix size N (bins per region)
    resolution_bp : bin size in bp (only used for bookkeeping / bed output)
    n_top_tads    : approximate number of top-level domains at the root
    nesting_depth : maximum nesting depth of the root hierarchy
    min_size      : minimum domain size in bins
    min_score_node_tads : minimum hidden TAD count required at every scored
                      internal node (A/B/C). Regions are regenerated until
                      this condition holds.

    Evolution (parsimony)
    ---------------------
    events_per_edge : integer event budget applied on every parent->child edge.
                      Small k => most boundaries conserved across edges.
    shift_delta     : max magnitude (bins) of a shift_boundary move.

    Lun/Smyth matrix recipe
    -----------------------
    A            : contact amplitude on the diagonal
    alpha        : distance-decay exponent, lambda ~ (|i-j|+1)^(-alpha)
    enrichment   : intra-domain multiplier (compounds for nested domains)
    boundary_depletion : multiplier applied to contacts crossing adjacent
                 sibling-domain boundaries. Values below 1 create insulation.
    boundary_stripe_width : half-width in bins for boundary insulation stripes.
    boundary_stripe_depletion : multiplier for stripe contacts crossing each
                 domain boundary. This creates local minima OnTAD can detect.
    domain_strength_jitter : lognormal sigma for per-domain enrichment
                 variation, preventing unrealistically uniform repeated blocks.
    bcv          : biological coefficient of variation for the NegBinom noise
    noise        : random-ligation fraction rho (uniform background)
    depth        : target total counts D per matrix; None => no rescaling

    Replication / evaluation
    ------------------------
    n_regions : number of independent regions (analogous to autosomes)
    tolerance : boundary match tolerance t (bins) for downstream P/R/F1
    seed      : master RNG seed
    """

    # geometry / hierarchy
    n_bins: int = 1500
    resolution_bp: int = 40_000
    n_top_tads: int = 35
    nesting_depth: int = 2
    min_size: int = 20
    gap_prob: float = 0.0           # inter-TAD gap frequency; low => domains mostly tile
    min_score_node_tads: int = 20
    max_generation_attempts: int = 200

    # evolution
    events_per_edge: int = 3
    shift_delta: int = 4

    # matrix recipe
    A: float = 2000.0
    alpha: float = 1.0
    enrichment: float = 18.0    # per-level intra-domain boost (compounds for nested domains)
    boundary_depletion: float = 0.001
    boundary_stripe_width: int = 2
    boundary_stripe_depletion: float = 0.02
    domain_strength_jitter: float = 0.35
    bcv: float = 0.02
    noise: float = 0.001
    depth: float | None = None

    # replication / evaluation
    n_regions: int = 5
    tolerance: int = 2
    seed: int = 12345


DEFAULT_PARAMS = SimParams()
