#!/usr/bin/env python3
"""Generate simulated Hi-C data + hidden evolved hierarchies for the tree.

For every independent region (a.k.a. instance):
  1. generate a random laminar root hierarchy,
  2. evolve it top-down along the tree (small per-edge budget => parsimony),
  3. simulate a Hi-C matrix at every node from its evolved hierarchy,
  4. write the matrix and the evolved hierarchy.

The evolved hierarchy is the hidden generator, not the final benchmark object.
The experiment runs OnTAD on every simulated matrix after this step:
  * OnTAD(root + leaves) -> reconstruction inputs
  * OnTAD(A / B / C)     -> internal-node evaluation ground truth

The hidden evolved hierarchies are retained for QC: they verify that the
simulated biology is parsimonious and let us measure OnTAD recovery.

Output layout (under --outdir, default ./data):
  meta.json                          tree topology + params + node roles
  region_XXX/matrices/<node>.matrix  dense N x N counts
  region_XXX/truth/<node>.tad        hidden evolved hierarchy, OnTAD .tad layout
  region_XXX/truth/<node>.json       hidden evolved hierarchy, canonical JSON

Two modes:
  * default (no --region-index): generate --n-regions regions in one call
    (interactive / manual use).
  * --region-index IDX: generate exactly region IDX and nothing else -- this
    is the mode the Snakefile uses to fan out instances across jobs. Combine
    with --meta-only (no region generated) to (re)write meta.json alone.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

import numpy as np

import config
from evolve import evolve
from hierarchy import (
    all_domains,
    boundaries,
    generate_root,
    hierarchy_to_json,
    save_hierarchy,
    write_tad,
)
from simulate_matrix import simulate_matrix, write_matrix


def region_dir_name(region_idx: int) -> str:
    return f"region_{region_idx:03d}"


def build_region(region_idx: int, params: config.SimParams):
    """Evolve every node's hierarchy for one region. Returns {node: hierarchy}."""
    base_seed = params.seed * 1_000_003 + region_idx
    for attempt in range(params.max_generation_attempts):
        rng = random.Random(base_seed + attempt * 9_176)
        hiers: dict[str, list] = {}
        hiers[config.ROOT] = generate_root(
            params.n_bins,
            params.n_top_tads,
            params.nesting_depth,
            params.min_size,
            rng,
            gap_prob=params.gap_prob,
        )
        for parent in config.NODE_ORDER:
            for child in config.EDGES.get(parent, []):
                hiers[child] = evolve(hiers[parent], params.events_per_edge, params, rng)

        score_counts = {node: len(all_domains(hiers[node])) for node in config.SCORE_NODES}
        if min(score_counts.values()) >= params.min_score_node_tads:
            return hiers

    counts = ", ".join(f"{node}={count}" for node, count in score_counts.items())
    raise RuntimeError(
        f"Could not generate region_{region_idx:03d} with at least "
        f"{params.min_score_node_tads} hidden TADs in every score node after "
        f"{params.max_generation_attempts} attempts; last counts: {counts}"
    )


def write_meta(params: config.SimParams, outdir: Path) -> None:
    meta = {
        "root": config.ROOT,
        "edges": config.EDGES,
        "leaves": config.LEAVES,
        "internal": config.INTERNAL,
        "score_nodes": config.SCORE_NODES,
        "given_nodes": config.GIVEN_NODES,
        "node_order": config.NODE_ORDER,
        "params": dataclasses.asdict(params),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))


def generate_region(region_idx: int, params: config.SimParams, outdir: Path) -> list[dict]:
    """Generate one region (all 8 nodes). Returns its summary rows."""
    region_dir = outdir / region_dir_name(region_idx)
    mat_dir = region_dir / "matrices"
    truth_dir = region_dir / "truth"
    mat_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    hiers = build_region(region_idx, params)
    mat_rng = np.random.default_rng(params.seed * 7 + region_idx)

    rows = []
    for node in config.NODE_ORDER:
        hier = hiers[node]
        write_tad(hier, params.n_bins, truth_dir / f"{node}.tad")
        save_hierarchy(hier, truth_dir / f"{node}.json")
        M = simulate_matrix(hier, params.n_bins, params, mat_rng)
        write_matrix(M, mat_dir / f"{node}.matrix")
        rows.append(
            {
                "region": region_idx,
                "node": node,
                "n_domains": len(all_domains(hier)),
                "n_boundaries": len(boundaries(hier)),
                "depth": int(M.sum()),
            }
        )
    return rows


def generate(params: config.SimParams, outdir: Path) -> None:
    """Generate params.n_regions regions in one call (interactive / manual use)."""
    write_meta(params, outdir)

    summary_rows = []
    for r in range(params.n_regions):
        summary_rows.extend(generate_region(r, params, outdir))
        print(f"{region_dir_name(r)}: wrote 8 matrices + truth", flush=True)

    (outdir / "generation_summary.json").write_text(json.dumps(summary_rows, indent=2))

    # Compact console summary.
    print("\nGeneration summary (domains per node, averaged over regions):")
    for node in config.NODE_ORDER:
        vals = [row["n_domains"] for row in summary_rows if row["node"] == node]
        role = (
            "given" if node in config.GIVEN_NODES else "scored"
        )
        print(f"  {node:>4} [{role}]  avg domains = {np.mean(vals):.1f}")
    print(f"\nWrote {len(summary_rows)} matrices to {outdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    d = config.DEFAULT_PARAMS
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "data")
    p.add_argument("--region-index", type=int, default=None,
                    help="generate exactly this one region (for Snakemake fan-out)")
    p.add_argument("--meta-only", action="store_true",
                    help="write meta.json only, generate no region")
    p.add_argument("--n-bins", type=int, default=d.n_bins)
    p.add_argument("--n-regions", type=int, default=d.n_regions)
    p.add_argument("--n-top-tads", type=int, default=d.n_top_tads)
    p.add_argument("--nesting-depth", type=int, default=d.nesting_depth)
    p.add_argument("--min-size", type=int, default=d.min_size)
    p.add_argument("--gap-prob", type=float, default=d.gap_prob)
    p.add_argument("--min-score-node-tads", type=int, default=d.min_score_node_tads)
    p.add_argument("--max-generation-attempts", type=int, default=d.max_generation_attempts)
    p.add_argument("--events-per-edge", type=int, default=d.events_per_edge)
    p.add_argument("--shift-delta", type=int, default=d.shift_delta)
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--enrichment", type=float, default=d.enrichment)
    p.add_argument("--boundary-depletion", type=float, default=d.boundary_depletion)
    p.add_argument("--boundary-stripe-width", type=int, default=d.boundary_stripe_width)
    p.add_argument("--boundary-stripe-depletion", type=float, default=d.boundary_stripe_depletion)
    p.add_argument("--domain-strength-jitter", type=float, default=d.domain_strength_jitter)
    p.add_argument("--bcv", type=float, default=d.bcv)
    p.add_argument("--noise", type=float, default=d.noise)
    p.add_argument("--amplitude", type=float, default=d.A, dest="A")
    p.add_argument("--depth", type=float, default=d.depth)
    p.add_argument("--tolerance", type=int, default=d.tolerance)
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params = config.SimParams(
        n_bins=args.n_bins,
        resolution_bp=config.DEFAULT_PARAMS.resolution_bp,
        n_top_tads=args.n_top_tads,
        nesting_depth=args.nesting_depth,
        min_size=args.min_size,
        gap_prob=args.gap_prob,
        min_score_node_tads=args.min_score_node_tads,
        max_generation_attempts=args.max_generation_attempts,
        events_per_edge=args.events_per_edge,
        shift_delta=args.shift_delta,
        A=args.A,
        alpha=args.alpha,
        enrichment=args.enrichment,
        boundary_depletion=args.boundary_depletion,
        boundary_stripe_width=args.boundary_stripe_width,
        boundary_stripe_depletion=args.boundary_stripe_depletion,
        domain_strength_jitter=args.domain_strength_jitter,
        bcv=args.bcv,
        noise=args.noise,
        depth=args.depth,
        n_regions=args.n_regions,
        tolerance=args.tolerance,
        seed=args.seed,
    )

    if args.meta_only:
        write_meta(params, args.outdir)
        print(f"Wrote {args.outdir / 'meta.json'}", flush=True)
    elif args.region_index is not None:
        generate_region(args.region_index, params, args.outdir)
        print(f"{region_dir_name(args.region_index)}: wrote 8 matrices + truth", flush=True)
    else:
        generate(params, args.outdir)


if __name__ == "__main__":
    main()
