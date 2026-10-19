#!/usr/bin/env python3
"""Quality-control report for generated simulation data.

Prints the reference lines called out in the design, aggregated over regions:

  1. Truth parsimony        -- boundary Jaccard(parent, child) in the *evolved*
                               hierarchies. Should be high (most structure
                               conserved across edges), by construction.
  2. OnTAD recovery         -- boundary P/R/F1 of OnTAD calls vs the evolved
                               truth, per node. The leaf numbers are the
                               reference line: how much error is calling (not
                               reconstruction). Requires run_ontad.py first.
  3. OnTAD-call parsimony   -- boundary Jaccard(parent, child) of the OnTAD
                               *calls*. This is the signal the reconstruction
                               must exploit; OnTAD calling noise lowers it
                               relative to the truth parsimony above.

This is data QC only. It does not run reconstruction (the next phase).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent          # simulation/ (shared generation + data)
sys.path.insert(0, str(SIM_ROOT))     # shared config.py/hierarchy.py/metrics.py

import config
from hierarchy import boundaries, load_hierarchy
from metrics import boundary_prf, jaccard, parse_tad_boundaries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=SIM_ROOT / "data")
    p.add_argument("--tolerance", type=int, default=config.DEFAULT_PARAMS.tolerance)
    args = p.parse_args()

    region_dirs = sorted(args.outdir.glob("region_*"))
    if not region_dirs:
        raise SystemExit(f"No region_* directories under {args.outdir}")
    tol = args.tolerance
    parent = config.parent_of()
    ontad_paths = [
        rd / "ontad" / f"{n}.tad"
        for rd in region_dirs
        for n in config.NODE_ORDER
    ]
    have_any_ontad = any(p.exists() for p in ontad_paths)
    have_ontad = all(p.exists() for p in ontad_paths)

    truth_pars: dict[str, list[float]] = {c: [] for c in parent}
    call_pars: dict[str, list[float]] = {c: [] for c in parent}
    recovery: dict[str, list[tuple]] = {n: [] for n in config.NODE_ORDER}

    for rd in region_dirs:
        truth_b = {
            n: boundaries(load_hierarchy(rd / "truth" / f"{n}.json"))
            for n in config.NODE_ORDER
        }
        call_b = {}
        if have_ontad:
            for n in config.NODE_ORDER:
                call_b[n] = parse_tad_boundaries(rd / "ontad" / f"{n}.tad")

        for child, par in parent.items():
            truth_pars[child].append(jaccard(truth_b[par], truth_b[child]))
            if have_ontad:
                call_pars[child].append(jaccard(call_b[par], call_b[child]))
        if have_ontad:
            for n in config.NODE_ORDER:
                recovery[n].append(boundary_prf(call_b[n], truth_b[n], tol))

    n_regions = len(region_dirs)
    print(f"Regions: {n_regions}   tolerance: +/-{tol} bins   OnTAD calls complete: {have_ontad}\n")

    print("1. Truth parsimony  (evolved hierarchies, boundary Jaccard parent->child)")
    for child, par in parent.items():
        print(f"     {par:>4} -> {child:<4}  J = {np.mean(truth_pars[child]):.2f}")
    all_truth = [v for vals in truth_pars.values() for v in vals]
    print(f"     mean over all edges: {np.mean(all_truth):.2f}\n")

    if not have_ontad:
        if have_any_ontad:
            print("(OnTAD calls are partially present; complete run_ontad.py/Snakemake to get OnTAD recovery + call parsimony)")
        else:
            print("(run run_ontad.py to get OnTAD recovery + call parsimony)")
        return

    print("2. OnTAD recovery of evolved truth  (boundary P / R / F1)")
    for n in config.NODE_ORDER:
        arr = np.array(recovery[n])
        role = "given " if n in config.GIVEN_NODES else "scored"
        print(f"     {n:>4} [{role}]  P={arr[:,0].mean():.2f}  R={arr[:,1].mean():.2f}  F1={arr[:,2].mean():.2f}")
    leaf_arr = np.array([m for n in config.LEAVES for m in recovery[n]])
    print(f"     leaf reference line: P={leaf_arr[:,0].mean():.2f}  "
          f"R={leaf_arr[:,1].mean():.2f}  F1={leaf_arr[:,2].mean():.2f}\n")

    print("3. OnTAD-call parsimony  (boundary Jaccard parent->child of OnTAD calls)")
    for child, par in parent.items():
        print(f"     {par:>4} -> {child:<4}  J = {np.mean(call_pars[child]):.2f}")
    all_call = [v for vals in call_pars.values() for v in vals]
    print(f"     mean over all edges: {np.mean(all_call):.2f}")


if __name__ == "__main__":
    main()
