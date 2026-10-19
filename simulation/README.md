# TAD-reorganization simulation

Simulated Hi-C along a small phylogenetic tree with **parsimoniously-evolving
TAD hierarchies**, for benchmarking internal-node TAD reconstruction (as in the
`hemato_cells` real-data work) with precision / recall / F1.

The generation of matrices and hidden truth is **shared**; the TAD-caller-
specific analysis is split into two sibling folders:

```
simulation/
  Snakefile              shared generation (matrices + hidden truth) only
  config.py generate_data.py simulate_matrix.py evolve.py hierarchy.py metrics.py
  data/                  shared: region_XXX/{matrices,truth}/ + per-caller calls
  using_ontad/           OnTAD-caller analysis  (see using_ontad/README.md)
  using_robustad/        RobusTAD-caller analysis (see using_robustad/README.md)
```

Both analyses read the same `data/region_XXX/matrices/<node>.matrix`. Each writes
its own calls next to the matrices (`data/region_XXX/ontad/` vs
`data/region_XXX/robustad/`) and keeps its reconstruction outputs
(`method_results/`, `method_plots/`) inside its own folder.

## Tree

```
root
 └── A
      ├── B
      │    ├── D
      │    └── E
      └── C
           ├── F
           └── G
```

* **Given nodes** (reconstruction inputs): `root`, `D`, `E`, `F`, `G`.
* **Scored nodes** (reconstructed & evaluated): `A`, `B`, `C`.

Topology and defaults live in [`config.py`](config.py).

## Run order

```bash
# 1. Shared generation (matrices + hidden truth), once
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake --cores 8

# 2a. OnTAD analysis
cd using_ontad     && snakemake --cores 8 && snakemake -s Snakefile_methods --cores 8

# 2b. RobusTAD analysis (independent of 2a)
cd ../using_robustad && snakemake --cores 8 && snakemake -s Snakefile_methods --cores 8
```

The two caller analyses are independent and may be run in either order (or only
one), as long as shared generation (step 1) has run first. See each subfolder's
`README.md` for details, interpreters, and the design notes.

## Difference between the two analyses

The RobusTAD analysis is almost identical to the OnTAD one, with one substantive
change: RobusTAD emits a per-bin left/right **boundary score** in addition to a
per-TAD score, so its `inferred_given_root_weighted` method uses the
boundary-aware ILP `check_lp_constrained_weighted_updated.py` (leaf importance
`{x=TADscore, s=leftScore, e=rightScore}`), whereas the OnTAD analysis uses
`check_lp_constrained_weighted.py`. See
[`using_robustad/README.md`](using_robustad/README.md).
