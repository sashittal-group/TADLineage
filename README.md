# TADLineage: TAD reorganization across cell states

> **Draft README.** Paths, environment names and run instructions reflect the development machine and
> will be generalized before release.

This repository holds the code for TADLineage, a set of algorithms for comparing and reconstructing
hierarchies of topologically associating domains (TADs) across related cell states. Given the TADs
called at the observed cell states of a tree (a differentiation lineage, a phylogeny, or a simulated
tree), TADLineage

- reconstructs the TAD hierarchies of the unobserved ancestral cell states under a parsimony objective
  (small parsimony),
- infers the tree itself from the observed TAD hierarchies (large parsimony),
- computes median TAD hierarchies and clusters single cells by their TAD hierarchies, and
- identifies a largest consistent set of TAD merge and split events between two hierarchies by maximum flow.

The repository contains code only. Hi-C data, TAD calls, intermediate results and figures are not
included.

## Repository layout

### Core algorithms (repository root)

| File | Purpose |
|---|---|
| `check_lp_constrained.py` | Small parsimony integer linear program: minimizes TAD changes (`Ct`) plus boundary changes (`Cb`) over the tree edges, with laminarity constraints on the internal nodes. Supports a pinned root, restricting candidate TADs, and imputation at observed nodes. |
| `check_lp_constrained_weighted.py` | The same program with a per-node importance term (used with OnTAD calls). |
| `check_lp_constrained_weighted_updated.py` | Boundary-aware variant that uses per-TAD and per-boundary scores (used with RobusTAD calls). |
| `nussinov*.py` | Nussinov-style dynamic program for the same objective; `nussinov_with_imputation_weighted_fast.py` is the version the clustering pipelines use. |
| `rf_distance_fitch_hartigan.py`, `rf_distance_fitch_hartigan_weigthed.py` | Fitch–Hartigan and Sankoff baselines, treating each TAD as a binary character. |
| `tad_utilities.py` | Reading and writing TAD files, tolerance matching, laminarity checks, relaxed TAD and boundary distances. |
| `filtering.py` | `harmonize_leaf_boundaries`: snaps near-identical boundaries across input files to a shared position. |
| `cluster_sankey.py` | Shared alluvial (Sankey) plotting for cluster-versus-stage comparisons. |

Root-level `check_*.py`, `random_simulation.py` and `boundary_given_case_sim.py` are exploratory
experiments on the integrality of the linear program relaxation (see `SIMULATION_README.md`).

### Experiments

| Folder | Contents |
|---|---|
| `simulation/` | Synthetic Hi-C on a fixed tree. The root `Snakefile` generates contact matrices and the hidden true TADs; `using_ontad/` and `using_robustad/` call TADs and compare the reconstruction methods; `data_quality_sweep/` measures accuracy as resolution and sequencing depth decrease; `tree_informed_tad_inference_{ontad,robustad}/` repairs noisy TAD calls at the leaves using the tree; `tgif_benchmark/` compares against TGIF-DB. |
| `hemato_cells/` | Mouse hematopoietic lineage (LT-HSC → MPP → {CLP, CMP → …}). `using_ontad/` and `using_robustad/` reconstruct MPP and CMP and score them; `large_parsimony/` infers the lineage tree from the observed cell types; `tad_pathway_analysis/` runs gene set enrichment on reorganized TADs; `diffdomainanalysis*/` and `tadsplimer/` cross-check merge and split calls against external tools. |
| `scptr_robustad/`, `scptr_ontad/` | Single-cell Hi-C of early mouse embryos (1- to 64-cell stages): EM clustering of cells by TAD distance, median TAD hierarchies per cluster, and trees over the medians. `scptr_robustad/merge_split_real_biology/` analyzes merge, split and boundary events between stages. |
| `higashi_scptr/`, `schiccluster_scptr/`, `schicptr_scptr/` | Baselines on the same embryo data: Higashi, scHiCluster and scHiCPTR. |
| `robustad_predictions*/`, `refhic_predictions/`, `mouse_neuron/`, `scptr/` | TAD calling runs and data preparation for the datasets above. |

Most folders have their own `README.md` with a single-case example.

## Conventions

- **Intervals** are written `"[a,b]"` in files and held as `(a, b)` tuples in memory. Coordinates are
  1-based **bin indices**, not base pairs.
- **TAD files** follow `<NODE>/chr<C>/<RESOLUTION>/tads.tad` (for example `MEP/chr1/500kb/tads.tad`),
  with `boundaries.bed` alongside when the caller reports boundary scores. Harmonized and imputed
  variants are `tads_filtered.tad` and `tads_filtered_imputed.tad`.
- **`--tolerance`** (in bins) means the same thing everywhere: two boundaries within the tolerance are
  treated as one boundary, both for harmonization and for scoring.
- The reconstruction methods compared throughout are `given_root` (TADLineage), `given_root_weighted`
  (weighted TADLineage), `rf` (Fitch–Hartigan), `rf_weighted` (Sankoff), `union` and `intersection`.
  In the simulation folders `given_root_weighted` is a deliberately biased baseline that uses the
  held-out truth to calibrate its weights; every other method is scored without access to the truth.

## Requirements

Different steps need incompatible environments, so scripts and Snakefiles call interpreters by
absolute path (`PYTHON = ...` at the top of each Snakefile). Adjust these paths for your machine.

| Environment | Used for |
|---|---|
| Snakemake, Gurobi (`gurobipy`), pandas, matplotlib, seaborn | Workflows, every integer linear program, plotting. A Gurobi license is required. |
| `cooler` | RobusTAD and RefHiC TAD calling |
| OnTAD | OnTAD TAD calling |
| Higashi, scHiCluster, scHiCPTR, TADsplimer | The corresponding baselines |

Third-party tools (RobusTAD, RefHiC, OnTAD, Higashi, scHiCluster, scHiCPTR, TADCompare, dcHiC,
diffDomain, TADsplimer, Juicer, TGIF) are not vendored. Clone them into the repository root; scripts
that use RobusTAD add `<repo>/RobusTAD` to `PYTHONPATH`.

## Running a workflow

Each experiment folder has its own `Snakefile` and is run from inside that folder, for example

```bash
cd simulation/using_ontad
snakemake --cores 8 -p --rerun-incomplete
```

Folders with several workflow files (`Snakefile_methods`, `Snakefile_updated`,
`check_intermediate_*_snakemake`) are selected with `-s`. Cap `--cores` at about 4 for jobs that run
many Gurobi solves at once. Every script is also an argparse command-line tool and can be run directly.

Some pipelines run in two stages (`--stage call` in the TAD caller's environment, then `--stage score`
in the Gurobi environment). Both stages rebuild the same synthetic truth from the same seed, so they
must receive identical simulation parameters.

## Known pitfalls

- **Snakemake late binding.** Rules are written out one by one rather than generated in a Python
  `for` loop. A loop-generated rule with a bare-variable `shell:` body runs the last iteration's
  command for every rule.
- **Determinism across stages.** Seeds use `zlib.crc32(name)`, never Python's `hash()`, which is salted
  per process.
- **Imputation cost.** With `impute=True`, `impute_cost` must be positive; otherwise the solver adds an
  interval to a parent and all its children at no cost and floods every leaf.
- **Genome assembly.** `genome.chrom.sizes` holds mm9 chromosome lengths, but the single-cell embryo
  data are mm10. Genome-wide ("all") runs that lay chromosomes end to end with this file misplace
  contacts and TADs near the ends of chromosomes that are longer in mm10.

## Status

Research code accompanying a manuscript in preparation. Interfaces may change.
