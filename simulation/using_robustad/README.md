# Simulation analysis — RobusTAD caller

Parallel to [`../using_ontad`](../using_ontad), but the observed TAD hierarchies
are called by **RobusTAD** instead of OnTAD, and the weighted reconstruction uses
the **boundary-score-aware ILP** (`check_lp_constrained_weighted_updated.py`),
because RobusTAD emits a left/right boundary score per bin in addition to a
per-TAD score.

The simulated matrices and hidden truth are **shared** and produced once by
[`../Snakefile`](../Snakefile); this analysis only adds RobusTAD calls (into the
shared `../data/region_XXX/robustad/`) and its own reconstruction outputs here.

## Difference from the OnTAD analysis

Everything mirrors `using_ontad` except:

| | `using_ontad` | `using_robustad` |
|---|---|---|
| caller | OnTAD (`run_ontad.py`) | RobusTAD (`run_robustad.py`, dense→cooler→boundary+assembly) |
| calls stored in | `data/region_XXX/ontad/<node>.tad` | `data/region_XXX/robustad/<node>_*_with_bins.tsv` |
| TAD weight | OnTAD TADscore (col 5) | RobusTAD TAD `score` |
| boundary score | *(none)* | RobusTAD `leftScore` / `rightScore` per bin |
| `inferred_given_root_weighted` ILP | `check_lp_constrained_weighted` | **`check_lp_constrained_weighted_updated`** (uses `leaf_importance={x,s,e}`) |

As in `using_ontad`, `inferred_given_root_weighted` is a **deliberately biased**
baseline, not an honest weighted reconstruction: it runs the same
`check_lp_constrained_weighted_updated` ILP that RobusTAD's real per-leaf-interval
importance `{x=TADscore, s=leftScore[a], e=rightScore[b]}` would otherwise feed
(as in `hemato_cells/check_intermediate_given_root_weighted_updated.py`), but
substitutes a synthetic, calibrated importance built from the held-out A/B/C
truth (see `build_calibrated_leaf_importance` in `run_simulation_methods.py`).
This shows that TAD/boundary *weighting* can lift given_root's F1 in a
controlled way. Because this ILP only injects importance on the edge between a
node and its *leaf* children, only B (children D, E) and C (children F, G) get
a direct bias signal; A improves only indirectly, through the parsimony cost
that pulls it toward its now-more-accurate children. The honest weighted
signal (real RobusTAD TAD scores, no truth leakage) is still exercised by
`inferred_rf_weighted` (the Sankoff method).

## Prerequisites

RobusTAD needs `cooler` and its own dependencies. Two interpreters are used:

* **RobusTAD calling** (`run_robustad.py`, `check_robustad_counts.py`) — an env
  with `cooler` + RobusTAD deps, e.g. `refhic`:
  `/Users/PSLabMonisha/miniforge3/envs/refhic/bin/python`. The RobusTAD package
  is imported from `../../RobusTAD` via `PYTHONPATH` (handled inside the script).
* **Reconstruction** (`run_simulation_methods.py`, `plot_simulation_methods.py`) —
  the gurobi env, `snakemake-tutorial`.

Both are set in the two Snakefiles.

## Pipeline

```bash
# 1. Generate shared matrices + hidden truth (once, in the parent directory)
cd ..
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake --cores 8

# 2. Call RobusTAD on every matrix (this directory)
cd using_robustad
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake --cores 8

# 3. Reconstruct A/B/C with all methods + plot
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake \
    -s Snakefile_methods --cores 8
```

### Manual / single instance

```bash
# RobusTAD on one region/node (or omit --node for all 8 nodes)
/Users/PSLabMonisha/miniforge3/envs/refhic/bin/python run_robustad.py \
    --region-index 0 --node D

# reconstruct one region with one method
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python \
    run_simulation_methods.py --region-index 0 \
    --method inferred_given_root_weighted --outdir method_results/inferred_given_root_weighted
```

`run_robustad.py` converts each dense `<node>.matrix` to a temporary cooler
(constant unit balancing weights — the synthetic counts need no ICE), runs
RobusTAD `boundary` then `assembly`, and writes bin-indexed
`*_with_bins.tsv` tables. The intermediate `.cool` is deleted unless
`--keep-cooler` is given. Tune windows with `--minW/--maxW` (kb) and TAD sizes
with `--mintad/--maxtad` (bp).

## Methods (same six as `using_ontad`, so plots line up)

* `inferred_given_root` — unweighted fixed-root ILP (`check_lp_constrained`).
* `inferred_given_root_weighted` — **deliberately biased** boundary-aware
  weighted ILP (`check_lp_constrained_weighted_updated`) with a calibrated
  truth-reveal importance signal in place of real RobusTAD scores, the point
  of this analysis (see above).
* `inferred_intersection` / `inferred_union` — set operations on child calls.
* `inferred_rf` — Fitch–Hartigan.
* `inferred_rf_weighted` — weighted Sankoff, weights = RobusTAD TAD score.

## Outputs

```
using_robustad/
  method_results/<method>/jsons/region_XXX.json   per-region A/B/C P/R/F1
  method_results/<method>/tads/region_XXX/<node>.tad
  method_plots/method_metrics_long.csv
  method_plots/method_metrics_summary.csv
  method_plots/internal_nodes_f1_boxplot.png
```
