# Data-quality sweep

How well are **internal-node TAD hierarchies reconstructed** as the input Hi-C
data degrades? This sub-experiment answers the professor's feedback directly:

1. **Score against the true TADs, not a caller's own output.** The fixed
   experiment (`../using_ontad`, `../using_robustad`) scores reconstructed
   `A/B/C` against the *caller's* `A/B/C` calls, so caller error and
   reconstruction error are tangled. Here the reconstruction is always scored
   against the **hidden ground-truth** hierarchies.
2. **Make data quality the experimental variable.** Instead of one fixed Hi-C
   quality, we sweep a ladder from a caller-free ceiling down to shallow/noisy
   Hi-C and watch F1 fall.
3. **Any tree, aggregated.** Trees of different sizes are just replication axes;
   results are pooled across trees and regions.

## The quality ladder (x-axis) and bin-size family (lines)

```
perfect TADs  ->  full-coverage Hi-C  ->  down-sampled Hi-C
   (ceiling)          (best caller)          (worst caller)
                  --- one line per bin size (resolution) ---
```

* **perfect TADs** — the *true* root/leaf TAD hierarchies are fed straight into
  reconstruction (no caller at all). This is the reconstruction-method ceiling:
  the best F1 achievable if boundary calling were perfect. It is caller- and
  bin-size-free, so it is a single shared anchor at `x = 0` that every resolution
  line starts from.
* **Hi-C points** — a clean base matrix per node is down-sampled by **sequencing
  depth** (binomial thinning, `1.0x → 0.01x`), then called by OnTAD / RobusTAD.
* **bin size (resolution)** — the whole depth ladder is repeated at several bin
  sizes, drawn as separate lines. Bin size is a **counts-density knob**: at a
  fixed region depth, finer bins pool fewer reads, so expected counts per pixel
  scale as `bin_kb^2` (base-matrix amplitude `= A_REF * (bin_kb/REF_KB)^2`). The
  TAD structure (in bins) and the caller settings (in bins) are held fixed, so
  only count density varies and F1 stays comparable across resolutions.

**Why the bin-size family matters.** At coarse bins (40 kb) the matrix is so
count-rich that even `0.01x` down-sampling leaves ample signal, so F1 is flat
across depth — the old figure's puzzle. At finer bins (10 kb, 5 kb) each pixel
starts count-poor, so down-sampling drives the caller (and thus reconstruction)
F1 down. Bin size is the lever that turns depth into a real degradation axis.

The old random-ligation **noise** axis was removed on the professor's note that
Hi-C is itself noisy, so a separate synthetic-noise axis is redundant.

## What is measured

For each `(tree, region, bin size, depth, method)` we reconstruct every **scored
internal node** and compute precision / recall / F1 against that node's true
hierarchy, **averaged over the internal nodes**. All three metrics are written
to CSV; the figures plot **F1 only** vs depth, with a 95% CI band over
`(tree, region)`. The headline figure draws `given_root` (one line per **bin
size**) per caller; the all-methods grid draws every method (caller × method
panels, one line per bin size).

## The six methods

The SCORE stage runs the same six reconstruction methods as the fixed
experiment's `../using_ontad` / `../using_robustad` boxplots, so the figure shows
six F1 lines per caller:

| method | what it does |
|---|---|
| `given_root` | laminar given-root ILP (`check_lp_constrained`) |
| `given_root_weighted` | **deliberately biased** weighted ILP — see below |
| `rf` (FH) | Fitch–Hartigan parsimony |
| `rf_weighted` (Sankoff) | weighted parsimony, leaves weighted by the caller's per-TAD score |
| `union` | union of child TAD sets up the tree |
| `intersection` | intersection of child TAD sets up the tree |

**`given_root_weighted` is a biased baseline, by design.** As in the fixed
experiment, it solves the honest given-root ILP and then re-solves a weighted ILP
with a synthetic importance signal that rewards the honest prediction plus a
calibrated set of the ground-truth TADs it missed, lifting F1 by ~0.21. The point
is to demonstrate that TAD/boundary *weighting* can push `given_root` upward. The
same bias is applied at every quality level (it is **not** re-tuned per
depth or bin-size level).

The weighted ILP is **caller-specific**, matching the fixed experiment exactly:

* **OnTAD** → `check_lp_constrained_weighted.py` (generic per-node importance;
  every scored node can be biased directly).
* **RobusTAD** → `check_lp_constrained_weighted_updated.py` (boundary-aware
  leaf-importance channel; only scored nodes with leaf children are biased
  directly, others improve indirectly via parsimony).

Only `rf_weighted` (Sankoff) uses the caller's *real* scores; that is why the
CALL stage now stores each interval's OnTAD TADscore / RobusTAD score alongside
the interval.

## Trees

You **fix a total node count** (`N_NODES` in the Snakefile) and the generator
makes `N_TREES` distinct **random** topologies of that size (`trees.py`,
`random_tree`). Constraint: **every internal node except the root has ≥ 2
children**; the root is a single-child stem (`root → n0`), matching the original
fixed tree, so `n0` is the top scored node. Different trees therefore have the
same node count but different shapes (varying depth, arity, leaf/scored counts).

Node roles are derived from the topology (`given = root + leaves`,
`scored = internal non-root`), so every generated tree just works. Trees are a
replication axis — results are pooled across them.

A tree's topology is a deterministic function of `(name, N_NODES, TREE_SEED)`, so
the CALL and SCORE stages (separate processes) rebuild the identical tree. Change
the mix by editing `N_NODES` / `N_TREES` / `TREE_SEED` at the top of the
`Snakefile`.

## Two stages (and two interpreters)

Reconstruction uses the Gurobi ILP; RobusTAD needs `cooler`. Those rarely live
in one env, so calling and scoring are split:

* **CALL** — run a caller on the degraded matrices, store the observed
  given-node calls **and their scores**. Runs in the caller's env (`numpy` for
  OnTAD; `cooler` + RobusTAD for RobusTAD). No Gurobi.
* **SCORE** — rebuild the (deterministic) ground truth, reconstruct internal
  nodes with all six methods, score against truth. Runs in a Gurobi env. No
  caller.

Because SCORE rebuilds the truth from the same seed, **CALL and SCORE must get
identical simulation params** — the `Snakefile` passes one shared `COMMON`
string to both.

The `perfect` anchor is score-only (`--caller perfect`, no calls file).

## Run

```bash
# full sweep (edit interpreters / N_REGIONS / TREES at the top of the Snakefile)
snakemake --cores 8

# a single result
snakemake --cores 4 results/medium/ontad/region_000.json
```

Manual, one region, without Snakemake:

```bash
RECON=/path/to/gurobi-env/python        # numpy + gurobipy
ONTAD=/path/to/numpy-env/python
RT=/path/to/refhic-env/python           # cooler + RobusTAD
COMMON="--n-bins 400 --n-top-tads 10 --min-size 8 --min-score-node-tads 6 --tolerance 2"

# perfect anchor (score only)
$RECON run_sweep.py --tree medium --region-index 0 --stage score --caller perfect \
    --outdir results/medium/perfect $COMMON

# OnTAD: call then score
$ONTAD run_sweep.py --tree medium --region-index 0 --stage call --caller ontad \
    --ontad-bin ../../OnTAD/OnTAD/src/OnTAD --outdir calls/medium/ontad $COMMON
$RECON run_sweep.py --tree medium --region-index 0 --stage score --caller ontad \
    --calls-json calls/medium/ontad/region_000.json --outdir results/medium/ontad $COMMON

# RobusTAD: call (refhic env) then score
$RT     run_sweep.py --tree medium --region-index 0 --stage call --caller robustad \
    --outdir calls/medium/robustad $COMMON
$RECON  run_sweep.py --tree medium --region-index 0 --stage score --caller robustad \
    --calls-json calls/medium/robustad/region_000.json --outdir results/medium/robustad $COMMON

# aggregate + plot
$RECON plot_sweep.py --results-dir results --outdir plots
```

## Outputs

```
calls/<tree>/<caller>/region_XXX.json     given-node calls + scores per level (CALL)
results/<tree>/<caller>/region_XXX.json    per-(level, method) avg P/R/F1 vs truth (SCORE)
results/<tree>/perfect/region_XXX.json     the caller-free ceiling, all six methods
plots/sweep_metrics_long.csv               every (tree, region, bin size, depth, method) row, all metrics
plots/sweep_metrics_summary.csv            mean/std/count per (caller, method, bin size, depth)
plots/f1_vs_depth_by_binsize.png           headline: caller x given_root, one line per bin size vs depth
plots/f1_vs_depth_by_binsize_allmethods.png  full grid: caller x method, one line per bin size vs depth
```

## Knobs

* Depth ladder: `DEPTH_LEVELS` in `run_sweep.py`. The old random-ligation noise
  axis was removed (Hi-C is itself noisy).
* Bin-size (resolution) family: `BIN_SIZE_LEVELS` / `REF_KB` / `A_REF` in
  `run_sweep.py` (or `--bin-sizes-kb`, `--a-ref`, `--ref-kb`). Bin size is a
  counts-density scaler: base-matrix amplitude = `A_REF * (bin_kb/REF_KB)^2`, so
  finer bins are proportionally sparser and depth becomes a real degradation
  axis (40 kb reproduces the old count-rich, depth-insensitive regime).
* Methods: the `METHODS` list in `run_sweep.py` (all six run every time). Bias
  strength of `given_root_weighted`: `BIASED_F1_LIFT` / `BIASED_F1_JITTER` there.
* Matrix size / hierarchy: `--n-bins`, `--n-top-tads`, `--nesting-depth`,
  `--min-size`, `--events-per-edge`. The defaults are smaller than the fixed
  experiment because the sweep runs many caller invocations.
