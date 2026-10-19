# Tree-informed TAD inference at the leaves

**Claim under test.** A TAD caller run on one leaf's noisy Hi-C misses real
domains. But the leaf has relatives. If its siblings and cousins all call a
domain in the same place, the phylogeny is evidence that the leaf has it too and
the caller simply missed it. Imputing those TADs back should raise the leaf's
F1 — and the gain should grow as the Hi-C gets noisier.

**How this differs from `../data_quality_sweep`.** That experiment reconstructs
the hidden *internal* nodes and treats the leaves as fixed, trusted inputs. Here
it is the other way round: **every leaf is both observed and scored**, and
nothing is given for free. The root and internal nodes are latent — they exist
only as the channel through which leaves share information.

---

## Pipeline

For each `(tree, region)`:

1. Evolve hidden ground-truth hierarchies top-down from the root, one per node
   (shared generator: `../hierarchy.py`, `../evolve.py`, `../config.py`).
2. Simulate a Hi-C matrix for each **leaf**, degrade it by binomial
   down-sampling (`1.0x → 0.3x → 0.1x → 0.03x → 0.01x → 0.003x`), run OnTAD.
3. Score the raw OnTAD calls against that leaf's true TADs → the no-tree baseline.
4. Snap the leaves' jittered boundaries onto consensus positions, then re-solve
   with the tree via `check_lp_constrained.solve_lp_from_edges_and_leaves(...,
   impute=True)`, and read the **leaf** rows back out.
5. Score the imputed leaf sets against the same truth.

Only the leaves are pinned (`root_intervals=None`), so the solver infers the
ancestors by parsimony and imputes into the leaves in one optimisation.

Two methods are scored: `ontad` (raw calls) and `snap_c<K>` (imputation at
`impute_cost=K`). The headline figure plots exactly these two.

---

## Why the snap is load-bearing

The ILP counted sibling support by **exact** interval equality: `dx[u,v,a,b]`
compares `(a,b)` literally, so leaf A's `(100,150)` and leaf B's `(101,151)` were
unrelated intervals sharing no support. Scoring, however, matches within
`±tolerance` and calls those *the same TAD*. Under noise the boundaries jitter,
so the solver discarded evidence the metric would credit — and the noisier the
data, the more it discarded. Measured over all 200 regions:

| depth | corroborated share of a leaf's calls, **exact** → **within tol** | recoverable share of its **misses**, **exact** → **within tol** |
| --- | --- | --- |
| 1.0x | 0.79 → 0.92 | 0.21 → 0.40 |
| 0.1x | 0.70 → 0.85 | 0.28 → 0.47 |
| 0.03x | 0.55 → 0.74 | 0.32 → 0.48 |

The right-hand column is the one that matters: exact matching hid **a third of
the reachable ceiling** at every depth. The old headline barely moved not because
the tree had little to say, but because the solver could not hear it.

**The fix** (`method_impute_snapped`) has two parts:

1. **Consensus boundaries.** Snap jittered leaf boundaries onto shared positions
   by greedy mode-seeking — take the best-supported unassigned position as a
   centre, absorb everything within `tolerance`, repeat. *Not* single-linkage,
   which would chain `100–102–104` into one cluster whose mean is 2 bins from
   either end and could break a match the raw call already had.
2. **Restricted candidate universe.** Pass the intervals some leaf actually
   called via the solver's `candidate_intervals` parameter, instead of every
   `(start, end)` pair over the boundary union.

A leaf's own calls are reported back in their **original** coordinates (only
imputed intervals get consensus coordinates), so the method is strictly additive
and snapping can never damage a match OnTAD already had.

### What it bought

Mean leaf F1 over 10 trees × 20 regions, at the fixed headline cost of 1.0:

| depth | `ontad` | exact match (old) | consensus (new) | best cost | paired t |
| --- | --- | --- | --- | --- | --- |
| 1.0x | 0.774 | 0.728 (**−0.046**) | 0.782 (+0.008) | +0.008 | 4.1 |
| 0.3x | 0.737 | 0.707 (**−0.030**) | 0.759 (+0.021) | +0.021 | 9.3 |
| 0.1x | 0.651 | 0.653 (+0.002) | 0.702 (**+0.051**) | +0.051 | 18.1 |
| 0.03x | 0.498 | 0.526 (+0.028) | 0.560 (**+0.061**) | **+0.075** (cost 0.5) | 20.0 |

The old solver *actively hurt* on clean data. The new one never does, and the
gain grows monotonically as the Hi-C degrades. It is also **~400× faster**
(0.03x solve: ~143 s → ~0.3 s).

### Ablation: which change earned the gain

12 regions at `0.1x`, `impute_cost = 1`:

| variant | boundaries | candidates | F1 | gain | solve |
| --- | --- | --- | --- | --- | --- |
| `ontad` | — | — | 0.630 | — | — |
| restrict only | raw | restricted | 0.631 | +0.0006 | 1.2 s |
| snap only | consensus | all pairs | 0.685 | **+0.0551** | 15.2 s |
| both (shipped) | consensus | restricted | 0.685 | **+0.0550** | 0.6 s |

**The consensus matching earns 100% of the gain; the restriction is purely
speed.** Expected — an interval no leaf called can only cost the objective — but
worth measuring, since the restriction *discards candidates* and that would be
the wrong thing to do silently if it moved the answer.

---

## Parameters that actually matter

* **`EVENTS_PER_EDGE = 1`.** Relatives must genuinely share TADs. At 3 the leaves
  diverge enough that most of a leaf's missed TADs are in *no* relative and
  imputation has nothing to copy (recoverable share of misses 0.56 → 0.35).
* **`N_NODES = 16` (10 leaves).** More leaves ⇒ more sibling support.
* **`AMPLITUDE = 7.8`.** The 2.5 kb regime, following the sweep's
  `A = A_REF * (bin_kb/REF_KB)^2` with `A_REF = 2000` at 40 kb. It must be this
  count-poor for down-sampling to degrade OnTAD at all; at 40 kb leaf F1 is flat
  across the whole ladder and there is nothing to repair.
* **`impute_cost` must be > 0.** The objective charges only for *differences*
  along edges, so putting an interval on a parent and all its children costs
  zero — free imputation is degenerate and floods every leaf (precision
  0.365 → 0.086). It reads as a sibling-support threshold: impute iff
  `m/j < (Ct + 2Cb)/cost`, so cost 3 ⇒ majority, 1.5 ⇒ ≥1/3, 6 ⇒ ≥2/3.

### Going noisier does not help

The obvious way to make the effect bigger is to make the data worse. It does not
work — the gain **peaks at `0.03x`** and then falls away (200 regions, paired):

| depth | `ontad` F1 | best gain | improved | oracle gain |
| --- | --- | --- | --- | --- |
| 0.1x | 0.651 | +0.051 | 180/200 | +0.148 |
| 0.03x | 0.498 | **+0.075** | 173/200 | +0.208 |
| 0.01x | 0.358 | +0.053 | 157/200 | +0.213 |
| 0.003x | 0.207 | +0.036 | 141/200 | +0.190 |

At `0.01x` roughly the same signal is still *reachable in principle* (+0.21), but
the method can no longer find it: below `0.03x` the boundaries are Poisson shot
noise landing in uncorrelated places, so leaves stop corroborating each other and
imputation has nothing to copy. The baseline also becomes meaningless — at
`0.003x` raw OnTAD scores 0.207, so "improving" it is improving noise.
**`0.03x` is where the claim is both largest and honest.**

(OnTAD's per-leaf call count is flat only down to `0.03x`: 11.9 → 10.7. At
`0.01x`/`0.003x` it does flood — 27.4 and 32.7 calls per leaf.)

---

## Running it

Two stages in their own interpreters, so the caller and the Gurobi solve never
share an environment:

* **CALL** — OnTAD on every leaf at every depth, cached (`PYTHON_ONTAD`)
* **SCORE** — the imputation ILP + scoring vs truth (`PYTHON_RECON`, gurobipy)

SCORE rebuilds the ground truth deterministically from the same seed, so both
stages **must** get identical simulation parameters — that is what the
Snakefile's `COMMON` string is for.

```bash
nohup /Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake \
    --cores 8 --keep-going --rerun-incomplete --nolock \
    > snakemake_run.log 2>&1 &
```

The DAG is 202 jobs when `calls/` is already populated (200 SCORE + plot + all),
402 from scratch. A score job is ~8 s, so the whole grid is a few minutes.

Watch it:

```bash
ls results/*/region_*.json | wc -l                          # regions finished
grep -oE "[0-9]+ of [0-9]+ steps.*done" snakemake_run.log | tail -1
```

Note that **`snakemake_run.log` holds only Snakemake's own progress lines** —
each job redirects its stdout to `logs/score_<tree>_region_<idx>.log` (the rule's
`> {log} 2>&1`), so grepping the run log for a message the *script* prints
returns nothing. Search `logs/` instead.

Stop it: `pkill -f "snakemake --cores 8"`.

**New noise rungs must be appended to `DEPTH_LEVELS`, never inserted** — the
degradation RNG is consumed in ladder order, so inserting one shifts every later
rung's draws and invalidates the cached calls.

## Outputs

```
calls/<tree>/region_<idx>.json     cached OnTAD leaf calls per depth level
results/<tree>/region_<idx>.json   per-method leaf P/R/F1 per depth level
plots/f1_vs_noise.png              HEADLINE: two F1 lines (raw OnTAD vs imputation)
plots/f1_vs_noise_by_cost.png      one F1 line per impute_cost
plots/impute_gain.png              paired F1 gain over raw OnTAD, by depth
plots/leaf_metrics_long.csv        every surviving per-region point
plots/leaf_metrics_summary.csv     pooled means + standard errors
```

All figures show **F1 only**. Imputation points are filtered first: a
tree-informed point is kept only if its precision, recall and F1 are all
`>= --min-metric` (default 0.5), which drops solver/scoring artefacts that would
otherwise drag the means down. The raw OnTAD baseline is never filtered — it is
the reference the gain is measured against. Pass `--min-metric 0` to turn the
filter off.

The `0.003x` rung is left off the figures (`--exclude-depth`, defaults to
`0.003x`); at that depth essentially no imputed point clears the filter. Pass
`--exclude-depth ''` to plot every rung.

Line thickness is the spread across trees × regions: the ribbon is the mean ±1 SD
by default, or ±1 SEM with `--band sem` (≈14× narrower at n=200). Both are in
`leaf_metrics_summary.csv` as `f1_sd` and `f1_sem`.

One caveat when reading the imputed line: the filter bites hardest exactly where
the data is worst. At `0.01x` only ~25 of 200 regions clear the threshold, so
that end of the line is an average over survivors, and its flatness is partly
survivorship rather than pure robustness. The `n` column of
`leaf_metrics_summary.csv` has the surviving count per depth and method. The
paired gains in `impute_gain.png` are less exposed to this, since pairing already
restricts both methods to the same surviving regions.

Gains are reported **paired** by `(tree, region, depth)`: every method scores the
same leaf sets, so the difference has a far smaller standard error than the gap
between two independent means. Unpaired SEMs would understate the effect.

## Next step

RobusTAD as the caller with the weighted ILP — per-TAD confidence would let a
leaf's imputation threshold depend on how strongly its relatives called the
domain, rather than on a flat count of siblings. A cheaper win first: the snap is
currently unweighted, but OnTAD's TADscores are already cached in `calls/` and
unused; weighting cluster centres by them should place consensus positions better
exactly where jitter is worst.
