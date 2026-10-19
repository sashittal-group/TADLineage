# Tree-informed TAD inference at the leaves — RobusTAD + the weighted ILP

**Claim under test.** Identical to [`../tree_informed_tad_inference_ontad`](../tree_informed_tad_inference_ontad):
a TAD caller run on one leaf's noisy Hi-C misses real domains, but the leaf has
relatives; if they call a domain in the same place, the phylogeny is evidence the
leaf has it too. Imputing those TADs back should raise the leaf's F1, and the
gain should grow as the Hi-C degrades.

**What is new here.** That arm ends with "next step: RobusTAD as the caller with
the weighted ILP — per-TAD confidence would let a leaf's imputation threshold
depend on how strongly its relatives called the domain, rather than on a flat
count of siblings." This is that experiment.

The simulation, the seeds and the noise ladder are shared with the OnTAD arm, so
a leaf's hidden truth is bit-identical between the two and only the caller and
the solver differ.

---

## Why RobusTAD changes what is possible

OnTAD reports one number per called TAD. The tree can therefore only *count*
relatives: a flat `impute_cost` that reads as "impute iff at least this fraction
of siblings called it". RobusTAD reports two things OnTAD does not:

1. a **per-TAD score** for every domain it assembles, and
2. a **per-bin left/right boundary score for the whole region** — including the
   bins where it assembled no TAD at all.

(2) is the interesting one. It is a graded read-out of a leaf's own evidence at a
position it never called, which is exactly the situation imputation is in. Three
mechanisms use these, all through
`check_lp_constrained_weighted_updated.solve_lp_from_edges_and_leaves`:

| mechanism | what it does | what it needs from RobusTAD |
| --- | --- | --- |
| per-`(leaf, interval)` `impute_cost` | the price a leaf pays to **gain** `(a,b)` moves with *its own* boundary evidence at `a` and `b` | `leftScore` / `rightScore` at **uncalled** bins |
| `leaf_importance = {x, s, e}` | a better-than-median leaf call pulls the domain onto its **parent**, from where relatives can reach it; a worse-than-median one pushes it away | TAD score, `leftScore[a]`, `rightScore[b]` |
| score-weighted snapping | consensus boundary positions are chosen by sibling count **plus** boundary score, not count alone | `leftScore` / `rightScore` |

The first is the one with no OnTAD analogue, and measurably the one that earns
the gain (see the ablation below). Under a flat cost every leaf demands the same
corroboration; here a leaf that half-sees a domain is talked into it by fewer
relatives than a leaf that sees nothing there.

Two design points in that price are load-bearing:

* **Percentile, not the raw score.** RobusTAD's boundary field is dense and sits
  high — mean support 0.95 at intervals that are true for a leaf against 0.76 at
  intervals that are not. The *ordering* is informative (AUC ≈ 0.78) but the
  *levels* are nearly equal, so a raw-score discount prices true and false
  candidates within 20% of each other. Ranking within the leaf's own candidate
  pool restores the spread.
* **Two-sided around the flat price.** Rank 0.5 costs exactly `impute_cost`, so
  the *mean* price is the flat price and the weighted method spends the **same
  imputation budget** as the unweighted control — it just spends it elsewhere. A
  one-sided discount would simply impute more, and any F1 difference would be
  confounded with the extra volume.

`impute_cost` **must stay > 0**, per the OnTAD arm: the objective charges only
for *differences* along edges, so putting an interval on a parent and all its
children costs zero and free imputation floods every leaf. `--price-floor` keeps
the two-sided price bounded away from zero for that reason.

### Calibrating `leaf_importance` (why raw scores fail)

The solver's importance term is `|var - 3c|` for a score `c`, so selecting the
variable is *penalised* below `c = 1/6` and *rewarded* above it: **1/6 is the
indifference point**. RobusTAD's scores are excellent at separating a leaf's true
from false calls (AUC 0.92–0.95, mean 0.84 vs 0.33) — but *both* means sit far
above 1/6, so feeding them in raw rewards the parent for carrying true and false
domains alike, differing only in degree. Measured on the pilot, raw importance
was neutral at `w = 0.1` and clearly harmful at `w = 0.3` (−0.012 F1 at `1.0x`):
it inflates the parent instead of discriminating on it.

So each score is passed as its **percentile rank within that leaf's own calls**,
scaled so the median call lands exactly on the indifference point. The parent is
then pulled toward a leaf's better-than-median calls and pushed away from its
worse-than-median ones. After calibration `w` is flat over 0.1–0.3 and no longer
does damage. `--raw-importance` reproduces the uncalibrated behaviour.

The `* 3` multiplier inside the shared solver was deliberately **not** touched —
all of this calibration happens on the scores before they are handed over.

---

## Tolerance: the solver matches exactly, the metric matches within ±2

Scoring calls `[1,10]` and `[1,11]` **the same TAD** (`tolerance = 2`), but the
ILP compares intervals by literal equality — `dx[u,v,a,b]` sees two unrelated
intervals. That mismatch is what consensus snapping exists to fix, and it is
load-bearing: in the OnTAD arm it turned a +0.028 gain into +0.075 and stopped
imputation from actively hurting on clean data.

It also creates a hazard worth checking rather than assuming. Matching is greedy
and **one-to-one** (`match_tads_within_tolerance` marks each truth TAD used), so
if imputation gave a leaf `[1,11]` when it already called `[1,10]`, the second
copy would claim nothing and land as a pure false positive. Snapping should
prevent it — both ends go to one consensus — but mode-seeking does not *prove*
it: if 12 is absorbed by centre 13 first, 10 can survive as its own centre, and
10 vs 12 are still "the same" to the metric.

Measured over 15 regions × 6 depths × both methods:

| check | result |
| --- | --- |
| imputed intervals within tolerance of one the leaf already had | **0 / 2337** |
| within-tolerance pairs inside RobusTAD's own raw calls | **0** at every depth |
| within-tolerance pairs inside the **truth** | 25 / 2189 (**1.14%**) |

So the hazard does not materialise: RobusTAD's assembly emits no near-duplicate
calls, and since the candidate universe is the union of (snapped) leaf calls,
there are none to propagate.

**No dedup guard was added, and the third row is why.** A rule like "never add an
interval within tolerance of one the leaf already has" would look like free
insurance, but where the truth holds two colliding domains a near-duplicate
prediction legitimately claims the second one. That rule would trade real recall
on ~1% of truth for a failure that occurs 0 times in 2337.

Tolerance is honoured in one more place: `BoundaryScores` lookups take the best
score within ±tolerance, so a leaf's own boundary evidence still counts when its
boundary sits a bin off the consensus position.

---

## Methods scored at every noise level (all evaluated on the LEAVES)

| method | what it is |
| --- | --- |
| `robustad` | raw RobusTAD calls — the no-tree baseline |
| `snap_c<K>` | flat-cost imputation, **no scores used**: the OnTAD arm's method run on RobusTAD calls |
| `wsnap_c<K>` | all three weighted mechanisms above |
| `wimp_c<K>` | ablation: `leaf_importance` only |
| `wpri_c<K>` | ablation: the discounted price only |

`snap` is the control that matters: the distance from `robustad` to `snap` is
what the *phylogeny* buys, and the distance from `snap` to `wsnap` is what the
*scores* buy.

All four are **scored** and land in the CSVs, but by default only `wsnap` is
**drawn**, so the figures carry one comparison: the caller alone against the
caller repaired by the weighted tree. `PLOT_METHODS = "all"` in the Snakefile
(or `--plot-methods snap,wsnap`) puts the control and the ablations back on the
figures, and re-enables `weighting_gain.png` — which needs the control to
difference against, and so must also be added back to `rule all` and the plot
rule's outputs. The weighting gain is printed to stdout either way.

---

## Results (10 trees × 20 regions = 200, `impute_cost = 0.5`, nothing filtered)

Mean leaf F1, and the gain over raw RobusTAD **paired** by `(tree, region, depth)`:

| depth | raw RobusTAD | weighted imputation | paired gain ± SEM | |
| --- | --- | --- | --- | --- |
| 1.0x | 0.540 | 0.535 | −0.005 ± 0.002 | |
| 0.3x | 0.550 | 0.545 | −0.005 ± 0.002 | |
| 0.1x | 0.533 | 0.540 | +0.007 ± 0.002 | |
| 0.03x | 0.420 | 0.479 | **+0.059 ± 0.005** | 11σ |
| 0.01x | 0.365 | 0.372 | +0.008 ± 0.002 | |
| 0.003x | 0.275 | 0.352 | **+0.077 ± 0.004** | 21σ |

**The claim holds where it is supposed to and nowhere else.** RobusTAD is flat
from `1.0x` to `0.1x` (0.54 → 0.55 → 0.53) — nothing is broken, so imputation has
nothing to repair and costs a statistically real but practically irrelevant
−0.005. From `0.03x` down the caller's *recall* collapses (0.53 → 0.30 → 0.25 →
0.18) while its precision climbs to 0.68–0.78: it stops calling rather than
calling wrongly. That is exactly the failure imputation addresses, and it does:
+0.059 and +0.077 F1, both overwhelmingly significant.

If you need a setting that **never** costs anything on clean data,
`impute_cost = 3` gives −0.000 at `1.0x` and still +0.044 at `0.003x`.
`f1_vs_noise_by_cost.png` is that trade-off: noisier data rewards a cheaper,
more trusting imputation.

### What each mechanism is worth (same 200 regions, paired gain over raw)

| method | 1.0x | 0.1x | 0.03x | 0.003x |
| --- | --- | --- | --- | --- |
| `snap` — no scores at all (the OnTAD arm's method) | −0.0094 | −0.0001 | +0.0496 | +0.0740 |
| `wimp` — + calibrated `leaf_importance` | −0.0084 | +0.0008 | +0.0506 | +0.0745 |
| `wpri` — + boundary-discounted price | −0.0053 | +0.0063 | **+0.0599** | +0.0764 |
| `wsnap` — everything | −0.0052 | +0.0070 | +0.0592 | **+0.0769** |

**The per-leaf boundary-discounted price earns essentially all of the weighting
gain**: `wpri` alone reproduces `wsnap` to within noise, while `leaf_importance`
adds ~+0.001 and the score-weighted snap adds nothing measurable. The honest
summary is that **the boundary scores at bins RobusTAD did *not* call are the
useful signal**, not the confidence attached to the calls it did make.

The weighting is nonetheless worth having, and unlike the tree itself it helps at
**every** depth — paired `wsnap − snap`:

| 1.0x | 0.3x | 0.1x | 0.03x | 0.01x | 0.003x |
| --- | --- | --- | --- | --- | --- |
| +0.0042 ±0.0008 | +0.0054 ±0.0011 | +0.0071 ±0.0014 | +0.0096 ±0.0021 | +0.0033 ±0.0016 | +0.0029 ±0.0015 |

It also halves the clean-data damage (−0.009 → −0.005), because pricing by the
receiving leaf's own evidence suppresses exactly the imputations that were
adding false positives where the caller was already right.

### Do not turn `--min-metric` back on

The filter drops tree-informed points below a P/R/F1 threshold but **never
filters the baseline**, so it compares the imputed regions that scored well
against *all* baseline regions — two different samples. At the default 0.5 it
reports the `1.0x` gain as roughly **+0.08** when the paired truth is **−0.005**,
and leaves `n = 1` at some noisy rungs. It defaults to 0 here for that reason.

---

## Is the weighting even possible? (`diagnose_scores.py`)

Two assumptions have to hold before any F1 curve means anything, and both are
measurable straight from the cached calls:

* **A.** RobusTAD's TAD score separates a leaf's true positives from its false
  positives. If not, `leaf_importance` pulls right and wrong domains onto the
  parent equally.
* **B.** A leaf keeps boundary evidence at the domains it **missed** — its own
  boundary support is higher at intervals that are true for it than at intervals
  that are not. If not, the per-leaf price discount is charging at random.

```bash
python diagnose_scores.py --calls-dir calls
```

reports both as an AUC per depth rung (0.5 = no information). Run it before
reading the figures; it is the fastest way to tell a tuning problem from a
"there is nothing here" result.

Measured on the first 10 pilot regions, **both hold comfortably**:

| depth | A: TAD score, TP vs FP | mean TP / FP | B: own support, missed-true vs false | mean true / false | calls/leaf |
| --- | --- | --- | --- | --- | --- |
| 1.0x | **0.952** | 0.84 / 0.33 | **0.762** | 0.95 / 0.76 | 13.2 |
| 0.1x | **0.918** | 0.72 / 0.24 | **0.816** | 0.95 / 0.73 | 11.4 |
| 0.03x | **0.937** | 0.73 / 0.20 | **0.714** | 0.82 / 0.67 | 5.7 |
| 0.01x | **0.952** | 0.66 / 0.21 | **0.765** | 0.88 / 0.66 | 4.1 |
| 0.003x | **0.868** | 0.45 / 0.20 | **0.712** | 0.76 / 0.62 | 3.4 |

Two things to read off this. The score signal (A) is nearly perfect and barely
degrades — RobusTAD *knows* which of its calls are good even on ruined data. And
(B) survives at every rung: a leaf really does keep graded boundary evidence at
the domains it failed to assemble, which is what makes the per-leaf price
meaningful. Note also that RobusTAD's calls per leaf **fall** with depth
(13.2 → 3.4) rather than exploding — it turns conservative rather than noisy, so
what the noisy end loses is recall, which is exactly the failure mode imputation
addresses.

---

## RobusTAD settings

Tuned on tree00/region_000 across the depth ladder (4–5 leaves per cell):

| setting | F1 @1.0x | F1 @0.1x | note |
| --- | --- | --- | --- |
| default (`minW=80`) | 0.477 | 0.395 | 80 kb = 2 bins at this resolution |
| **`minW=160`** | **0.487** | **0.513** | shipped |
| `minW=320` | 0.465 | 0.495 | |
| `minW=40` | 0.413 | 0.172 | chases single-bin noise; precision collapses |
| `delta=0.2` | 0.531 | 0.422 | higher precision, worse at depth |
| `ratio=1.05` | 0.395 | 0.223 | |
| `alpha`, `maxW` | no effect | no effect | measured at 0.05/0.1/0.2 and 800/1200/2000 |

`minW` is the only setting that moves the result at both ends of the ladder: at
40 kb bins the 80 kb default is a 2-bin window, small enough to chase shot noise
once the matrix is thinned. `alpha` and `maxW` genuinely do nothing here — worth
recording so nobody re-tunes them.

The baseline is weaker than the OnTAD arm's (≈0.43 vs ≈0.77 at `1.0x`), which is
a property of the caller on 500-bin, count-poor synthetic matrices, not of the
tree. The comparison that matters is each method against **its own** baseline.

### Where the effect lives, and why

RobusTAD is far more depth-robust than OnTAD on these matrices. Its raw F1 is
**flat from `1.0x` to `0.1x`** (0.43 → 0.44 → 0.43) and only then falls
(0.35 → 0.29 → 0.22), where OnTAD had already dropped from 0.77 to 0.65 by
`0.1x`. So the whole imputation story sits at `0.03x` and below — above that
there is simply nothing broken for the tree to repair, and the measured gain is
correspondingly ~0. That is a difference in the *caller*, not in the claim.

### A note on bin size

`AMPLITUDE = 7.8` **is** the 2.5 kb regime: the sweep's convention models
resolution as a count-density scaler, `A = A_REF * (bin_kb/REF_KB)^2` with
`A_REF = 2000` at 40 kb. The `resolution_bp = 40000` that reaches RobusTAD is
only the unit its `--minW` / `--mintad` arguments are expressed in, and those are
configured here in **bins** (`RT_MINW` is quoted in kb only because RobusTAD's
CLI demands kb). Re-running at `resolution_bp = 2500` with the window arguments
rescaled by the same factor is the identical experiment on the identical
matrices — so the count-poor fine-resolution regime is already what is being
measured, and there is no second bin size to sweep.

---

## Running it

Two stages in their own interpreters, so the caller and the Gurobi solve never
share an environment:

* **CALL** — RobusTAD on every leaf at every depth, cached (`PYTHON_ROBUSTAD`, the
  `refhic` env: needs `cooler` + RobusTAD)
* **SCORE** — the imputation ILP + scoring vs truth (`PYTHON_RECON`, gurobipy)

SCORE rebuilds the ground truth deterministically from the same seed, so both
stages **must** get identical simulation parameters — that is what the
Snakefile's `COMMON` string is for.

```bash
nohup /Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake \
    --cores 5 --keep-going --rerun-incomplete --nolock \
    > snakemake_run.log 2>&1 &
```

**CALL is the expensive half.** RobusTAD is ~8 s per leaf per depth: ~8 minutes
per region for 10 leaves × 6 rungs, and each run wants ~1.7 cores, so `--cores 5`
already saturates a 10-core machine. A SCORE job is a couple of seconds. Start
with `N_TREES = 2` or `3`, confirm the effect, then widen — the calls are cached,
so re-tuning the solver only re-runs SCORE:

```bash
snakemake --cores 8 --forcerun score plot     # re-score cached calls
```

Watch it:

```bash
ls calls/*/region_*.json | wc -l                            # regions called
grep -oE "[0-9]+ of [0-9]+ steps.*done" snakemake_run.log | tail -1
```

Each job redirects its own stdout to `logs/`, so `snakemake_run.log` holds only
Snakemake's progress lines — grep `logs/` for anything a *script* printed.

**New noise rungs must be appended to `DEPTH_LEVELS`, never inserted** — the
degradation RNG is consumed in ladder order, so inserting one shifts every later
rung's draws and invalidates the cached calls.

---

## Outputs

```
calls/<tree>/region_<idx>.json     cached RobusTAD leaf calls + per-bin boundary
                                   scores, per depth level
results/<tree>/region_<idx>.json   per-method leaf P/R/F1 per depth level
plots/f1_vs_noise.png              HEADLINE: raw RobusTAD vs weighted imputation
plots/f1_vs_noise_by_cost.png      one F1 line per impute_cost
plots/impute_gain.png              paired F1 gain over raw RobusTAD
plots/leaf_metrics_long.csv        every per-region point, ALL scored methods
plots/leaf_metrics_summary.csv     pooled means + SD/SEM, ALL scored methods
```

Gains are reported **paired** by `(tree, region, depth)`: every method scores the
same leaf sets, so the difference has a far smaller standard error than the gap
between two independent means. Unpaired SEMs would understate the effect.

Disk: the cache runs to ~1.5 MB per region (~300 MB for the full 10 × 20 grid),
and `work/` keeps each leaf's raw RobusTAD output — another ~0.8 GB, safe to
delete once `calls/` is written.

Unlike the OnTAD arm, `--min-metric` defaults to **0** (no filtering). That
filter drops tree-informed points whose P/R/F1 fall below a threshold, which is
survivorship bias the OnTAD arm can just about afford at a 0.77 baseline; at
RobusTAD's ≈0.5 it would discard most of the grid.

## Cached boundary scores

RobusTAD assigns a non-zero boundary score to ~95% of bins, nearly all of it
floor noise, so the cache keeps only scores `>= --boundary-floor` (0.02),
rounded to 4 dp. Positions are stored **shifted to the call convention**
(`bin + 1`), so an interval `(a, b)` reads its own evidence as `left[a]` and
`right[b]` with no arithmetic anywhere downstream — see `BoundaryScores` in
`run_tree_impute_rt.py`. Lookups take the best score within `±tolerance`, since
snapping moves a boundary by up to that much and the scoring metric treats that
distance as the same TAD anyway.

## What was changed outside this directory

`check_lp_constrained_weighted_updated.py` (shared with `hemato_cells/` and
`simulation/using_robustad/`) gained `impute_cost` and `candidate_intervals`,
and its laminarity cuts now inspect leaves **when `impute=True`**, since under
imputation a leaf can gain intervals and stop being laminar. Every addition is
inert at the defaults (`impute=False`, `impute_cost=0`,
`candidate_intervals=None`): the non-imputation model those pipelines solve is
unchanged, and the importance term — including its `* 3` score multiplier — was
deliberately left alone.
