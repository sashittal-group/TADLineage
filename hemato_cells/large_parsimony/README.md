# Large parsimony on the hematopoietic quartet

Does the large parsimony ILP (TAD-LP, Supp. Sec. "The large parsimony integer
linear program") recover the hematopoietic tree when it is **not** given one?

The small parsimony pipelines are given

    LT -> MPP -> {CLP, CMP -> {leaf1, leaf2}}        leaf1 in {MEP, MKP, MK}, leaf2 in {GMP, G}

with LT pinned at the root. With the root removed, MPP and CMP both have degree 3,
so this is the binary quartet **{LT, CLP} | {leaf1, leaf2}**. The large
parsimony objective does not depend on where the root sits. So we give the ILP
the four observed hierarchies LT, CLP, leaf1 and leaf2 as leaves, with no tree,
and count how often it returns that quartet. The harmonization tolerance is
treated as a free parameter: tolerances 0–5 are tried and each case counts once,
at its best tolerance.

## Analyses

Three analyses, one folder each. Each folder holds its own `Snakefile` and
`results/`. They mirror the three small parsimony pipelines:

| folder | leaf TAD calls | objective | small parsimony counterpart |
|---|---|---|---|
| `ontad_unweighted/` | `hemato_cells/<NODE>/` (OnTAD) | parsimony | `../check_intermediate_snakemake_given_root` |
| `ontad_weighted/` | `hemato_cells/using_ontad/<NODE>/` (OnTAD) | parsimony + internal-vertex importance | `../using_ontad/check_intermediate_snakemake_given_root_weighted` |
| `robustad_weighted/` | `hemato_cells/using_robustad/<NODE>/` (RobusTAD) | parsimony + leaf importance | `../using_robustad/check_intermediate_snakemake_given_root_weighted` |

**OnTAD weighted.** This uses the importance term of
`check_intermediate_given_root_weighted.py`. RobusTAD TAD scores of MPP and CMP
(`robustad_predictions/`, `tads_all`, λ = 1) are added to the objective as
−Σ_t [x·imp + (1 − x)(1 − imp)] over the internal vertex's candidate TADs. The
root sits on LT's edge, so in every quartet vertex 6 is the vertex next to LT and
gets MPP's scores, and vertex 5 is the cherry parent and gets CMP's. These scores
come from the Hi-C of MPP and CMP themselves, the nodes the tree is inferred
around. So this analysis uses information about the internal nodes that the
unweighted one does not have.

**RobusTAD weighted.** This uses the importance term of
`check_intermediate_given_root_weighted_updated.py`. The RobusTAD TAD and boundary
scores of CLP, leaf1 and leaf2 (weight 0.1, c = 3·score) reward a leaf's parent
for keeping the leaf's TADs and boundaries: w·[c + q(1 − 2c)]. In the large ILP
the parent is chosen by the solver, so the term uses the arc variables. LT gets no
scores, as in that pipeline, where LT is the root rather than a leaf.

Both weightings import the pipeline's own score builder, so the scores are the same.

**Validated against the weighted small parsimony solvers.** On chr19 at 2 Mb and
1 Mb and chr7 at 1 Mb, each of the three quartets was solved as the small
parsimony tree rooted at LT:

- **RobusTAD weighted**: all nine costs matched to machine precision.
- **OnTAD weighted**: every cost differed by the same constant within a case.
  `check_lp_constrained_weighted.py` counts the fixed root LT as an internal node,
  so it adds −Σ_t [0.1 if LT has t, else 0.9]. That term does not depend on the
  tree, so rankings and margins are unaffected.

## Files

| file | what it does |
|---|---|
| `large_parsimony_ilp.py` | The ILP, written from the supplement: arc variables `e[u,v]`, one parent per non-root vertex, two children per internal vertex, difference constraints released by `(1 - e[u,v])`, all start×end candidate TADs, and laminarity cuts added only when a solution violates them. Generic in k. Optional `vertex_importance` / `leaf_importance` terms (convex only). |
| `infer_quartet_tree.py` | One instance: harmonize the four calls, solve, score all three quartets, and write the inferred MPP and CMP when the tree is correct. `--data-dir` and `--weighting {none, ontad_weighted, robustad_weighted}` select the analysis. |
| `aggregate_best_tolerance.py` | Reads the JSONs and writes the counts, per chromosome and genome-wide, each case at its best tolerance. No new solves. |
| `quartet_rules.smk` | The rules, shared by the three Snakefiles. Each Snakefile sets `DATA_DIR` and `WEIGHTING`. Grid: 19 chromosomes × 5 bin sizes × 6 leaf pairs × tolerances {0, …, 5} = 3420 instances. |

## Run

From the analysis folder, e.g.:

```bash
cd hemato_cells/large_parsimony/robustad_weighted
nohup caffeinate -is /Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake \
    --cores 8 -p --rerun-incomplete --keep-going > snakemake_run.log 2>&1 &
```

A single instance, from the analysis folder:

```bash
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python ../infer_quartet_tree.py \
    --data-dir ../../using_robustad --weighting robustad_weighted \
    --chrom 19 --resolution 1mb --leaf1 MK --leaf2 G --tolerance 1 \
    --output results/jsons/quartet_chr19_1mb_MK_G_1.json
```

## Reading the result

`<analysis>/results/best_tolerance/best_tolerance_summary.txt`. A **case** is a chromosome ×
bin size × leaf pair (570 cases), or, genome-wide, a bin size × leaf pair (30).

- **`correct_best_tol`**: at least one of the tolerances 0–5 makes the correct
  quartet the *unique* optimum. Ties count as not correct. The best tolerance is
  the one where the correct quartet wins by the largest relative margin,
  (best wrong cost − correct cost) / correct cost. The table also shows how
  often each tolerance is the best one.
- **`chance_best_tol`**: the tolerance is picked knowing the answer, so chance is
  no longer 1/3. With six tries, a random winner hits the correct quartet more
  often. This column gives the exact chance level for each case: the number of
  different quartets that are the unique optimum at some tolerance, divided by 3.
  That is the hit rate if the correct quartet were one of the three picked at
  random. It keeps the correlation between tolerances, so it is lower than
  1 − (2/3)^6. Compare the result to this column, not to 1/3.

The relative margin divides by the *parsimony* part of the correct quartet's cost.
With the OnTAD importance term, the total cost is often negative.

**Genome-wide.** For a fixed tree the chromosomes do not interact. No TAD spans
two chromosomes, laminarity only compares TADs on one chromosome, and the cost is
a sum over TADs and boundaries. So a quartet's genome-wide cost is the **sum of
its per-chromosome costs**. Every JSON already holds the cost of all three
quartets, so this is exact and needs no new solves. It was checked against a
joint ILP on concatenated chromosomes, with cross-chromosome candidate TADs
removed, on chr17–19 at 2 Mb and chr16–19 at 1 Mb. All six quartet costs matched.

## Design notes

- **The root sits on the edge above LT** (`root_leaf`, which `--free-root` turns
  off). Every unrooted tree has an edge to LT, and the objective ignores the root,
  so this excludes no topology. It cuts the 18 rooted trees down to 3, one per
  quartet. On chr1 at 250 kb this reduced the runtime from 442 s to 42 s, with the
  same optimum.
- **Laminarity cuts are added during the solve.** Crossing TAD pairs are cut
  with Gurobi lazy constraints inside a single branch and bound. The earlier
  approach re-solved from scratch after each round, and at chr1 100 kb each
  re-solve took about 3 minutes while adding only 1–5 pairs. The cuts become
  permanent model constraints afterwards, so the other quartet solves start
  with them. The lexicographic modes keep the re-solve loop, because Gurobi
  callbacks do not work with multi-objective models.
- **Pinning a quartet**: vertex 5 can only have leaves as children, so its two
  children are the cherry. `solve(pin_partner=c)` fixes the cherry. This lets all
  three quartets be scored with one model, and the laminarity cuts carry over
  between solves.
- **Validation**: on 16 instances (chr 7, 11, 19 at 500 kb and 1 Mb, two leaf
  pairs, tolerances 0 and 2), each quartet's cost equals the cost the independent
  small parsimony code (`check_lp_constrained.py`) gives on the same tree. The
  free solve's cost always equals the minimum of the three.
- **Harmonization runs on private temporary copies.** `harmonize_leaf_boundaries`
  writes `*_filtered.tad` next to its inputs. Concurrent jobs share LT and CLP, so
  they would otherwise overwrite each other's filtered files.
- **Objective**: convex with `C_t = C_b = 1`, as in the paper, in all three
  analyses. The small parsimony hemato scripts, weighted ones included, default to
  `lex_boundary_first`. `MODE` in `quartet_rules.smk` can be changed for the
  unweighted analysis only, because the importance terms are implemented for
  convex only.
- **Solver settings**: `MIPGap = 0`, because comparing topologies needs true optima,
  not optima within 0.01%. With whole-number `C_t` and `C_b` every cost is a
  whole number, so `MIPGapAbs` is set just under 1. This stops the solver as
  soon as the gap is less than one unit, which is still exact. The weighted
  analyses have fractional costs, so they skip this setting. Their costs are
  compared with a 1e-6 tie tolerance.
- **Logs**: each job's log in `results/logs/` includes Gurobi's progress, so a
  long job can be checked while it runs.

# All ten cell types: quartet consistency (`ontad_unweighted_full/`)

Instead of one quartet at a time, every cell type is given — LT, ST, MPP, CLP,
CMP, MEP, GMP, MKP, MK, G (OnTAD, `hemato_cells/<NODE>/`) — as the leaves of one
large parsimony problem with no tree. The inferred tree is compared with the known
lineage

    LT -> ST -> MPP -> {CLP, CMP};  CMP -> {MEP -> MKP -> MK, GMP -> G}

by **quartet consistency**: of the 210 sets of four cell types, the 174 that some
edge of the known tree splits two-and-two are kept (the other 36 have no defined
split, e.g. {LT, MPP, CLP, CMP}); the score is the fraction the inferred tree splits
the same way. A random binary tree scores 1/3. Each tolerance is reported on its own.

**Solver.** The single ILP needs 9 minutes on chr19 at 1 Mb (31 candidate TADs)
just to prove optimality, so `topology_search.py` solves the same problem exactly
by branch and bound over topologies:

1. UB = exact cost of a greedy stepwise-addition tree.
2. Stepwise addition over all topologies, pruning with a Fitch lower bound (every
   observed TAD, start and end as an independent binary character: the objective
   without laminarity or TAD/boundary coupling; it never decreases when a leaf is
   added).
3. Surviving complete trees, in increasing Fitch bound, each settled by the
   cheapest step that proves its exact cost or rules it out:
   - the Fitch labeling of the TADs (boundaries derived from them): if laminar
     it is a feasible solution, exact when its cost equals the Fitch bound;
   - a grouped bound: each TAD's cost split half to its start and half to its
     end, each start/end group (its TADs, a phantom for unobserved TADs, the
     boundary) solved exactly on the tree. It keeps the TAD/boundary coupling
     the Fitch bound drops;
   - the LP relaxation without laminarity (one relaxed all-arcs model, tree
     imposed by arc bounds); exact if its solution is integral and laminar;
   - the fixed-tree small parsimony ILP with a cutoff at UB.
   Trees are stored as their insertion paths, and quartet consistency is read
   from a precomputed split -> consistent-quartets table, so millions of tied
   trees are cheap.

It returns **every** co-optimal tree; an instance's consistency is the mean over
them (min and max are recorded). This matters at 2 Mb, where few TADs make many
trees tie (chr19, 2 Mb: 1575 optimal trees). Checked against the single ILP on
chr19 at 2 Mb and 1 Mb (same optimum; the ILP's tree is among the co-optimal ones).

Timing, chr1 (largest), tolerance 1: 1 Mb 8 s, 500 kb 28 s, 250 kb 4 min.
Hardest cases: chr11 250 kb tol 0 in 20 min (72k trees with Fitch bound below the
optimum; the grouped bound rules out 70k of them); chr18 2 Mb, where all 2,027,025
trees are optimal, in 83 s. The certificates came in after 437 of the 456
instances had run with the plain LP/ILP scoring; both are exact and give the same
optimal trees (checked on chr19 2 Mb/1 Mb, chr1 1 Mb/500 kb, chr15 2 Mb).

| file | what it does |
|---|---|
| `infer_full_tree.py` | one instance: harmonize the ten calls, solve (`--method bnb`, default; `--method ilp` for the single ILP), score quartet consistency of every optimal tree |
| `topology_search.py` | the branch and bound |
| `aggregate_full_tree.py` | per bin size x tolerance: mean / median / min / max consistency over chromosomes, chromosomes matching the known tree exactly, median number of optimal trees |
| `ontad_unweighted_full/Snakefile` | the grid; output `results/summary/full_tree_summary.txt` |

## Weighted: `ontad_weighted_full/`

Same ten cells and scoring, with the paper's weighted TAD distance d_w
(`--weighting ontad_score`): each TAD change costs its OnTAD confidence
(column 5 of `tads.tad`), each boundary change the largest weight of the TADs
starting / ending there. Harmonization drops the score column, so each harmonized
TAD is traced back to the cell's OnTAD call(s) with the same boundary clusters; a
TAD's weight is the mean of its scores over the cells calling it, scaled so the
mean over observed TADs is 1. Unobserved candidate TADs cost 1.

The quartet analysis's `ontad_weighted/` (RobusTAD scores of MPP and CMP on the
internal vertices) does not carry over: with all ten cells as leaves no internal
vertex stands for a known cell.

The branch and bound handles d_w exactly: Fitch with costs floored to 1/1000
(binary planes of popcounts), the grouped bound and the Fitch labeling with the
per-TAD / per-boundary costs, and the LP / ILP with the weighted objective
(`tad_weights`, `start_weights`, `end_weights` of `LargeParsimonyILP`). Checked on
16 random trees (Fitch <= grouped <= LP <= exact <= labeling) and against the single
weighted ILP on chr19 at 2 Mb and 1 Mb (same optimum; its tree among the
co-optimal ones).

## RobusTAD: `robustad_unweighted_full/`, `robustad_weighted_full/`

Leaves from `hemato_cells/using_robustad/<NODE>/`. ST had no RobusTAD call:
`GSE152918_ST.hic` was downloaded from GEO and converted with hic2cool 0.8.3 like
the other nine (chromosomes renamed to `chr*` as in theirs), then
`robustad_predictions/infer_st.py` ran RobusTAD for ST with the saved OnTAD-biased
panels and windows of `build_ontad_panel_and_infer.py`, including its calibration
to the cell's own OnTAD calls. Like the other cells, most of ST's chromosomes fall
back to OnTAD-derived TADs (2 Mb: the assembly step fails for 9 of 10 cells;
1 Mb-250 kb: < 90% agreement on 21 of 22 chromosomes), so at fine bin sizes the
RobusTAD TAD sets are essentially OnTAD's and RobusTAD contributes the scores.

Weighted (`--weighting robustad_score`): TAD weights from RobusTAD's TAD score,
boundary weights from its left / right boundary scores, floored at 0.01, averaged
over the cells that have the TAD / boundary, each scaled to mean 1. A TAD ending
in bin b has its right boundary at bin b + 1 of the boundary file.
