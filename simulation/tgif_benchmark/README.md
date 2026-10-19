# Benchmarking TGIF-DB against the tree-informed imputation arm

**What this folder is.** [`../tree_informed_tad_inference_robustad`](../tree_informed_tad_inference_robustad)
claims that a tree of related samples carries information a per-leaf TAD caller
throws away, and that using it repairs the leaf's calls on noisy Hi-C.
[TGIF](https://github.com/Roy-lab/tgif) (Roy lab) makes the same bet by the
opposite mechanism. This folder runs it on **the identical simulated data** and
scores both against the same hidden truth.

```
ours     RobusTAD calls each leaf independently  ->  tree-informed ILP repairs the calls
TGIF-DB  tree-regularized multi-task NMF over all leaves at once  ->  boundaries fall out of the fit
```

Same inputs (leaf matrices + a tree), same output (per-leaf boundaries), two
genuinely different mechanisms. That is what makes it a real baseline rather
than a strawman, and it is what the professor asked for.

TGIF also ships **TGIF-DC** for compartments. It is compiled by `build_tgif.sh`
but nothing here uses it — compartments are a different problem from TADs.

### What TGIF can and cannot be benchmarked on

This repo runs two different experiments on the tree, and TGIF is only a
baseline for one of them.

| our experiment | can TGIF do it? |
| --- | --- |
| **leaf repair** (`tree_informed_tad_inference_*`) — better boundaries at the *observed* leaves, using the tree | **yes.** This is exactly TGIF-DB's job, and it is what this folder benchmarks. |
| **internal-node reconstruction** (`using_ontad/`, `using_robustad/`, `hemato_cells/`) — the TAD hierarchy at *unobserved* ancestors | **no, not as shipped.** |

The second row is a property of TGIF's output, not an oversight. `tgif-db`
computes its boundary score from each **leaf's** `U` factor
(`get_cosine_distance_to_upstream_neighbor(((Leaf*) tree[t])->get_U(), ...)`),
loops `t` over the leaf tasks only, and its output tables carry exactly one
column per input matrix. Internal nodes have no matrix, get no `U`, and appear
nowhere in the output.

There *is* a plausible extension — every node, internal ones included, carries a
`V` factor of the same shape, so the same cosine-distance-to-upstream-neighbour
readout could in principle be taken on an internal node's `V` to give it a
boundary profile. But `tgif-db` has no flag to write factors at all (`-u` exists
only in `tgif-dc`), so that is a change to TGIF's source and a research question
in its own right, not a benchmark. Worth raising with the professor if they want
the ancestor-reconstruction claim compared to TGIF too; it is not what this
folder does.

---

## Run it

```bash
cd simulation/tgif_benchmark
./build_tgif.sh                                    # once: clone, patch, compile
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake --cores 4 -p
```

One region end to end:

```bash
snakemake --cores 2 results/tree00/region_000.json
```

**Only one interpreter is needed**, unlike the RobusTAD arm: TGIF links the GSL
that lives in `snakemake-tutorial`, and the scoring stage needs `gurobipy` from
the same env.

Long runs, as elsewhere in this repo:

```bash
screen -dmS tgif caffeinate -is /Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake \
    --cores 4 -p --rerun-incomplete --keep-going
```

---

## Building TGIF on this machine: three real bugs

TGIF is written for Linux/gcc with a bundled GSL that is **not actually in the
repo** (`gsl/lib` is empty). `build_tgif.sh` clones it, applies the fixes below
idempotently, and compiles against the GSL 2.7 in the `snakemake-tutorial` env
— the only GSL here, and the version TGIF's README requires. Re-running the
script is safe; re-cloning from scratch and re-running it reproduces everything.

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | `error: call to undeclared function 'gsl_linalg_QR_decomp'` | `modules/random_svd/matrix_vector_functions_gsl.c` never includes `<gsl/gsl_linalg.h>`; that was legal under C89's implicit declarations, and clang 17 defaults to C23 where it is an error | `-include gsl/gsl_linalg.h` at compile time (no source edit) |
| 2 | **every run died with SIGTRAP** inside `TGIF::clean()`, on the first window, before any output file existed | `clean()` does `delete tree[i]` on a `vector<Node*>` where `Node` is **abstract with a non-virtual destructor**. That is UB, and clang emits a literal trap instruction for it (`-Wdelete-abstract-non-virtual-dtor`) | `virtual ~Node()` in `modules/node.h`. Also makes `~Leaf()` run, which frees the residual matrix `R` |
| 3 | **SIGSEGV in `io::write_sigDB`** at every noisy rung, *after* the boundary files were written | `for (i = 0; i < sigDbs.size()-1; i++)` then `sigDbs.back()`. When a pair of tasks shares no significantly differential boundary — the normal case once the matrices are thinned and nothing is called — `sigDbs` is empty, `size()-1` wraps as `size_t`, and it walks off the end | early return on `sigDbs.empty()`, so the pair gets a header-only file |

Two more things that are not bugs but will waste an afternoon:

* **Options must come *before* the two positional arguments**, the opposite of
  the order in TGIF's README. BSD/macOS `getopt` does not permute `argv`, so
  `./tgif-db tree.txt bins.bed -o out/` stops at the first non-option, sees
  `argc - optind == 6`, and silently prints the usage text instead of running.
* **TGIF hard-floors the factorization window at 100 bins** (and the stride at
  half of it) regardless of the `-w`/`-s` you pass in bp. `--window-bins` in
  `run_tgif_bench.py` is specified in bins for that reason, so the log says what
  actually ran.
* **`-a` is read with `atof` but stored in an `int`** (`Node::alpha`), so a
  fractional alpha silently truncates — `-a 0.5` is `-a 0`. TGIF's own usage
  text does say "specify an integer value"; keep alphas integral.

---

## The comparison is at the boundary level, and it has to be

TGIF-DB emits **per-bin boundary calls**; it never assembles nested TAD
intervals. Our methods emit TAD intervals. There is no way to score TGIF on
intervals, so everything is scored on boundaries:

* a boundary is an **interface between bins**, not a bin: a 0-indexed half-open
  domain `[s, e)` has interfaces `s` and `e`, so an interval `(a, b)` (1-indexed
  inclusive) contributes `{a-1, b}`;
* predicted boundaries are matched to true ones **one-to-one within
  ±`tolerance`** (2 bins), the same rule and the same tolerance the arm uses on
  TADs;
* the two region edges (`0` and `n_bins`) are dropped — every method knows them
  for free, and TGIF cannot report interface 0 at all.

**The interface convention is load-bearing, not cosmetic.** TGIF's score at
index *i* is the cosine distance between bin *i−1* and bin *i*, i.e. the
interface at *i* — so it speaks the same language. Scoring domains by their
first and last *bin* (`{a-1, b-1}`) instead would put every right-hand boundary
one bin off, and worse: two tiling domains would yield two adjacent truth
boundaries (`e-1` and `e`) for one physical interface. Our interval-emitting
methods would reproduce that pair and match both; TGIF calls the interface once
and could only match one, so its recall would be halved by the convention alone.

`f1_interval` is still recorded for our methods so this experiment's numbers can
be lined up against the arm's own figures — but the headline number is the
boundary one, because it is the only one TGIF can be part of.

**Two scorings are reported for every method, and both are in the CSVs:**

| suffix | what it means |
| --- | --- |
| `_full` | over the whole region. TGIF is charged for truth in the bins it dropped. This is the honest number: the task is "call this leaf's boundaries". |
| `_masked` | restricted to the bins TGIF evaluated, applied to **every** method alike. The charitable number: how good is TGIF *where it chose to speak*. |

Neither is cherry-picked; they answer different questions and the gap between
them is itself a result.

One caveat on `_masked`, since it is not always the kinder number: masking drops
truth **and** predictions inside the N/A bins, so a kept prediction whose
matching truth boundary fell inside the mask becomes a false positive at the
mask's edges. That is why our methods' masked F1 is sometimes *lower* than their
full F1. The effect applies to every method alike, but it means `_masked` should
be read as "restricted to TGIF's field of view", not as "an upper bound".

---

## The sparsity filter is the single biggest adaptation finding: turn it OFF

TGIF drops a bin — in **every** task — when fewer than `-t` of that bin's
within-window neighbours are non-zero in **any one** task, and marks it `N/A`.
Its own default is `0.5`, tuned for deep real Hi-C; its README suggests `0.2`
for sparse input. **On this simulation both cripple it, and benchmarking at
either would have been unfair to the point of being wrong.** Measured on
`tree01/region_000`, boundary F1 at `1.0x`:

| `-t` | what TGIF does |
| --- | --- |
| `0.2` (TGIF's advice for sparse data) | calls **10** boundaries across 10 leaves, half the region `N/A`, and then **refuses to run at all at every noisier rung** |
| **`0` (shipped here)** | full coverage at every rung, F1 **0.691** at `1.0x` against RobusTAD's 0.761, and 0.48–0.55 at the noisy end |

That is the difference between a baseline that competes and a baseline that
declines to, so `TGIF_THRESHOLD = 0` is what the `Snakefile` ships. The price is
runtime: ~50 s per (region, depth) instead of ~5 s, because TGIF now factorizes
the whole region instead of a fraction of it.

**Even at `-t 0` the coverage figure stays the first thing to read.** The filter
still bites at the bottom of the ladder (a bin with *no* non-zero neighbours is
dropped at any threshold — 286/500 bins at `0.003x`), and past a point TGIF
cannot run at all. A TGIF F1 of ~0 under high `N/A` is *"TGIF declined to call
anything here"*, not *"TGIF called and was wrong"*. Those are different failures
and must not be reported as the same one; `n_bins_na` and `tgif_status` are
recorded at every rung so the distinction stays recoverable from the CSVs.

Past a point the filter stops TGIF from running **at all**: when fewer bins
survive than one factorization window, TGIF dies inside GSL — with
`matrix dimension n1 must be positive integer` at exactly zero survivors, and
`failed to allocate space for row indices` at anything under the window size,
where its `nzmax = N + (windowSize-1)*(2N - windowSize)` goes negative. Both are
the same event, and `run_tgif_bench.py` records them as `status:
"too-few-bins"` with an empty call set, every bin N/A, and `n_bins_usable`
kept so "nothing survived" stays distinguishable from "a few bins survived, but
fewer than one window". **Every other non-zero exit is re-raised** — an
uncharacterised crash must never be quietly scored as "TGIF found nothing",
which would turn a broken run into a baseline defeat.

`--tgif-threshold 0.2` reproduces the crippled behaviour above, if you want to
see the filter's effect for yourself.

---

## Identical data, not merely similar data

The comparison only means anything if TGIF and RobusTAD saw the *same* leaf
matrices, so `run_tgif_bench.py` replays the arm's generation rather than
re-deriving it:

* `rt.build_hierarchies` is **imported** from the arm, not copied, so the hidden
  truth is the same object;
* the base matrices are drawn from one `sim_rng` in `tree.leaves` order;
* the degradation draws come from one `deg_rng` consumed **depth-major, leaf-minor**
  — and `rt.downsample()` does **not** touch the RNG when `frac >= 1.0`, so the
  `1.0x` rung consumes nothing and every later rung's draws depend on that.

The generation block in `base_matrices()` is a deliberate mirror of
`rt.run_call_stage`; if that block ever changes in the arm, it has to change
here too, or the two experiments quietly stop being about the same leaves. Every
simulation parameter in the `Snakefile` is likewise copied from the arm's
`Snakefile` and must stay equal to it.

The scoring stage **re-solves** our ILP from the arm's cached calls rather than
reading its `results/` JSONs. Those files hold interval-level *metrics*, and
boundaries cannot be recovered from a metric. The arm's own `method_raw` and
`method_impute` are called at the arm's own defaults, so `robustad` and `wsnap`
here are the same objects that produced its figures.

The `Snakefile`'s `TREES`/`REGIONS` are **the arm's full grid, 10 trees × 20
regions**, and they have to be. This figure is shown next to the arm's, and two
curves are only comparable if they are means over the *same* 200 (tree, region)
pairs — matching every generation parameter is not enough, because a smaller
grid is still a different sample of regions. It also means the arm's cached
calls (`../tree_informed_tad_inference_robustad/calls/`) exist for every point
here; where one is missing the score stage warns and scores TGIF alone.

---

## Methods scored

| method | what it is |
| --- | --- |
| `tgif` | TGIF-DB, `significant_boundaries_summit_only.txt` — one bin per significant stretch. The primary TGIF number. |
| `tgif_wide` | TGIF-DB, `significant_boundaries.txt` — whole significant stretches. Higher recall, lower precision; reported so the summit choice is not doing hidden work. |
| `robustad` | raw RobusTAD calls — the no-tree baseline, shared with the arm. |
| `snap` | the arm's unweighted tree-informed ILP. |
| `wsnap` | the arm's weighted tree-informed ILP (its headline). |

**`tgif` vs `wsnap` is the comparison that matters.** Both use the tree;
`robustad` is the shared no-tree baseline they are each trying to beat, and the
question is which mechanism — joint tree-regularized factorization, or
independent calls repaired by a tree-informed ILP — buys more, and where on the
noise ladder.

---

## Give TGIF its best shot before quoting a number against it

This turned out to matter more than anything else in the adaptation: at the
settings TGIF's own documentation suggests for sparse data, it scored ~0.25
boundary F1 and refused to run on most rungs; with one parameter changed it
scores ~0.81 and **beats** our baseline on clean data. A benchmark is only worth
what the loser's configuration is worth, so tune before reporting.

TGIF is also being run outside the regime it was built for, and saying so is
part of reporting the result honestly. Three mismatches:

* **Domain size.** This simulation puts boundaries as little as 8 bins apart
  (`MIN_SIZE = 8`), and much of a region is a dense run of small domains. TGIF
  was designed for 25 kb bins where domains are tens of bins wide, and
  `significant_boundaries_summit_only.txt` collapses each significant *stretch*
  to one summit — so a run of true boundaries 8–15 bins apart is reported as
  fewer calls than there are boundaries. `tgif_wide` is scored alongside for
  exactly this reason: it is the same run without the summit collapse.
* **Window.** TGIF hard-floors the factorization window at 100 bins, a fifth of
  a 500-bin region here. Boundary detection is local so this is not fatal, but
  nothing about the window can be tuned downward.
* **Counts.** `AMPLITUDE = 7.8` makes these matrices far sparser than the real
  Hi-C TGIF assumes, which is what drives the N/A rate above.

None of this is a reason to discount the comparison — both methods get the same
data, and "one method degrades more gracefully on sparse input" is a legitimate
finding, which is in fact what the pilot shows. It *is* a reason to say plainly
in any write-up which regime the comparison is in: these are 500-bin, count-poor
regions with domains as small as 8 bins, not the deep 25 kb Hi-C TGIF's paper
uses.

### The alpha pilot, and a trap worth not repeating

`alpha` (`-a`) is TGIF's tree-regularization strength — its analogue of our
`impute_cost`, the knob that says how much the tree is allowed to matter. The
shipped value is TGIF's own default, `1e6`, and getting there took a wrong turn
that is worth recording.

Swept **at the old `-t 0.2`**, on `tree00/region_000` at `1.0x`:

| alpha | 1e3 | 1e5 | 1e6 (TGIF default) | 1e8 | 1e9 | 2e9 |
| --- | --- | --- | --- | --- | --- | --- |
| boundary F1 | 0.222 | 0.224 | 0.223 | **0.299** | 0.303 | 0.303 |
| summit calls / 10 leaves | 75 | 74 | 75 | 58 | 61 | 61 |

Flat over three orders of magnitude, then a clean step up at `1e8` that
plateaus. It looks like a +0.08 F1 win for a stronger tree prior, and the
`Snakefile` briefly shipped `1e8` because of it.

Re-swept **with the filter off** (`-t 0`), on `tree01/region_000` at `1.0x`, the
ordering **reverses**:

| alpha | 1e6 (TGIF default) | 1e8 |
| --- | --- | --- |
| boundary F1 | **0.727** | 0.691 |

The step was an artifact of tuning inside a configuration where TGIF was barely
running. **Tune the threshold first, then alpha — never the reverse**, and treat
any parameter conclusion drawn under heavy `N/A` coverage as provisional.

Both sweeps are one region at one depth. Re-run over ~5 regions before treating
either table as settled. `alpha` is stored in an `int` inside TGIF, so it cannot
go far past `2e9`; the plateau arrives well before that.

A cheap pilot, one depth rung only, is enough to see whether a knob moves TGIF
at all (`--depths` skips the TGIF run at the other rungs while still consuming
the ladder's RNG, so the matrices stay identical):

```bash
PY=/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python
for a in 1000 100000 1000000 100000000; do
  $PY run_tgif_bench.py --tree tree00 --region-index 0 --stage call \
      --depths 1.0x --tgif-alpha $a --outdir /tmp/alpha_$a --work-dir /tmp/alphawork
  $PY run_tgif_bench.py --tree tree00 --region-index 0 --stage score \
      --tgif-json /tmp/alpha_$a/region_000.json \
      --arm-calls-json ../tree_informed_tad_inference_robustad/calls/tree00/region_000.json \
      --outdir /tmp/alpha_${a}_res
done
```

**A `--depths` cache is partial and must never be mixed into `results/`
alongside full ones** — the plots would average different sets of rungs.

## Knobs worth sweeping

| knob | why |
| --- | --- |
| `TGIF_THRESHOLD` (`-t`) | **sweep this first, always.** It decides whether TGIF runs at all, and every other parameter conclusion is conditional on it. Shipped at 0. |
| `TGIF_ALPHA` (`-a`) | tree-regularization strength — TGIF's analogue of our `impute_cost`. Shipped at TGIF's default; the pilot found no better value once the threshold was right. Only meaningful to sweep *after* the threshold. |
| `WINDOW_BINS` | 100 is TGIF's floor and 2× the mean top-level TAD here (500 bins / 10 domains). Larger windows are slower and blur small domains. |

---

## Result (full grid: 10 trees × 20 regions = 200)

Shipped settings (`-t 0`, `alpha = 1e6`). TGIF ran successfully at every rung of
every region except three at `0.003x`, where too little of the region survived
for it to run at all. TGIF's own curve — the line `f1_vs_noise.png` draws:

| depth | boundary F1 | ±1 SD | precision | recall | calls/leaf | bins N/A |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0x | 0.706 | 0.109 | 0.646 | 0.826 | 20.6 | 0.0 |
| 0.3x | 0.674 | 0.086 | 0.605 | 0.815 | 21.5 | 0.1 |
| 0.1x | 0.652 | 0.108 | 0.600 | 0.798 | 22.1 | 0.7 |
| 0.03x | 0.599 | 0.120 | 0.512 | 0.821 | 27.4 | 12.7 |
| 0.01x | 0.475 | 0.145 | 0.391 | 0.745 | 35.7 | 40.1 |
| 0.003x | 0.268 | 0.137 | 0.276 | 0.339 | 27.4 | 172.5 |

**TGIF is a real baseline, and it fails in the opposite direction to ours.** Its
recall holds at ~0.80–0.83 down to `0.03x` — it keeps calling as the data
degrades, ~21 calls per leaf at `1.0x` rising to ~36, against ~16 true
boundaries — so what collapses is precision (0.65 → 0.28). The callers do the
reverse: they stop calling rather than calling wrongly, which is exactly the
failure the tree-informed imputation was built to repair.

Against our lines on the same 200 regions, paired by `(tree, region, depth)`
(`plots/paired_gain.csv`), TGIF is level with the caller at the clean end and
loses badly at the bottom — `robustad − tgif` is +0.002 ± 0.009 at `1.0x` and
**+0.149 ± 0.007** at `0.003x`. The three-line figures in each arm's `plots/`
folder tell the whole story; see below.

`tgif_wide` is worse than `tgif` by 0.17–0.24 at every rung, so the summit
collapse is helping TGIF rather than hiding its performance — worth stating,
since dropping it would look like a way to boost TGIF's recall.

## The figure that goes in the paper lives in the ARMS, not here

`make_arm_comparison.py` adds a TGIF line to each arm's leaf figure, writing
into that arm's own `plots/` folder so the comparison sits next to the
experiment it belongs to:

```bash
python make_arm_comparison.py --arm robustad   # -> ../tree_informed_tad_inference_robustad/plots/
python make_arm_comparison.py --arm ontad      # -> ../tree_informed_tad_inference_ontad/plots/
```

Both arms work because they **share the simulation**: same hierarchies, same
seeds, same down-sampling draws in the same order, differing only in caller and
solver. So one TGIF run per (tree, region, depth) is the correct TGIF line for
both.

### Two figures, because there is no single honest one

TGIF has **no interval output at all**, so "the arm's figure plus one more line"
cannot be done on a single consistent metric. Both versions are written:

| file | what it is |
| --- | --- |
| `f1_vs_noise_with_tgif_interval.png` | the arm's own figure, reproduced **exactly** — its metric (interval F1), its numbers straight from its `results/`, its filters — with TGIF overlaid. The grey and blue curves are identical to the arm's existing `f1_vs_noise.png`. TGIF's line is boundary F1, so **two metrics share one axis**, flagged in the title and in TGIF's legend entry. |
| `f1_vs_noise_with_tgif_boundary.png` | all three lines on boundary F1. One consistent axis, but the grey and blue curves will **not** match the arm's existing figure, because that one is interval F1. |
| `f1_vs_noise_with_tgif_boundary.csv` | the per-region boundary scores behind it |

The arm's own `f1_vs_noise.png` is never touched.

### The arm's filters are honoured, and imported from the arm

`MIN_METRIC_DEFAULT` and `EXCLUDE_DEPTHS_DEFAULT` are read out of each arm's own
plot module rather than restated, so they cannot drift:

| arm | min-metric | excluded depths |
| --- | --- | --- |
| robustad | 0.4 | none |
| ontad | 0.5 | **`0.003x` dropped entirely** — so its x-axis stops at `0.01x` |

Two rules on top, both to avoid manufacturing the comparison. Depth exclusions
apply to **every** line including TGIF — an excluded rung is simply not on the
axis. The min-metric filter is applied exactly as the arm applies it: to the
tree-informed line only, never to the baseline, and **never to TGIF**. That
filter is survivorship bias (the robustad arm's own plot module documents this
at length); it is reproduced rather than fixed so these figures match the arm's,
but filtering only one side of a comparison would be a different thing entirely.

### Nothing is re-solved twice

The boundary scores are computed once and cached to
`f1_vs_noise_with_tgif_boundary.csv`; later runs reuse it. That one computation
is unavoidable — the arms' `results/` store interval-level *metrics*, and a
boundary set cannot be recovered from an F1 number — but `--recompute` is the
only way to trigger it again.

### What the boundary figures show (200 regions, filters applied)

| depth | RobusTAD alone | ours | TGIF | | OnTAD alone | ours | TGIF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0x | 0.708 | 0.692 | 0.706 | | 0.710 | 0.707 | 0.706 |
| 0.3x | 0.693 | 0.677 | 0.674 | | 0.716 | 0.716 | 0.674 |
| 0.1x | 0.694 | 0.675 | 0.652 | | 0.696 | **0.730** | 0.652 |
| 0.03x | 0.614 | **0.685** | 0.599 | | 0.593 | **0.687** | 0.599 |
| 0.01x | 0.493 | **0.598** | 0.475 | | 0.449 | **0.635** | 0.475 |
| 0.003x | 0.417 | **0.586** | 0.268 | | *excluded by this arm* | | |

The pattern is the same in both arms: **all three methods are level on clean
data, and the tree-informed line pulls away as the Hi-C degrades while TGIF
falls fastest.** Paired by `(tree, region, depth)` on the RobusTAD arm,
ours − TGIF runs from −0.014 ± 0.009 at `1.0x` to **+0.221 ± 0.008** at
`0.003x`.

Read the `0.01x` and `0.003x` rows together with `tgif_coverage.png`: TGIF's
sparsity filter has dropped 40 and 172 of 500 bins there on average, so part of
its decline is reduced coverage rather than wrong calls.

## Cost

At `TGIF_THRESHOLD = 0` TGIF is ~30–60 s per (region, depth), so ~4–5 min per
region over the 6-rung ladder — comparable to RobusTAD's ~8 min per region in
the arm. (At `-t 0.2` it was ~5 s, because it was factorizing a fraction of the
region; that speed was the symptom, not a saving.)

The grid is the arm's full **10 trees × 20 regions**, so ~4 h at `--cores 4`.
That is not padding: the figure here is meant to be read next to the arm's, and
two curves are only comparable if they are means over the *same* 200 (tree,
region) pairs. A cheaper grid would be a different sample of regions even though
every generation parameter matches. Calls are cached, so re-scoring and
re-plotting after the one long run are seconds.

By default the per-depth TGIF inputs and raw outputs are deleted after parsing
(`--keep-work` retains them, ~40 MB per region). Note TGIF writes one pairwise
differential file per **pair** of leaves — 45 files per run at 10 leaves —
none of which this benchmark uses.

## Outputs

```
calls/<tree>/region_<idx>.json     cached TGIF boundary calls (summit + wide),
                                   N/A bin list, and timings, per depth level
results/<tree>/region_<idx>.json   per-method boundary P/R/F1 per depth level,
                                   scored both full-region and masked
plots/f1_vs_noise.png              HEADLINE: mean leaf boundary F1 down the
                                   noise ladder. Same name, axes, styling and
                                   ±1 SD ribbon as the arm's figure, so the two
                                   can be shown side by side
plots/f1_vs_noise_masked.png       the same, restricted to bins TGIF evaluated
plots/precision_recall.png         P and R separately -- the methods fail in
                                   opposite directions, and the mean F1 hides it
plots/tgif_coverage.png            fraction of bins TGIF marked N/A. READ FIRST.
plots/leaf_metrics_long.csv        every per-region point, ALL scored methods
plots/leaf_metrics_summary.csv     pooled mean / sd / sem / n per (depth, method)
plots/paired_gain.csv              per-rung paired differences, against both the
                                   no-tree baseline and TGIF
```

**Only TGIF is DRAWN by default** (`PLOT_METHODS = "tgif"` in the `Snakefile`).
Everything else is still scored and still written to the CSVs; the figure
contributes the one line the arm cannot produce, on identical axes over the
identical 200 regions, and `robustad` / `snap` / `wsnap` come from the arm's own
`f1_vs_noise.png` rather than being drawn twice. `PLOT_METHODS = "all"` puts
them all on one figure if you want the comparison in a single place.

Gains are reported **paired** by `(tree, region, depth)`: every method scores the
same leaf sets, so the paired difference has a far smaller standard error than
the gap between two independent means. Unpaired SEMs would understate the
effect — the same convention the arm uses.
