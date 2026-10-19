# TAD-reorganization simulation

Simulated Hi-C along a small phylogenetic tree with **parsimoniously-evolving
TAD hierarchies**, for benchmarking internal-node TAD reconstruction (as in the
`hemato_cells` real-data work) with precision / recall / F1.

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

* **Given nodes** (inputs to reconstruction): `root`, `D`, `E`, `F`, `G`.
* **Scored nodes** (reconstructed & evaluated): `A`, `B`, `C` — every internal
  node is a branch point, so all three carry signal.

Topology and defaults live in [`config.py`](config.py).

## Design

The evolved hierarchy at each node is the **hidden generator** used to create
that node's simulated Hi-C matrix. The benchmark objects are the TADs called
from those matrices by OnTAD, matching the real-data protocol:

* `root` + leaf OnTAD calls (`D/E/F/G`) are the reconstruction inputs.
* internal-node OnTAD calls (`A/B/C`) are the evaluation ground truth.
* hidden evolved hierarchies are retained only for QC and sanity checks.

Parsimony is enforced in the evolver by a small per-edge event budget, so most
boundaries are conserved along every edge. Those shared structures are encoded
into the matrices as similar enriched contact blocks, and OnTAD should recover
many of them as shared called boundaries.

To avoid degenerate benchmark cases, generation now requires at least 20 hidden
TADs in every scored internal node. The matrix simulator also adds nonuniform
domain strengths and narrow boundary-insulation stripes, then OnTAD is run with
`-log2` and a permissive local-minimum cutoff. A QC gate fails the workflow if
OnTAD itself calls fewer than 20 TADs at any of `A/B/C`; no TADs are copied from
the hidden truth into the OnTAD outputs.

## Pipeline

1. **Root hierarchy** — a random laminar interval hierarchy over `N` bins
   ([`hierarchy.py`](hierarchy.py)).
2. **Evolve** each parent→child edge with `k` local, laminarity-preserving,
   boundary-reusing events ([`evolve.py`](evolve.py)):
   `split` (insert a boundary → two siblings), `merge` (drop a shared
   boundary), `insert`, `delete`, `shift_boundary`. Small `k` ⇒ parsimony.
   Nesting depth is capped at `nesting_depth`.
3. **Simulate** a Hi-C matrix at every node from its hierarchy via the
   Lun/Smyth recipe ([`simulate_matrix.py`](simulate_matrix.py)):
   distance-decay background `A·(|i−j|+1)^(−α)`, ×`enrichment` inside each
   domain (compounds for nested domains), `boundary_depletion` across adjacent
   sibling domains, narrow `boundary_stripe_depletion` around domain
   boundaries, per-domain strength jitter, NegBinom biological noise (`bcv`),
   uniform random-ligation background (`noise` = ρ), symmetrize, optional depth
   rescale.
4. **Call OnTAD** on every matrix ([`run_ontad.py`](run_ontad.py)):
   `ontad/{root,D,E,F,G}.tad` are observed inputs, and
   `ontad/{A,B,C}.tad` are observed internal-node truth.
5. **Reconstruct** `A/B/C` from the `ontad/{root,D,E,F,G}.tad` hierarchies —
   *next phase*, using the existing ILP / Fitch–Hartigan solver (not
   reimplemented here).
6. **Evaluate** boundary P/R/F1 of predicted vs `ontad/{A,B,C}.tad` at
   `A/B/C` ([`metrics.py`](metrics.py)) — *next phase*.

## Usage

This is the OnTAD-caller analysis. Matrix/truth **generation is shared** and
lives in the parent directory — run it first (see `../README.md`), then run the
Snakemake pipeline here.

### Shared generation (parent directory, once)

```bash
cd ..
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake --cores 8
```

Generates `N_INSTANCES` (100 by default) independent instances — each an
independent random root hierarchy evolved down the whole tree — writing matrices
and hidden truth under `../data/region_XXX/`.

### Call OnTAD + QC (this directory)

```bash
cd using_ontad
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake --cores 8
```

Writes OnTAD calls for every node into `../data/region_XXX/ontad/`,
`../data/ontad_count_check.txt`, and `../data/generation_report.txt` (hidden-truth
parsimony + OnTAD recovery/call-parsimony QC). Then reconstruct + plot:

```bash
snakemake -s Snakefile_methods --cores 8
```

### Manual / single-instance (no Snakemake)

```bash
# generation lives in the parent directory:
cd .. && python3.12 generate_data.py --region-index 7 && cd using_ontad
python3.12 run_ontad.py --log2 --ldiff 0.5        # call OnTAD on all matrices
python3.12 check_ontad_counts.py                  # require >=20 calls at A/B/C
python3.12 check_generation.py                    # QC report (stdout)
```

`python3.12` (`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`)
is the interpreter with numpy/scipy on this machine.

### Call OnTAD on one matrix

```bash
python3.12 run_ontad.py --region-index 7 --node A --log2 --ldiff 0.5
```

The OnTAD binary is expected at `../../OnTAD/OnTAD/src/OnTAD`.

Key generation knobs (all overridable on `generate_data.py`; see `--help`):
`--n-bins --n-top-tads --nesting-depth --min-size --gap-prob
--min-score-node-tads --events-per-edge --enrichment --boundary-depletion
--boundary-stripe-width --boundary-stripe-depletion --domain-strength-jitter
--bcv --noise --amplitude --alpha --seed`.

## Output layout (`data/`)

```
data/
  meta.json                        tree + params + node roles
  generation_report.txt            QC report (from check_generation.py)
  region_XXX/                      one independent instance (000..099)
    matrices/<node>.matrix         dense 1500x1500 counts
    truth/<node>.json              hidden evolved hierarchy (canonical);
                                    matrix generator + QC reference
    truth/<node>.tad               hidden evolved hierarchy, OnTAD .tad layout
    ontad/<node>.tad               OnTAD calls used by reconstruction/evaluation
```

`.tad` files are 1-indexed inclusive with a nesting level, matching OnTAD.

## Reference line: truth parsimony (100 instances, defaults)

| Edge | Boundary Jaccard(parent, child) |
|---|---|
| root → A | 0.79 |
| A → B | 0.79 |
| A → C | 0.78 |
| B → D | 0.77 |
| B → E | 0.79 |
| C → F | 0.77 |
| C → G | 0.77 |
| **mean** | **0.78** |

~0.78 boundary overlap parent→child confirms the evolved hierarchies are
parsimonious by construction — most TADs/boundaries are shared along every
edge, which is what makes internal-node reconstruction from `root` + leaves
feasible.
