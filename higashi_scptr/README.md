# Higashi scptr

This folder prepares and runs chromosome-specific Higashi jobs for the `scptr/cell_differentiation` dataset.

## Files

- `setup_inputs.py` converts each per-cell contact file into a Higashi v2 input file for one chromosome.
- `run_higashi_scptr.py` writes a run-specific config and runs training/imputation.
- `cluster_embeddings.py` clusters Higashi embeddings, not compartment features.
- `plot_clusters.py` plots the embedding clusters.
- `Snakefile` runs the full Higashi embedding clustering workflow for chr1 through chr5 at 1000kb.
- `config.json` is the base CPU-only configuration; run-specific copies are written under `results/<resolution>/<chromosome>/config.json`.

## Usage

Run the full chr1-chr5, 1000kb workflow with Snakemake:

```bash
cd higashi_scptr
snakemake --cores 1
```

To inspect the planned jobs without running them:

```bash
cd higashi_scptr
snakemake --dry-run --cores 1
```

Outputs are isolated by run:

- processed data: `higashi_scptr/data/<resolution>/<chromosome>/processed`
- Higashi temp/model/embed files: `higashi_scptr/temp/<resolution>/<chromosome>`
- non-compartment clustering results: `higashi_scptr/results/<resolution>/<chromosome>/cluster_results.txt`
- cluster figure: `higashi_scptr/results/<resolution>/<chromosome>/cluster_figure.png`

For the provided Snakefile, `<resolution>` is `1000000` because 1000kb is passed to the scripts as 1,000,000 bp.
