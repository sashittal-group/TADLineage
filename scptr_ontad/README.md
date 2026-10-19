# SCPTR OnTAD clustering

This folder mirrors the updated `scptr_robustad` clustering workflow, but uses
native OnTAD calls from:

```text
../scptr/cell_differentiation/<stage>/<GSM>/<chr>/<resolution>/tads.tad
```

The per-chromosome rules cluster those existing `tads.tad` files directly.

The all-chromosome rules first run `merge_ontad_genomewide.py`, which builds
one full genomewide Hi-C matrix per cell and runs OnTAD once on that matrix.
It writes one genomewide `tads.tad` per cell under:

```text
results_updated/scptr_merged_all_chr/<resolution>/<stage>/<GSM>/all/<resolution>/tads.tad
```

The script offsets each chromosome's bins with `../genome.chrom.sizes` before
building the matrix, so the resulting OnTAD calls use one genomewide coordinate
system.

Run Snakemake from this directory using the Snakemake environment:

```bash
conda run -n snakemake-tutorial snakemake -j 1
```

If per-chromosome OnTAD files are missing and need to be inferred from the
stage-level contact `.txt` files, use the OnTAD environment:

```bash
conda run -n hic_analysis python infer_ontad_genomewide.py \
  --base-dir ../scptr/cell_differentiation \
  --ontad-bin ../OnTAD/OnTAD/src/OnTAD \
  --resolutions 1000kb \
  --merge-after
```

By default, `infer_ontad_genomewide.py` skips existing
`<chr>/<resolution>/tads.tad` files. Add `--overwrite` to rerun OnTAD.

For just the full-matrix genomewide OnTAD generation:

```bash
conda run -n snakemake-tutorial python merge_ontad_genomewide.py \
  --input-base ../scptr/cell_differentiation \
  --output-base results_updated/scptr_merged_all_chr \
  --ontad-bin ../OnTAD/OnTAD/src/OnTAD \
  --genome-sizes ../genome.chrom.sizes \
  --resolution 1000kb
```
