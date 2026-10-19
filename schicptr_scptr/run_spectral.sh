#!/bin/bash
# Add the spectral clustering of Fig. 2b of the scHiCPTR paper across the whole grid.
# The per-chromosome reductions are cached, so this recomputes only the embedding and the
# clustering of it.
set -e
PY=/Users/PSLabMonisha/miniforge3/envs/schicptr/bin/python
cd "$(dirname "$0")"
for res in 1000000 500000 250000; do
  $PY run_schicptr_scptr.py --all-chroms --res "$res" --cluster-method spectral --ncpus 8
done
$PY plot_ari_summaries.py
