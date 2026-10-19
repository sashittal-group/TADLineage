#!/bin/bash
# Build the cached per-chromosome reductions and run every clustering, one bin size at a time.
# 250 kb is the expensive one, so the bin sizes run coarse to fine.
set -e
PY=/Users/PSLabMonisha/miniforge3/envs/schicptr/bin/python
cd "$(dirname "$0")"
for res in 1000000 500000 250000; do
  $PY run_schicptr_scptr.py --prepare-only --res "$res" --ncpus 8
  $PY run_schicptr_scptr.py --all-chroms   --res "$res" --ncpus 8
done
$PY plot_ari_summaries.py
