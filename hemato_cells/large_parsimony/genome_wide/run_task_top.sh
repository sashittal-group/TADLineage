#!/bin/bash
# One genome-wide weighted job on each cell's top-scoring OnTAD calls: fraction resolution tolerance.
f=$1; r=$2; t=$3
out=jsons/ontad_score_${r}_${t}_top${f}.json
[ -s "$out" ] && exit 0
[ "$t" = 0 ] && [ -z "$RUN_TOL0" ] && exit 0   # tol 0 runs last, alone, with a bigger LP cache
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python ../genome_tree.py --data-dir ../.. \
    --resolution $r --tolerance $t --weighting ontad_score --top-fraction $f --threads 1 --verbose \
    --output $out > logs/ontad_score_${r}_${t}_top${f}.log 2>&1
