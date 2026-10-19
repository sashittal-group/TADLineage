#!/bin/bash
# One genome-wide job: weighting resolution tolerance. Skips finished jobs, so the
# whole task list can simply be relaunched after an interruption.
w=$1; r=$2; t=$3
out=jsons/${w}_${r}_${t}.json
[ -s "$out" ] && exit 0
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python ../genome_tree.py --data-dir ../.. \
    --resolution $r --tolerance $t --weighting $w --threads 1 --verbose --output $out > logs/${w}_${r}_${t}.log 2>&1
