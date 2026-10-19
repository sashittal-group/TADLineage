#!/bin/bash
# args: tag weighting Ct Cb tol res...
tag=$1; w=$2; ct=$3; cb=$4; t=$5; shift 5
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python ../genome_tree.py --data-dir ../.. --resolution "$@" --tolerance $t --weighting $w --Ct $ct --Cb $cb --threads 1 --verbose --output jsons/${tag}_${t}.json > logs/${tag}_${t}.log 2>&1
