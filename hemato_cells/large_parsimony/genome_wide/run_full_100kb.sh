#!/bin/bash
# 100 kb with ALL OnTAD calls (no --top-fraction) at the tolerances where it fits in
# memory: tol >= 2 cuts chr1 from 53,918 candidate TADs (tol 0) to <= 23,554. Two
# jobs at a time (run_task.sh skips finished ones), 10 GB guard, ~5 GB LP cache each.
MAX_GB=10 MIN_FREE_PCT=12 ./mem_guard.sh &
printf "ontad_score 100kb %s\n" 5 4 3 2 | LP_BUDGET=40000 xargs -P 2 -L 1 ./run_task.sh
