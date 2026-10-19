#!/bin/bash
# After the main queue: the tol-0 top-fraction jobs, one at a time, with a larger
# LP cache (tol 0 has the most candidate TADs; ~120 KB each) and a 13 GB guard.
while pgrep -f "xargs -P 2 -L 1 ./run_task_to[p]" >/dev/null; do sleep 30; done
while pgrep -f "mem_guar[d].sh" >/dev/null; do sleep 10; done   # main queue's 10 GB guard exits
MAX_GB=13 MIN_FREE_PCT=15 ./mem_guard.sh &
RUN_TOL0=1 LP_BUDGET=55000 ./run_task_top.sh 0.25 100kb 0
#UN_TOL0=1 LP_BUDGET=80000 ./run_task_top.sh 0.33 100kb 0
