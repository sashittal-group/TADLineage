#!/bin/bash
# Kills a genome_tree.py job before it can exhaust memory (the 2026-10-01 02:10
# watchdog reboot). Every 10 s: any job above MAX_GB resident is killed; if the
# system's free memory falls below MIN_FREE_PCT, the largest job is killed.
# Killed jobs are logged to guard.log and simply rerun later (run_task.sh skips
# finished ones). Stops when no job has run for a minute.
MAX_GB=${MAX_GB:-6}
MIN_FREE_PCT=${MIN_FREE_PCT:-15}
idle=0
while [ $idle -lt 6 ]; do
    jobs=$(pgrep -f "python.*genome_tre[e].py")
    if [ -z "$jobs" ]; then idle=$((idle + 1)); sleep 10; continue; fi
    idle=0
    free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
    biggest=""; biggest_kb=0
    for p in $jobs; do
        kb=$(ps -o rss= -p $p 2>/dev/null | tr -d ' ')
        [ -z "$kb" ] && continue
        if [ "$kb" -gt $((MAX_GB * 1024 * 1024)) ]; then
            echo "$(date) killed $p ($((kb / 1024)) MB > ${MAX_GB} GB): $(ps -o command= -p $p | grep -o 'resolution.*weighting [a-z_]*')" >> guard.log
            kill $p
        elif [ "$kb" -gt "$biggest_kb" ]; then biggest=$p; biggest_kb=$kb; fi
    done
    if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_PCT" ] && [ -n "$biggest" ]; then
        echo "$(date) free memory ${free}%: killed largest $biggest ($((biggest_kb / 1024)) MB): $(ps -o command= -p $biggest | grep -o 'resolution.*weighting [a-z_]*')" >> guard.log
        kill $biggest
    fi
    sleep 10
done
