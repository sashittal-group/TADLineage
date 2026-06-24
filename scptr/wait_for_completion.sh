#!/bin/bash

BASE_DIR="/Users/PSLabMonisha/Desktop/tads/scptr/cell_differentiation"
EXPECTED_TADS=$((508 * 20 * 4))  # 508 cells × 20 chroms × 4 resolutions = 40,640
INTERVAL=300  # Check every 5 minutes

echo "=== Waiting for TAD Generation to Complete ==="
echo "Expected TAD files: $EXPECTED_TADS"
echo "Update interval: ${INTERVAL}s (5 minutes)"
echo ""

START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date)
    TAD_COUNT=$(find "$BASE_DIR" -name "tads.tad" 2>/dev/null | wc -l)
    RUNNING=$(pgrep -f "generate_matrices_and_tads.py" | wc -l)
    
    if [ $RUNNING -eq 0 ]; then
        PROCESS_STATUS="STOPPED"
    else
        PROCESS_STATUS="RUNNING"
    fi
    
    PERCENTAGE=$((TAD_COUNT * 100 / EXPECTED_TADS))
    ELAPSED=$(($(date +%s) - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    
    # Clear screen and show status
    clear
    echo "=== TAD Generation Progress ==="
    echo "Time: $CURRENT_TIME"
    echo "Elapsed: ${HOURS}h ${MINUTES}m"
    echo ""
    echo "TAD Files:     $TAD_COUNT / $EXPECTED_TADS ($PERCENTAGE%)"
    echo "Process:       $PROCESS_STATUS"
    echo ""
    
    # Progress bar
    FILLED=$((PERCENTAGE / 5))
    EMPTY=$((20 - FILLED))
    printf "Progress:      ["
    printf "%${FILLED}s" | tr ' ' '='
    printf "%${EMPTY}s" | tr ' ' '-'
    printf "] $PERCENTAGE%%\n"
    echo ""
    
    # Show memory usage if running
    if [ $RUNNING -gt 0 ]; then
        echo "Memory usage:"
        ps aux | grep "python generate_matrices_and_tads" | grep -v grep | awk '{printf "  PID %s: %.1f MB\n", $2, $6/1024}'
        echo ""
    fi
    
    # Check if we're done
    if [ $TAD_COUNT -ge $EXPECTED_TADS ]; then
        echo "✓ TAD generation COMPLETE!"
        break
    elif [ $RUNNING -eq 0 ] && [ $TAD_COUNT -lt $EXPECTED_TADS ]; then
        echo "⚠ Process stopped but not all TADs generated ($TAD_COUNT < $EXPECTED_TADS)"
        echo "Checking if there's an error..."
        
        # Check if there's a process still running (might just be slow to exit)
        if pgrep -f "generate_matrices_and_tads" > /dev/null 2>&1; then
            echo "Process still exists, waiting..."
        else
            echo "Process has exited. Check log for errors:"
            echo "  tail -100 tad_generation.log"
            break
        fi
    else
        echo "Checking again in $INTERVAL seconds..."
    fi
    
    sleep $INTERVAL
done

END_TIME=$(date)
TOTAL_ELAPSED=$(($(date +%s) - START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))

echo ""
echo "=== COMPLETION SUMMARY ==="
echo "Completed: $END_TIME"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "Final TAD count: $TAD_COUNT"
