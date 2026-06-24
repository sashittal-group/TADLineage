#!/bin/bash
# Monitor TAD generation progress

BASE_DIR="/Users/PSLabMonisha/Desktop/tads/scptr/cell_differentiation"

echo "=== TAD Generation Progress ==="
echo "Timestamp: $(date)"

# Count TAD files generated
TAD_COUNT=$(find "$BASE_DIR" -name "tads.tad" 2>/dev/null | wc -l)
echo "TAD files generated: $TAD_COUNT"

# Count cells started
CELL_COUNT=$(find "$BASE_DIR" -maxdepth 3 -type d -name "GSM*" 2>/dev/null | wc -l)
echo "Cells processing: $CELL_COUNT"

# Check if process still running
RUNNING=$(pgrep -f "generate_matrices_and_tads.py" | wc -l)
echo "Processes running: $RUNNING"

# Show memory usage
if [ $RUNNING -gt 0 ]; then
    echo ""
    echo "Memory usage:"
    ps aux | grep "python generate_matrices_and_tads" | grep -v grep | awk '{print "  PID " $2 ": " $6/1024 " MB"}'
fi

# Show sample of latest TADs
echo ""
echo "Sample TAD directories:"
find "$BASE_DIR" -name "tads.tad" -newer /tmp 2>/dev/null | head -5 | while read f; do
    echo "  $(dirname $f)"
done
