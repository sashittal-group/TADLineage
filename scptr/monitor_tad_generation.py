#!/usr/bin/env python3
"""
Monitor TAD generation pipeline progress and provide updates.
Runs continuously and logs progress to a file.
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime
import sys

BASE_DIR = Path("/Users/PSLabMonisha/Desktop/tads/scptr/cell_differentiation")
EXPECTED_TADS = 40640  # 508 cells × 20 chroms × 4 resolutions
INTERVAL = 300  # Check every 5 minutes
LOG_FILE = Path("/Users/PSLabMonisha/Desktop/tads/scptr/monitoring.log")

def get_tad_count():
    """Count TAD files generated."""
    try:
        result = subprocess.run(
            f"find {BASE_DIR} -name 'tads.tad' 2>/dev/null | wc -l",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return int(result.stdout.strip())
    except:
        return 0

def is_process_running():
    """Check if generation process is still running."""
    try:
        result = subprocess.run(
            "pgrep -f 'generate_matrices_and_tads.py' | grep -v awk | wc -l",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        count = int(result.stdout.strip())
        return count > 1  # Should have at least 2 processes (main + conda wrapper)
    except:
        return False

def get_memory_usage():
    """Get memory usage of main process."""
    try:
        result = subprocess.run(
            "ps aux | grep 'python generate_matrices_and_tads.py' | grep -v grep | awk '{print $6}' | head -1",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        mb = int(result.stdout.strip()) / 1024
        return f"{mb:.1f} MB"
    except:
        return "N/A"

def log_status(message):
    """Log message to both file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    
    print(log_msg)
    
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def main():
    """Main monitoring loop."""
    log_status("=== TAD Generation Monitoring Started ===")
    log_status(f"Expected total: {EXPECTED_TADS} TAD files")
    log_status(f"Check interval: {INTERVAL} seconds (5 minutes)")
    log_status("")
    
    start_time = time.time()
    last_count = 0
    
    while True:
        try:
            current_count = get_tad_count()
            is_running = is_process_running()
            memory = get_memory_usage()
            
            elapsed_secs = int(time.time() - start_time)
            hours = elapsed_secs // 3600
            minutes = (elapsed_secs % 3600) // 60
            
            percentage = (current_count * 100) // EXPECTED_TADS
            
            # Calculate rate
            if last_count > 0 and current_count > last_count:
                tads_since_last = current_count - last_count
                rate_per_hour = (tads_since_last * 3600) / INTERVAL
                remaining_tads = EXPECTED_TADS - current_count
                estimated_hours = remaining_tads / rate_per_hour if rate_per_hour > 0 else 0
                estimated_time = f", ~{estimated_hours:.1f}h remaining"
            else:
                estimated_time = ""
            
            status = "RUNNING" if is_running else "STOPPED"
            
            log_status(
                f"{status:7s} | TADs: {current_count:5d}/{EXPECTED_TADS} ({percentage:3d}%) | "
                f"Elapsed: {hours}h {minutes}m | Memory: {memory}{estimated_time}"
            )
            
            # Check completion
            if current_count >= EXPECTED_TADS:
                log_status("")
                log_status("✓ TAD GENERATION COMPLETE!")
                log_status(f"Total time: {hours}h {minutes}m")
                log_status("")
                break
            elif not is_running and current_count < EXPECTED_TADS:
                log_status("")
                log_status("⚠ WARNING: Process stopped but not all TADs generated!")
                log_status(f"Generated: {current_count}/{EXPECTED_TADS}")
                log_status("Check tad_generation.log for errors")
                log_status("")
                break
            
            last_count = current_count
            time.sleep(INTERVAL)
            
        except KeyboardInterrupt:
            log_status("\nMonitoring stopped by user")
            break
        except Exception as e:
            log_status(f"ERROR: {e}")
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
