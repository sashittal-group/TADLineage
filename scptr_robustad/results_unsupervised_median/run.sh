#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 0 ]; then
    echo 'run.sh reproduces the fixed configuration. Use run_pipeline.py for custom arguments.' >&2
    exit 2
fi
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_python="${EXPERIMENT_PYTHON:-/Users/PSLabMonisha/miniforge3/envs/hic_analysis/bin/python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$experiment_dir/cache"
export MPLCONFIGDIR="$experiment_dir/cache/matplotlib"
"$experiment_python" "$experiment_dir/run_pipeline.py"
"$experiment_python" "$experiment_dir/evaluate_final.py" --output final
