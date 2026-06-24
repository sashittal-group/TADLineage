#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/PSLabMonisha/miniforge3/envs/refhic/bin/python}"

cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.mplcache}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_ontad_panel_and_infer.py" "$@"
