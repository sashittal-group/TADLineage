#!/bin/bash

# Script to create a 500 kb RobusTAD reference panel from mouse hemato cell samples
# Uses LT, MK, G, and CLP samples

set -e  # Exit on error

# Define paths
MCOOL_DIR="/Users/PSLabMonisha/Desktop/tads/refhic_predictions/converted_mcool"
OUTPUT_DIR="/Users/PSLabMonisha/Desktop/tads/robustad_predictions_new"
ROBUSTAD_ROOT="/Users/PSLabMonisha/Desktop/tads/RobusTAD"
REFERENCE_PANEL_DIR="${OUTPUT_DIR}/reference_panel/500kb"
BOUNDARY_SCORES_DIR="${REFERENCE_PANEL_DIR}/boundary_scores"
ROBUSTAD_CMD=(conda run -n refhic python -m robustad.robustad)
RESOLUTION=500000
MINW_KB=500
MAXW_KB=6000
MIND_KB=500

# Create directories
mkdir -p "${BOUNDARY_SCORES_DIR}"
mkdir -p "${OUTPUT_DIR}/.matplotlib"

export PYTHONPATH="${ROBUSTAD_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${OUTPUT_DIR}/.matplotlib"

# Samples to use for reference panel
SAMPLES=("LT" "MK" "G" "CLP")

echo "=========================================="
echo "Creating 500 kb Reference Panel for RobusTAD"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Samples: ${SAMPLES[@]}"
echo ""

# Step 1: Compute boundary scores for all samples
echo "Step 1: Computing boundary scores for all samples..."
echo "=========================================="

for SAMPLE in "${SAMPLES[@]}"; do
    MCOOL_FILE="${MCOOL_DIR}/GSE152918_${SAMPLE}.mcool"
    OUTPUT_PREFIX="${BOUNDARY_SCORES_DIR}/GSE152918_${SAMPLE}"
    
    if [ ! -f "${MCOOL_FILE}" ]; then
        echo "ERROR: File not found: ${MCOOL_FILE}"
        exit 1
    fi
    
    echo "Processing: ${SAMPLE}"
    echo "  Input: ${MCOOL_FILE}"
    echo "  Output: ${OUTPUT_PREFIX}.bed"
    
    "${ROBUSTAD_CMD[@]}" boundary \
        "${MCOOL_FILE}" \
        "${OUTPUT_PREFIX}" \
        --resol "${RESOLUTION}" \
        --mind "${MIND_KB}" \
        --minW "${MINW_KB}" \
        --maxW "${MAXW_KB}" \
        --ratio 1.2 \
        --alpha 0.05
    
    echo "  ✓ Completed"
    echo ""
done

# Step 2: Create the reference database
echo "Step 2: Creating reference database..."
echo "=========================================="

# Create a file listing all boundary files
BOUNDARY_LIST="${REFERENCE_PANEL_DIR}/boundary_files.txt"
> "${BOUNDARY_LIST}"  # Clear the file

for SAMPLE in "${SAMPLES[@]}"; do
    echo "${BOUNDARY_SCORES_DIR}/GSE152918_${SAMPLE}.bed" >> "${BOUNDARY_LIST}"
done

echo "Boundary files list:"
cat "${BOUNDARY_LIST}"
echo ""

# Create the database
DATABASE_OUTPUT="${REFERENCE_PANEL_DIR}/robustad_mouse_hemato_panel_500kb"

echo "Creating database with boundary files from: ${BOUNDARY_SCORES_DIR}"
echo "Database output: ${DATABASE_OUTPUT}"

"${ROBUSTAD_CMD[@]}" createdatabase \
    --resol "${RESOLUTION}" \
    "${BOUNDARY_LIST}" \
    "${DATABASE_OUTPUT}"

echo ""
echo "=========================================="
echo "✓ Reference panel creation completed!"
echo "=========================================="
echo ""
echo "Reference panel location: ${DATABASE_OUTPUT}"
echo ""
echo "To use this panel with RobusTAD, run:"
echo "  robustad config init"
echo "and select option (2) to load your own panel."
echo "Then provide the path: ${DATABASE_OUTPUT}"
echo ""
