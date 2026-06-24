# GSE129029 TAD Generation Pipeline - Progress & Summary

## Status
**Started:** April 17, 2026 11:15 PM  
**Total Cells:** 508  
**Expected Processing Time:** ~48-72 hours (sequential processing)

## Current Progress
- **TAD files generated:** Check with `bash monitor_progress.sh`
- **Cells completed:** Track by counting GSM_* subdirectories with all resolutions

## Pipeline Configuration
- **Resolutions:** 1MB (1000kb), 500kb, 250kb, 100kb
- **Chromosomes:** 1-19, X (not Y, M)
- **TAD Detection Tool:** OnTAD (penalty=0.1)
- **Memory Strategy:** Sequential cell processing + immediate cleanup of intermediate matrices

## Output Structure
```
cell_differentiation/
├── 1-cell/
│   ├── GSM3690920/
│   │   ├── chr1/
│   │   │   ├── 1000kb/tads.tad
│   │   │   ├── 500kb/tads.tad
│   │   │   ├── 250kb/tads.tad
│   │   │   └── 100kb/tads.tad
│   │   ├── chr2/
│   │   │   └── ... (same for each resolution)
│   │   └── ... (chrX)
│   └── ... (other GSM IDs)
├── 2-cell/ (94 cells)
├── 4-cell/ (87 cells)
├── 8-cell/ (101 cells)
└── 64-cell/ (171 cells)
```

## Key Files

### Running the pipeline
```bash
# Start full pipeline (runs in background)
nohup conda run -n hic_analysis python generate_matrices_and_tads.py > tad_generation.log 2>&1 &

# Monitor progress
bash monitor_progress.sh

# Check log
tail -50 tad_generation.log
```

### Testing
```bash
# Test with single cell (small sample)
conda run -n hic_analysis python test_single_cell.py
```

## Expected Output Per Cell
- **Total TAD files per cell:** 4 resolutions × 20 chromosomes = 80 files
- **Total TAD files for entire dataset:** 508 cells × 80 = 40,640 TAD files
- **Estimated disk space:** ~500 MB - 1 GB (depends on TAD count per file)

## Cleanup
Intermediate Hi-C matrices are automatically deleted after OnTAD completes to prevent memory bloat. No manual cleanup needed.

## Next Steps (After Pipeline Completes)
1. Validate TAD calls across cells
2. Compare resolutions and identify robust domains
3. Analyze TAD evolution across cell stages
4. Generate visualization of TAD landscape

## Troubleshooting

### If process dies/gets interrupted
```bash
# Check if still running
ps aux | grep generate_matrices_and_tads

# Resume would require tracking which cells completed (check directory structure)
# Currently no checkpoint/resume mechanism in place
```

### If matrix parsing fails
- Verify contact files are plaintext (not gzipped)
- Check file format has correct columns (chr, pos, chr, pos in columns 1,2,4,5)

### Memory issues
- Pipeline processes one cell sequentially to manage memory
- If issues occur, reduce RESOLUTIONS_KB or CHROMOSOMES in script
