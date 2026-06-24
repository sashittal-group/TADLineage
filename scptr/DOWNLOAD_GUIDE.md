# GSE129029 Data Download Guide

## Quick Links
- **GEO Dataset**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129029
- **FTP Server**: ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE129nnn/GSE129029/suppl/
- **File List**: https://www.ncbi.nlm.nih.gov/geo/series/GSE129nnn/GSE129029/suppl/filelist.txt

## Dataset Overview
- **Total Files**: 508 single-cell contact maps
- **Format**: gzip-compressed plaintext (`*.allValidPairs.txt.gz`)
- **Total Size**: ~8 GB (compressed)
- **Individual File Sizes**: 200 KB - 10 MB each

## Available Files
All 508 files are directly accessible from NCBI GEO FTP:

### 1-Cell Stage (55 files)
```
GSM3690920_1CSE_10.allValidPairs.txt.gz
GSM3690922_1CSE_101.allValidPairs.txt.gz
GSM3690923_1CSE_102.allValidPairs.txt.gz
... (52 more files)
```

### 2-Cell Stage (94 files)
```
GSM3691018_2CSE_1.allValidPairs.txt.gz
GSM3691019_2CSE_10.allValidPairs.txt.gz
... (92 more files)
```

### 4-Cell Stage (87 files)
### 8-Cell Stage (101 files)
### 64-Cell Stage (171 files)

## Download Methods

### Option 1: Bulk Download (Recommended)
Download the tar archive with all files:
```bash
cd /Users/PSLabMonisha/Desktop/tads/scptr/temp_downloads
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE129nnn/GSE129029/suppl/GSE129029_RAW.tar
tar -xf GSE129029_RAW.tar
```

**Expected time**: 30-60 minutes (depending on connection)
**Storage needed**: ~16 GB

### Option 2: Selective Download with wget
Create a file `urls.txt` with the desired URLs and download:
```bash
cd /Users/PSLabMonisha/Desktop/tads/scptr/temp_downloads

# Download using wget
wget -i urls.txt

# Or download with retry and resume support
wget -c -i urls.txt
```

### Option 3: Using Python Script
```bash
cd /Users/PSLabMonisha/Desktop/tads/scptr
python3 get_files_from_geo.py
```

### Option 4: Manual Download
1. Visit: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129029
2. Scroll to "Supplementary file" section
3. Click "Open" or "Download" link
4. This will show all 508 files
5. Download the files you need (or use your browser's batch download feature)
6. Extract `.gz` files using: `gunzip *.gz`

## File Naming Pattern
```
{GSM_ACCESSION}_{CELL_ID}.allValidPairs.txt.gz
```

Examples:
- `GSM3690920_1CSE_10.allValidPairs.txt.gz` = Cell 10 at 1-cell stage
- `GSM3691019_2CSE_10.allValidPairs.txt.gz` = Cell 10 at 2-cell stage
- `GSM3691148_4CSE_100.allValidPairs.txt.gz` = Cell 100 at 4-cell stage

## File Organization After Download
Once files are downloaded to `temp_downloads/`, run:
```bash
cd /Users/PSLabMonisha/Desktop/tads/scptr
python3 download_gse129029.py
```

This will automatically:
1. Read metadata from `GSE129029_metadata.csv`
2. Parse filenames to extract GSM accessions
3. Move files into stage-specific folders:
   - `cell_differentiation/1-cell/`
   - `cell_differentiation/2-cell/`
   - `cell_differentiation/4-cell/`
   - `cell_differentiation/8-cell/`
   - `cell_differentiation/64-cell/`

## File Format
Each contact file contains plaintext data:
```
fragment_id_1  fragment_id_2  contact_count
1              2              1
1              5              2
2              10             1
...
```

Where:
- Columns are tab-separated
- Fragment IDs refer to GATC restriction sites in mm9 genome
- Contact count is the integer number of observed interactions
- Each row represents one contact

## Troubleshooting

### Downloads are slow
- Use aria2c for parallel downloads: `aria2c -x8 -i urls.txt`
- Download during off-peak hours
- Consider downloading the tar archive instead of individual files

### FTP connection fails
- Try HTTP mirror: Check GEO website for alternative URLs
- Use a download manager (JDownloader, Transmission, etc.)
-  Add passive mode to wget: `wget --ftp-user=anonymous -p`

### Disk space issues
- Download files in batches (by cell stage)
- Delete `.gz` files after extraction to save space
- Use external storage if needed

### Files not found after organization
- Verify metadata CSV exists: `GSE129029_metadata.csv`
-Check that filenames match expected pattern
- Run: `ls -la temp_downloads/ | head -20`

## Verification
After organizing files, verify the structure:
```bash
cd /Users/PSLabMonisha/Desktop/tads/scptr/cell_differentiation

# Count files per stage
echo "1-cell: $(ls 1-cell/*.txt 2>/dev/null | wc -l)"
echo "2-cell: $(ls 2-cell/*.txt 2>/dev/null | wc -l)"
echo "4-cell: $(ls 4-cell/*.txt 2>/dev/null | wc -l)"
echo "8-cell: $(ls 8-cell/*.txt 2>/dev/null | wc -l)"
echo "64-cell: $(ls 64-cell/*.txt 2>/dev/null | wc -l)"

# Sample data from first file
gunzip -c 1-cell/*_1CSE_10.txt.gz | head -5
```

## Tips
- Download during off-peak times (evening/weekend) for faster speeds
- Use `wget -c` to resume interrupted downloads
- Consider downloading by cell stage to manage disk space
- Keep the original `.gz` files for archival purposes

## Command Line Quick Start
```bash
# Go to download directory
cd /Users/PSLabMonisha/Desktop/tads/scptr/temp_downloads

# Method 1: Download all files (large, ~8 GB)
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE129nnn/GSE129029/suppl/GSE129029_RAW.tar
tar -xf GSE129029_RAW.tar

# Method 2: Download selected files using metadata
cd ..
python3 get_files_from_geo.py

# Method 3: Download individual file example
cd temp_downloads
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE129nnn/GSE129029/suppl/GSM3690920_1CSE_10.allValidPairs.txt.gz

# After downloading, organize files
cd ..
python3 download_gse129029.py

# Verify
ls cell_differentiation/*/
```

## Support
- **GEO Support**: https://www.ncbi.nlm.nih.gov/geo/info/faq.html
- **Paper**: Collombet et al., Cell 2020
- **Questions**: See paper supplementary information

