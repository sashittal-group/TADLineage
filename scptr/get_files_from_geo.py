#!/usr/bin/env python3
"""
Download GSE129029 files directly from NCBI GEO FTP
Downloads all 508 contact files for 5 cell stages
"""

import subprocess
import csv
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent
METADATA_FILE = BASE_DIR / "GSE129029_metadata.csv"
TEMP_DIR = BASE_DIR / "temp_downloads"
TEMP_DIR.mkdir(exist_ok=True)

FTP_BASE = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE129nnn/GSE129029/suppl"

def load_metadata():
    """Load GSM to cell mapping"""
    metadata = {}
    if not METADATA_FILE.exists():
        print(f"ERROR: {METADATA_FILE} not found")
        return metadata
    
    with open(METADATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gsm = row['GSM'].strip()
            cell_id = row['CELL_ID'].strip()
            # Keep only first occurrence of each GSM
            if gsm not in metadata:
                metadata[gsm] = cell_id
    
    return metadata

def download_files(metadata, max_parallel=8):
    """
    Download all files using parallel wget
    """
    print(f"\n{'='*70}")
    print("DOWNLOADING GSE129029 FILES")
    print(f"{'='*70}\n")
    
    metadata_list = list(metadata.items())
    total = len(metadata_list)
    
    print(f"Total files to download: {total}")
    print(f"Using {max_parallel} parallel connections\n")
    
    # Create URL list file for parallel download
    url_list_file = TEMP_DIR / "download_urls.txt"
    with open(url_list_file, 'w') as f:
        for gsm, cell_id in metadata_list:
            filename = f"{gsm}_{cell_id}.allValidPairs.txt.gz"
            url = f"{FTP_BASE}/{filename}"
            f.write(f"{url}\n")
    
    print(f"Created URL list: {url_list_file}\n")
    
    # Use aria2c for parallel downloads if available, otherwise wget
    try:
        # Try aria2c first (better for parallel)
        cmd = [
            "aria2c",
            "-x", str(max_parallel),  # max parallel connections
            "-j", str(max_parallel),  # max parallel downloads
            "-d", str(TEMP_DIR),
            "-i", str(url_list_file),
            "--continue=true",
            "--auto-file-renaming=false"
        ]
        print(f"Using aria2c for parallel downloads...\n")
        result = subprocess.run(cmd, capture_output=False)
        
    except FileNotFoundError:
        print("aria2c not found, using wget instead...\n")
        # Fall back to wget
        cmd = [
            "wget",
            "-q",
            "-i", str(url_list_file),
            "-P", str(TEMP_DIR),
            "--continue",
            "--no-verbose"
        ]
        result = subprocess.run(cmd, capture_output=False)
    
    # Check results
    downloaded_files = list(TEMP_DIR.glob("*.gz"))
    print(f"\n{'='*70}")
    print(f"Download complete!")
    print(f"Files downloaded: {len(downloaded_files)}/{total}")
    print(f"Location: {TEMP_DIR}")
    print(f"{'='*70}\n")
    
    if len(downloaded_files) > 0:
        print("\nNext step: Run 'python3 download_gse129029.py' to organize files")
    
    return len(downloaded_files)

def main():
    print("\nGSE129029 File Downloader")
    print("="*70)
    
    # Load metadata
    metadata = load_metadata()
    if not metadata:
        print("ERROR: Could not load metadata")
        return
    
    print(f"Loaded metadata for {len(metadata)} samples\n")
    
    # Download files
    downloaded = download_files(metadata)
    
    if downloaded == 0:
        print("\nWARNING: No files were downloaded")
        print("Please ensure internet connection and FTP access to NCBI GEO")

if __name__ == "__main__":
    main()
