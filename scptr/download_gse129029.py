#!/usr/bin/env python3
"""
Download and organize GSE129029 cell differentiation dataset from GEO
Dataset: F1 hybrid embryonic stem cells at different cell stages
Stages: 1-cell (55), 2-cell (94), 4-cell (87), 8-cell (101), 64-cell (171) = 508 total cells

This script uses metadata mapping (GSE129029_metadata.csv) to organize files.
"""

import os
import csv
import gzip
import shutil
from pathlib import Path
import subprocess

# Configuration
BASE_DIR = Path(__file__).parent / "cell_differentiation"
METADATA_FILE = Path(__file__).parent / "GSE129029_metadata.csv"
GEO_ACCESSION = "GSE129029"
TEMP_DIR = Path(__file__).parent / "temp_downloads"

STAGE_COUNTS = {
    "1-cell": 55,
    "2-cell": 94,
    "4-cell": 87,
    "8-cell": 101,
    "64-cell": 171
}

def load_metadata():
    """Load GSM to cell stage mapping from CSV"""
    metadata = {}
    if not METADATA_FILE.exists():
        print(f"ERROR: Metadata file not found: {METADATA_FILE}")
        return metadata
    
    with open(METADATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gsm = row['GSM'].strip()
            stage = row['CELL_STAGE'].strip()
            cell_id = row['CELL_ID'].strip()
            metadata[gsm] = {'stage': stage, 'cell_id': cell_id}
    
    print(f"Loaded metadata for {len(metadata)} GSM samples")
    return metadata

def organize_files(temp_dir, metadata):
    """
    Organize downloaded files into stage-specific folders using metadata
    """
    if not temp_dir.exists():
        print(f"Temp directory {temp_dir} not found")
        print(f"Please download files from GEO and place them in: {temp_dir}")
        return 0
    
    print("\nOrganizing files by cell stage...")
    
    all_files = list(temp_dir.glob("**/*"))
    contact_files = [f for f in all_files if f.is_file() and 
                     (f.suffix in ['.txt', '.gz', '.bed'] or 'contact' in f.name.lower() or 'valid' in f.name.lower())]
    
    if not contact_files:
        print(f"No contact files found in {temp_dir}")
        return 0
    
    print(f"Found {len(contact_files)} potential contact files")
    
    organized_count = 0
    skipped_count = 0
    
    # Process each contact file
    for contact_file in contact_files:
        gsm = extract_gsm(contact_file.name)
        
        if gsm and gsm in metadata:
            stage_info = metadata[gsm]
            stage = stage_info['stage']
            cell_id = stage_info['cell_id']
            
            dest_dir = BASE_DIR / stage
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            output_name = f"{gsm}_{cell_id}"
            if contact_file.suffix == '.gz':
                output_name += '.txt'
                dest_file = dest_dir / output_name
                # Extract gzip file
                try:
                    with gzip.open(contact_file, 'rb') as f_in:
                        with open(dest_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"✓ Extracted: {contact_file.name:40} → {stage}/{output_name}")
                    organized_count += 1
                except Exception as e:
                    print(f"✗ Error extracting {contact_file.name}: {e}")
                    skipped_count += 1
            else:
                output_name += contact_file.suffix
                dest_file = dest_dir / output_name
                try:
                    shutil.copy2(contact_file, dest_file)
                    print(f"✓ Copied:    {contact_file.name:40} → {stage}/{output_name}")
                    organized_count += 1
                except Exception as e:
                    print(f"✗ Error copying {contact_file.name}: {e}")
                    skipped_count += 1
        else:
            if gsm:
                print(f"⚠ Skipped:   {contact_file.name:40} (GSM not in metadata)")
            skipped_count += 1
    
    return organized_count

def extract_gsm(filename):
    """Extract GSM accession from filename"""
    import re
    match = re.search(r'(GSM\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def print_summary():
    """Print summary of organized data"""
    print("\n" + "="*70)
    print("CELL DIFFERENTIATION DATASET SUMMARY")
    print("="*70)
    print(f"\nDataset: GSE129029 (GEO accession)")
    print(f"Source: F1 hybrid embryonic stem cells (Mus musculus)")
    print(f"Total expected cells: 508")
    print(f"Paper reference: Collombet et al., 2020")
    print(f"\nCell stage distribution:")
    
    total_found = 0
    for stage in ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]:
        stage_dir = BASE_DIR / stage
        if stage_dir.exists():
            existing = len(list(stage_dir.glob("*")))
            total_found += existing
            print(f"  {stage:10} - Expected: {STAGE_COUNTS[stage]:3} cells, Found: {existing:3} files")
        else:
            print(f"  {stage:10} - Expected: {STAGE_COUNTS[stage]:3} cells, Found:   0 files")
    
    print(f"\n  TOTAL: {total_found}/508 cells organized")
    print(f"\nData location: {BASE_DIR}")
    print("="*70)

def download_instructions():
    """Print download instructions"""
    print("\n" + "="*70)
    print("DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\n1. Visit: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129029")
    print("\n2. Scroll to 'Supplementary file' section")
    print("\n3. Download ALL files listed (they contain contact maps for each cell)")
    print(f"\n4. Extract and place them in: {TEMP_DIR}/")
    print("\n5. Run this script again to organize them by cell stage")
    print("\nNote: Files should contain 'valid_interactions' or contact data")
    print("Each GSM accession maps to one cell")
    print("="*70 + "\n")

def main():
    print(f"\n{'='*70}")
    print(f"GSE129029 Data Organizer")
    print(f"{'='*70}\n")
    
    # Ensure base directories exist
    for stage in STAGE_COUNTS.keys():
        stage_dir = BASE_DIR / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = load_metadata()
    if not metadata:
        print("ERROR: Could not load metadata. Exiting.")
        return
    
    # Check if temp directory has files
    if not TEMP_DIR.exists() or not list(TEMP_DIR.glob("*")):
        download_instructions()
        return
    
    # Organize files
    organized = organize_files(TEMP_DIR, metadata)
    
    # Print summary
    print_summary()
    
    if organized > 0:
        print(f"\n✓ Successfully organized {organized} files!")
    else:
        download_instructions()

if __name__ == "__main__":
    main()
