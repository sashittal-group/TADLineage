#!/usr/bin/env python3
"""
Generate Hi-C matrices from contact pairs and run OnTAD for TAD detection.
Processes GSE129029 contact files at multiple resolutions.
Focuses on intra-chromosomal (cis) contacts only.
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

# Configuration
BASE_DIR = Path("/Users/PSLabMonisha/Desktop/tads/scptr/cell_differentiation")
ONTAD_BIN = Path("/Users/PSLabMonisha/Desktop/tads/OnTAD/OnTAD/src/OnTAD")
RESOLUTIONS_KB = [1000, 500, 250, 100]  # 1MB, 500kb, 250kb, 100kb
RESOLUTIONS_BP = [r * 1000 for r in RESOLUTIONS_KB]
CHROMOSOMES = list(range(1, 20)) + ['X']  # 1-19, X (not Y, M)
PENALTY = 0.1

# Metadata
METADATA_FILE = Path("/Users/PSLabMonisha/Desktop/tads/scptr/GSE129029_metadata.csv")


def parse_contact_file(contact_file):
    """
    Parse JUICE format contact file.
    Extracts: chr1 pos1 strand1 chr2 pos2 strand2
    """
    try:
        df = pd.read_csv(
            contact_file, 
            sep='\t', 
            header=None,
            dtype={1: str, 2: int, 4: str, 5: int},
            usecols=[1, 2, 4, 5]
        )
        
        df.columns = ['chr1', 'pos1', 'chr2', 'pos2']
        
        # Strip 'chr' prefix and filter to valid chromosomes
        df['chr1'] = df['chr1'].str.replace('chr', '', regex=False)
        df['chr2'] = df['chr2'].str.replace('chr', '', regex=False)
        
        valid_chroms = set([str(i) for i in range(1, 20)] + ['X'])
        df = df[(df['chr1'].isin(valid_chroms)) & (df['chr2'].isin(valid_chroms))]
        
        return df if len(df) > 0 else None
    except Exception as e:
        print(f"  Error reading {contact_file}: {e}")
        return None


def bin_contacts(contacts, chr_name, resolution_bp):
    """
    Bin contacts for a single chromosome into resolution-sized bins.
    """
    # Filter for this chromosome (intra-chromosomal only)
    cis_contacts = contacts[(contacts['chr1'] == chr_name) & (contacts['chr2'] == chr_name)].copy()
    
    if len(cis_contacts) == 0:
        return None
    
    # Bin positions
    cis_contacts['bin1'] = cis_contacts['pos1'] // resolution_bp
    cis_contacts['bin2'] = cis_contacts['pos2'] // resolution_bp
    
    # Aggregate by bin pair
    binned = cis_contacts.groupby(['bin1', 'bin2']).size().reset_index(name='count')
    
    return binned


def create_dense_matrix(binned_data, num_bins):
    """
    Create symmetric dense matrix from sparse bin counts.
    """
    matrix = np.zeros((num_bins, num_bins), dtype=np.float32)
    
    for _, row in binned_data.iterrows():
        i, j = int(row['bin1']), int(row['bin2'])
        
        if i < num_bins and j < num_bins:
            count = row['count']
            matrix[i, j] += count
            if i != j:
                matrix[j, i] += count
    
    return matrix


def save_matrix_txt(matrix, filepath):
    """Save matrix as space-separated plaintext for OnTAD."""
    np.savetxt(filepath, matrix, fmt='%d', delimiter=' ')


def run_ontad(matrix_file, output_prefix):
    """
    Run OnTAD on a matrix file, then clean up intermediate files.
    """
    try:
        # Clean matrix (ensure proper formatting)
        cleaned_file = str(matrix_file) + '.cleaned'
        with open(matrix_file, 'r') as f_in:
            with open(cleaned_file, 'w') as f_out:
                for line in f_in:
                    stripped = ' '.join(line.split())
                    if stripped:
                        f_out.write(stripped + '\n')
        
        # Run OnTAD
        cmd = [str(ONTAD_BIN), cleaned_file, '-o', output_prefix, '-penalty', str(PENALTY)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"    OnTAD failed: {result.stderr[:200]}")
            return False
        
        # Cleanup intermediate files
        try:
            os.remove(matrix_file)
            os.remove(cleaned_file)
        except:
            pass
        
        return True
    except Exception as e:
        print(f"    Error running OnTAD: {e}")
        return False


def process_cell(txt_filename, stage):
    """
    Process a single cell: generate matrices for all chromosomes and resolutions, run OnTAD.
    """
    contact_file = BASE_DIR / stage / txt_filename
    
    if not contact_file.exists():
        print(f"  File not found: {contact_file}")
        return False
    
    # Extract GSM ID from filename (GSM*.txt pattern)
    gsm_id = txt_filename.split('_')[0]
    print(f"  Processing {gsm_id}...")
    
    # Parse all contacts
    contacts = parse_contact_file(contact_file)
    if contacts is None:
        print(f"    No valid contacts in {txt_filename}")
        return False
    
    print(f"    Total contacts: {len(contacts)}")
    
    # Process each resolution
    for res_kb, res_bp in zip(RESOLUTIONS_KB, RESOLUTIONS_BP):
        print(f"    Resolution {res_kb}kb...")
        
        # Process each chromosome
        for chr_name in CHROMOSOMES:
            chr_str = str(chr_name)
            
            # Bin the contacts
            binned = bin_contacts(contacts, chr_str, res_bp)
            
            if binned is None or len(binned) == 0:
                continue
            
            # Determine matrix size
            max_bin = int(max(binned['bin1'].max(), binned['bin2'].max())) + 1
            
            # Create output directory
            out_dir = BASE_DIR / stage / gsm_id / f"chr{chr_name}" / f"{res_kb}kb"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Create and save matrix
            matrix = create_dense_matrix(binned, max_bin)
            matrix_file = out_dir / "hic_matrix.txt"
            save_matrix_txt(matrix, str(matrix_file))
            
            # Run OnTAD
            tad_prefix = str(out_dir / "tads")
            if run_ontad(matrix_file, tad_prefix):
                # Check if output was created
                tad_file = Path(f"{tad_prefix}.tad")
                if tad_file.exists():
                    print(f"      chr{chr_name}: TADs created")
                else:
                    print(f"      chr{chr_name}: Warning - no TAD file generated")
            else:
                print(f"      chr{chr_name}: OnTAD failed")
    
    print(f"  Completed {gsm_id}")
    return True


def main():
    """Main orchestration function."""
    # Verify dependencies
    if not ONTAD_BIN.exists():
        print(f"Error: OnTAD not found at {ONTAD_BIN}")
        sys.exit(1)
    
    if not BASE_DIR.exists():
        print(f"Error: Base directory not found at {BASE_DIR}")
        sys.exit(1)
    
    # Load metadata
    if METADATA_FILE.exists():
        metadata = pd.read_csv(METADATA_FILE)
        print(f"Loaded metadata for {len(metadata)} cells")
    else:
        print(f"Warning: Metadata file not found, will process all .txt files found")
        metadata = None
    
    # Process each stage
    stages = ['1-cell', '2-cell', '4-cell', '8-cell', '64-cell']
    total_cells = 0
    
    for stage in stages:
        stage_dir = BASE_DIR / stage
        
        if not stage_dir.exists():
            print(f"Stage directory not found: {stage_dir}")
            continue
        
        txt_files = list(stage_dir.glob("*.txt"))
        print(f"\n=== {stage}: {len(txt_files)} files ===")
        
        for txt_file in sorted(txt_files):
            try:
                process_cell(txt_file.name, stage)
                total_cells += 1
            except Exception as e:
                print(f"  ERROR processing {txt_file.name}: {e}")
    
    print(f"\n=== Complete: {total_cells} cells processed ===")


if __name__ == "__main__":
    main()
