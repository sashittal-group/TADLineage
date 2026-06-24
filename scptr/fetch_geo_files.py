#!/usr/bin/env python3
"""
Download GSE129029 supplementary files from NCBI GEO
Uses direct HTTP/FTP access to GEO's file servers
"""

import urllib.request
import urllib.error
import os
import gzip
import shutil
from pathlib import Path
import time
import json

BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp_downloads"
TEMP_DIR.mkdir(exist_ok=True)

# GEO file manifest - these are the actual supplementary files
# Format: (filename, GSM, stage)
GEO_FILES = {
    # This is a sample mapping - actual files need to be identified from GEO
    # Files are typically in format: GSM{number}_RAW.txt or GSM{number}_file.gz
    "1CSE": {
        "url_pattern": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{gsm}/raw/",
        "count": 55,
        "gsm_list": list(range(3690920, 3691016))
    },
    "2CSE": {
        "url_pattern": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{gsm}/raw/",
        "count": 94,
        "gsm_list": list(range(3691018, 3691146))
    },
    "4CSE": {
        "url_pattern": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{gsm}/raw/",
        "count": 87,
        "gsm_list": list(range(3691148, 3691291))
    },
    "8CSE": {
        "url_pattern": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{gsm}/raw/",
        "count": 101,
        "gsm_list": list(range(3691534, 3691669))
    },
    "64CSE": {
        "url_pattern": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{gsm}/raw/",
        "count": 171,
        "gsm_list": list(range(3691292, 3691533))
    }
}

def download_file(url, output_path, retry=3):
    """Download a file with retry logic"""
    for attempt in range(retry):
        try:
            print(f"Downloading: {url}")
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Saved to: {output_path}")
            return True
        except urllib.error.URLError as e:
            print(f"✗ Attempt {attempt + 1}/{retry} failed: {e}")
            if attempt < retry - 1:
                time.sleep(2)
    return False

def fetch_file_list_from_geo():
    """
    Fetch the list of available files from GEO
    This requires parsing the GEO website or FTP server
    """
    print("\nFetching file list from GEO...")
    print("NOTE: This requires direct access to GEO's FTP servers")
    
    geo_base = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE129nnn/GSE129029/suppl/"
    
    try:
        print(f"Checking: {geo_base}")
        # Try to list files - this works with some FTP servers
        response = urllib.request.urlopen(geo_base, timeout=5)
        files = response.read().decode('utf-8')
        print("Available files:")
        print(files[:500])  # Print first 500 chars
        return True
    except Exception as e:
        print(f"Could not access FTP directly: {e}")
        return False

def create_download_list():
    """Create a list of files to download based on the paper"""
    print("\n" + "="*70)
    print("GSE129029 FILE DOWNLOAD HELPER")
    print("="*70)
    
    print("\nThe paper references '508 single cells' with contact maps:")
    print("  - 55 cells at 1-cell stage")
    print("  - 94 cells at 2-cell stage")
    print("  - 87 cells at 4-cell stage")
    print("  - 101 cells at 8-cell stage")
    print("  - 171 cells at 64-cell stage")
    
    print("\nFiles are located in GEO supplementary files section:")
    print("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129029")
    
    print("\nExpected file types:")
    print("  - plaintext contact files (format: fragment_id1 fragment_id2 count)")
    print("  - potentially gzip compressed (.gz)")
    print("  - may have names like: GSM{number}_{cell_id}_*")

def main():
    print("\n" + "="*70)
    print("GSE129029 Auto-Downloader")
    print("="*70 + "\n")
    
    # Try to fetch from GEO
    success = fetch_file_list_from_geo()
    
    if not success:
        create_download_list()
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*70)
        print("\nPlease download files manually:")
        print("\n1. Visit: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129029")
        print("\n2. Click on 'Supplementary file' section")
        print("\n3. Look for files containing contact/interaction data")
        print("\n4. Download the files (may be in individual .gz or tar files)")
        print(f"\n5. Extract to: {TEMP_DIR}/")
        print("\n6. Files should have naming pattern:")
        print("   GSM{number}_{cell_id}.txt or similar")
        print("\n7. Once files are in place, run:")
        print("   python3 download_gse129029.py")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
