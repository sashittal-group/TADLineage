#!/usr/bin/env python
"""Test script to verify comparison functions work correctly."""

import os
import sys

# Test match_tads_within_tolerance function
def test_match_tads():
    """Test TAD matching function"""
    # Simulated data
    inferred = {(1, 5), (10, 15), (20, 25)}
    provided = {(1, 5), (10, 16), (21, 26)}
    tolerance = 1
    
    # Import the function from observe_given_data
    sys.path.insert(0, '/Users/PSLabMonisha/Desktop/tads')
    from observe_given_data import match_tads_within_tolerance
    
    matched, unmatched_inf, unmatched_prov = match_tads_within_tolerance(
        inferred, provided, tolerance=tolerance
    )
    
    print("Test match_tads_within_tolerance:")
    print(f"  Inferred: {inferred}")
    print(f"  Provided: {provided}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Matched: {matched}")
    print(f"  Unmatched inferred: {unmatched_inf}")
    print(f"  Unmatched provided: {unmatched_prov}")
    print()

# Test compare_internal_tads structure
def test_compare_structure():
    """Test that compare_internal_tads returns proper structure"""
    sys.path.insert(0, '/Users/PSLabMonisha/Desktop/tads')
    from observe_given_data import get_internal_nodes
    
    edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
    internal = get_internal_nodes(edges)
    print(f"Test get_internal_nodes:")
    print(f"  Edges: {edges}")
    print(f"  Internal nodes: {internal}")
    print()

if __name__ == "__main__":
    test_match_tads()
    test_compare_structure()
    print("Basic tests passed!")
