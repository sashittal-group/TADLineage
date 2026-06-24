import check_min_cover

def read_tad_txt_file(txt_file, chrom="chr1"):
    """
    Read TAD data from a text file with format: chrom start end (one TAD per line).
    
    Parameters:
    -----------
    txt_file : str
        Path to the TAD text file
    chrom : str
        Chromosome to extract (e.g., "chr1")
    
    Returns:
    --------
    set of (a, b) tuples representing TADs (as interval endpoints)
    """
    tads = set()
    
    with open(txt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            fields = line.split()
            if len(fields) < 3:
                continue
            
            chr_col = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            
            # Filter by chromosome
            if chr_col != chrom:
                continue
            
            tads.add((start, end))
    
    return tads


def calculate_pairwise_parsimony_costs(tad_sets, names=None, Ct=1.5, Cb=2.0):
    """
    Calculate pairwise parsimony costs between multiple TAD sets.
    Uses the same cost calculation as edges in the phylogenetic tree.
    
    Parameters:
    -----------
    tad_sets : list of sets
        List of TAD sets, each as set of (a, b) tuples
    names : list of str or None
        Labels for the TAD sets (e.g., ["CN", "ES", "NPC"])
    Ct : float
        Cost weight for TAD changes
    Cb : float
        Cost weight for boundary changes
    
    Returns:
    --------
    dict : pairwise costs in format {(i, j): cost_value}
    """
    if names is None:
        names = [f"Set{i}" for i in range(len(tad_sets))]
    
    pairwise_costs = {}
    
    # Calculate all pairwise costs
    for i in range(len(tad_sets)):
        for j in range(i + 1, len(tad_sets)):
            Tu = tad_sets[i]
            Tv = tad_sets[j]
            
            # Calculate TAD Robinson-Foulds distance
            rf_tad = len(Tu - Tv) + len(Tv - Tu)
            
            # Calculate boundary Robinson-Foulds distance
            Su, Eu = check_min_cover.boundaries_from_tads(Tu)
            Sv, Ev = check_min_cover.boundaries_from_tads(Tv)
            
            rf_boundary = (
                len(Su - Sv) + len(Sv - Su)
                + len(Eu - Ev) + len(Ev - Eu)
            )
            
            # Calculate total cost
            cost = Ct * rf_tad + Cb * rf_boundary
            
            pairwise_costs[(i, j)] = {
                "set1": names[i],
                "set2": names[j],
                "tad_rf": rf_tad,
                "boundary_rf": rf_boundary,
                "cost": cost
            }
    
    return pairwise_costs


def print_pairwise_costs(pairwise_costs):
    """
    Pretty print the pairwise parsimony costs.
    
    Parameters:
    -----------
    pairwise_costs : dict
        Output from calculate_pairwise_parsimony_costs
    """
    print("\n" + "="*70)
    print("PAIRWISE PARSIMONY COSTS")
    print("="*70)
    
    for (i, j), info in sorted(pairwise_costs.items()):
        print(f"\n{info['set1']} <-> {info['set2']}:")
        print(f"  TAD RF distance:      {info['tad_rf']}")
        print(f"  Boundary RF distance: {info['boundary_rf']}")
        print(f"  Total Cost:           {info['cost']:.2f}")
    
    print("\n" + "="*70)


# Main execution
if __name__ == "__main__":
    # Read TAD data from three text files
    print("Reading TAD data from text files...")
    chrom="chrX"
    
    cn_tads = read_tad_txt_file("hic_matrix/tads/HiC_CN_DI_50kb.txt", chrom=chrom)
    es_tads = read_tad_txt_file("hic_matrix/tads/HiC_ES_DI_50kb.txt", chrom=chrom)
    npc_tads = read_tad_txt_file("hic_matrix/tads/HiC_NPC_DI_50kb.txt", chrom=chrom)
    
    print(f"  CN:  {len(cn_tads)} TADs")
    print(f"  ES:  {len(es_tads)} TADs")
    print(f"  NPC: {len(npc_tads)} TADs")
    
    # Create list of TAD sets and names
    tad_sets = [cn_tads, es_tads, npc_tads]
    names = ["CN", "ES", "NPC"]
    
    # Set cost parameters
    Ct = 1.5  # TAD change cost
    Cb = 2.0  # Boundary change cost
    
    # Calculate pairwise parsimony costs
    print(f"\nCalculating pairwise parsimony costs for {chrom} (Ct={Ct}, Cb={Cb})...")
    pairwise_costs = calculate_pairwise_parsimony_costs(tad_sets, names, Ct=Ct, Cb=Cb)
    
    # Print results
    print_pairwise_costs(pairwise_costs)
    