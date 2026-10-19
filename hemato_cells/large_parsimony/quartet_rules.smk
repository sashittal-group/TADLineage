"""
Rules shared by the large parsimony analyses, included by each analysis folder's
Snakefile. That Snakefile sets DATA_DIR and WEIGHTING and defines rule all itself
(a rule from an included file is never the default target), and is run from its folder.

For every chromosome / resolution / leaf pair / tolerance (0-5), give the large
parsimony ILP the four observed TAD hierarchies LT, CLP, leaf1 (MEP line) and
leaf2 (GMP line) and no tree, and check whether it infers {LT, CLP} | {leaf1, leaf2}
-- the unrooted form of the tree LT -> MPP -> {CLP, CMP -> {leaf1, leaf2}} that the
small parsimony pipelines are given. Tolerance is a free parameter: each
chromosome x resolution x leaf pair counts once, at its best tolerance.

Outputs (in the analysis folder):
    results/jsons/quartet_chr{chr}_{res}_{l1}_{l2}_{tol}.json   one per instance
    results/inferred/{MPP,CMP}/chr{chr}/{res}/inferred_{l1}_{l2}_{tol}.tad
                                                               (only when the tree is correct)
    results/best_tolerance/best_tolerance_summary.txt          the stat, per chromosome and
                                                               genome-wide (exact sum of
                                                               per-chromosome costs)

Rules are written out explicitly, not generated in a loop (see the late-binding
gotcha in the repo README.md).
"""

PYTHON = "/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python"   # gurobipy + pandas
SCRIPTS = ".."          # shared scripts live in hemato_cells/large_parsimony

LINE1 = ["MEP", "MKP", "MK"]
LINE2 = ["GMP", "G"]
CHRS = [str(i) for i in range(1, 20)]
N_CHROM = len(CHRS)
RESOLUTIONS = ["1mb", "2mb", "100kb", "250kb", "500kb"]
TOLS = [0, 1, 2, 3, 4, 5]   # tolerance is a free parameter: see rule best_tolerance

# Objective. The paper uses C_t = C_b = 1 with the convex objective; the importance
# terms of the weighted analyses are implemented for convex only.
MODE = "convex"
CT = 1.0
CB = 1.0

GUROBI_THREADS = 2      # Gurobi's progress goes to results/logs/<instance>.log (--solver-log),
                        # so a slow job can be inspected while it runs
TIME_LIMIT = 0          # seconds per Gurobi call; 0 = no limit. A timed-out instance
                        # is recorded with its status and left out of the counts.

wildcard_constraints:
    chr=r"\d+",
    res=r"\d+(kb|mb)",
    l1="|".join(LINE1),
    l2="|".join(LINE2),
    tol=r"\d+",


rule infer_quartet:
    input:
        lt=DATA_DIR + "/LT/chr{chr}/{res}/tads.tad",
        clp=DATA_DIR + "/CLP/chr{chr}/{res}/tads.tad",
        leaf1=DATA_DIR + "/{l1}/chr{chr}/{res}/tads.tad",
        leaf2=DATA_DIR + "/{l2}/chr{chr}/{res}/tads.tad",
    output:
        "results/jsons/quartet_chr{chr}_{res}_{l1}_{l2}_{tol}.json",
    log:
        "results/logs/quartet_chr{chr}_{res}_{l1}_{l2}_{tol}.log",
    threads: GUROBI_THREADS
    shell:
        "{PYTHON} {SCRIPTS}/infer_quartet_tree.py --data-dir {DATA_DIR} --weighting {WEIGHTING} "
        "--chrom {wildcards.chr} --resolution {wildcards.res} "
        "--leaf1 {wildcards.l1} --leaf2 {wildcards.l2} --tolerance {wildcards.tol} "
        "--mode {MODE} --Ct {CT} --Cb {CB} "
        "--threads {threads} --time-limit {TIME_LIMIT} --solver-log "
        "--inferred-dir results/inferred --output {output} > {log} 2>&1"


# Tolerance as a free parameter: each chromosome x resolution x leaf pair (and each
# genome-wide resolution x leaf pair) counts once, correct if its best tolerance
# gives the correct quartet. Reports the matching chance level, which is above 1/3.
rule best_tolerance:
    input:
        expand(
            "results/jsons/quartet_chr{chr}_{res}_{l1}_{l2}_{tol}.json",
            chr=CHRS, res=RESOLUTIONS, l1=LINE1, l2=LINE2, tol=TOLS,
        ),
    output:
        "results/best_tolerance/best_tolerance_summary.txt",
        "results/best_tolerance/best_tolerance_per_instance.csv",
        "results/best_tolerance/best_tolerance_genomewide_per_instance.csv",
    log:
        "results/logs/best_tolerance.log",
    shell:
        "{PYTHON} {SCRIPTS}/aggregate_best_tolerance.py --json-dir results/jsons "
        "--outdir results/best_tolerance --n-chrom {N_CHROM} > {log} 2>&1"
