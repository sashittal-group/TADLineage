import os
import re
import argparse
import sys
sys.path.append("..")  # Add parent directory to path for imports
from tad_utilities import read_tad_intervals, parse_interval, match_tads_within_tolerance, relaxed_boundary_rf, relaxed_tad_rf
from nussinov import infer_tads_with_dp
from filtering import harmonize_leaf_boundaries
from observe_given_data import fitch_hartigan_tads, compute_parsimony_score, is_laminar_set


def load_tads(path):
    """Read .tad file and return set of (start,end)."""
    tads = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                a, b = map(int, parts[:2])
            except ValueError:
                continue
            tads.add((a,b))
    return tads


def convert_to_strings(tads):
    """Convert (a,b) → '[a,b]' format expected by DP."""
    return [f"[{a},{b}]" for (a,b) in tads]


def collect_boundaries(tads):
    starts=set()
    ends=set()
    for a,b in tads:
        starts.add(a)
        ends.add(b)
    return starts,ends

def compute_leaf_distance(tadsA, tadsB, tolerance):

    # interval RF
    tad_dist = relaxed_tad_rf(tadsA, tadsB, tolerance)

    # boundary RF
    sA, eA = collect_boundaries(tadsA)
    sB, eB = collect_boundaries(tadsB)

    boundary_dist = relaxed_boundary_rf(sA, sB, eA, eB, tolerance)

    return tad_dist + boundary_dist
def compute_root_median(tadsA,tadsB):

    edges=[(0,1),(0,2)]

    leaf_intervals={
        1:convert_to_strings(tadsA),
        2:convert_to_strings(tadsB)
    }

    sA,eA=collect_boundaries(tadsA)
    sB,eB=collect_boundaries(tadsB)

    start_boundaries=sA|sB|eA|eB
    end_boundaries=sA|sB|eA|eB

    root_tads,score = infer_tads_with_dp(
        edges,
        leaf_intervals,
        start_boundaries,
        end_boundaries,
        Ct=1.0,
        Cb=1.0
    )

    return root_tads,score


def compute_metrics(pred,ref,tolerance):

    matched,unmatched_pred,unmatched_ref = match_tads_within_tolerance(
        pred,ref,tolerance
    )

    precision=len(matched)/len(pred) if pred else 0
    recall=len(matched)/len(ref) if ref else 0

    if precision+recall==0:
        f1=0
    else:
        f1=2*precision*recall/(precision+recall)

    return precision,recall,f1,matched

def novel_root_tads(root, ref, leafA, leafB, tolerance):
    """
    Root TADs that match the reference but do not match
    any TAD in the leaves (within tolerance).
    """

    # root vs reference
    matched_ref, _, _ = match_tads_within_tolerance(root, ref, tolerance)
    root_ref_matches = {pred for pred, _ in matched_ref}

    # root vs leaf A
    matched_A, _, _ = match_tads_within_tolerance(root, leafA, tolerance)
    root_A_matches = {pred for pred, _ in matched_A}

    # root vs leaf B
    matched_B, _, _ = match_tads_within_tolerance(root, leafB, tolerance)
    root_B_matches = {pred for pred, _ in matched_B}

    leaf_matches = root_A_matches | root_B_matches

    novel = root_ref_matches - leaf_matches

    return novel



def check_fitch_laminarity(fitch_assignments, tolerance=0):
    """
    Check whether the TAD sets assigned by Fitch are laminar
    at every node of the tree.
    """

    all_laminar = True

    for node, intervals in fitch_assignments.items():

        interval_list = list(intervals)

        if not is_laminar_set(interval_list, tolerance=tolerance):
            print(f"Node {node} is NOT laminar")
            print(f"Intervals: {sorted(interval_list)}")
            all_laminar = False
        else:
            print(f"Node {node} is laminar ({len(interval_list)} intervals)")

    return all_laminar
def compute_fitch_parsimony(tadsA, tadsB, tolerance, Ct=1.0, Cb=1.0):

    edges = [(0,1),(0,2)]

    leaf_intervals = {
        1: convert_to_strings(tadsA),
        2: convert_to_strings(tadsB)
    }

    fitch_assignments = fitch_hartigan_tads(edges, leaf_intervals)
    
    # laminar = check_fitch_laminarity(fitch_assignments, tolerance=tolerance)

    # if laminar:
    #     print("All Fitch assignments are laminar")
    # else:
    #     print("WARNING: Fitch assignments contain non-laminar intervals")

    fitch_score, fitch_per_edge_scores = compute_parsimony_score(
        edges,
        fitch_assignments,
        Ct=Ct,
        Cb=Cb,
        tolerance=tolerance
    )

    return fitch_score


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--tadA")
    parser.add_argument("--tadB")
    parser.add_argument("--reference")
    parser.add_argument("--tolerance",type=int,default=2)

    args=parser.parse_args()

    # -----------------------------
    # Step 1: Harmonize boundaries
    # -----------------------------
    harmonize_leaf_boundaries([args.tadA,args.tadB],args.tolerance)

    tadA_filtered=args.tadA.replace(".tad","_filtered.tad")
    tadB_filtered=args.tadB.replace(".tad","_filtered.tad")

    # -----------------------------
    # Step 2: Load filtered TADs
    # -----------------------------
    tadsA=load_tads(tadA_filtered)
    tadsB=load_tads(tadB_filtered)
    ref=load_tads(args.reference)

    ref_dir = os.path.dirname(args.reference)
    ref_base = os.path.basename(args.reference)

    name, ext = os.path.splitext(ref_base)

    output_file = os.path.join(
        ref_dir,
        f"{name}_inferred_{args.tolerance}{ext}"
    )

    # -----------------------------
    # Step 3: Compute DP median
    # -----------------------------
    root_tads,score=compute_root_median(tadsA,tadsB)

    print("\nRoot median TADs:",len(root_tads))
    print("Parsimony score:",score)
    with open(output_file,"w") as f:
        for a,b in sorted(root_tads):
            f.write(f"{a}\t{b}\n")

    print("Saved inferred TADs to:",output_file)

    # -----------------------------
    # Step 4: Compare with reference
    # -----------------------------
    precision,recall,f1,matched = compute_metrics(
        root_tads,
        ref,
        args.tolerance
    )

    print("\nEvaluation vs reference")
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1:",f1)

#     fitch_score = compute_fitch_parsimony(
#     tadsA,
#     tadsB,
#     args.tolerance
# )

#     print("Fitch parsimony score:", fitch_score)

#     if score == fitch_score:
#         print("DP median matches Fitch parsimony")
#     else:
#         print("WARNING: DP median differs from Fitch parsimony")

    novel = novel_root_tads(
    root_tads,
    ref,
    tadsA,
    tadsB,
    args.tolerance
)

    print("\nRoot TADs matching reference but absent in leaves:")
    print(len(novel))

#     leaf_distance = compute_leaf_distance(
#     tadsA,
#     tadsB,
#     args.tolerance
# )

#     print("\nLeaf-to-leaf TAD distance:", leaf_distance)
#     print("Root parsimony score:", score)

    # if abs(score - leaf_distance) < 1e-6:
    #     print("✓ Parsimony score equals leaf TAD distance")
    # else:
    #     print("WARNING: Parsimony score differs from leaf TAD distance")
    # print those tads
    # for a in novel:
    #     print(f"Novel root TAD: [{a[0]},{a[1]}]")


if __name__=="__main__":
    main()