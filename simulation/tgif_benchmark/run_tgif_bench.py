#!/usr/bin/env python3
"""Benchmark TGIF-DB against the tree-informed imputation arm, on the same data.

WHY TGIF IS THE RIGHT EXTERNAL BASELINE
---------------------------------------
``../tree_informed_tad_inference_robustad`` tests one claim: a TAD caller run on
one leaf's noisy Hi-C misses real domains, and the leaf's relatives are evidence
it has them, so the phylogeny can repair the leaf's calls. TGIF-DB (Roy lab,
https://github.com/Roy-lab/tgif) attacks exactly that problem from the other
end: instead of calling each leaf independently and repairing afterwards, it
factorizes all the leaves *jointly* with a tree-guided regularizer, so a leaf's
boundary estimate is shrunk toward its parent's during inference.

    ours   : call per leaf (RobusTAD) -> tree-informed ILP repairs the calls
    TGIF-DB: tree-regularized multi-task NMF -> boundaries fall out of the fit

Same inputs (leaf matrices + a tree), same output (per-leaf boundaries), two
completely different mechanisms. That makes it a real baseline and not a
strawman, which is why the professor asked for it.

WHAT THIS SCRIPT DOES
---------------------
It reproduces the RobusTAD arm's simulation BIT-FOR-BIT -- same trees, same
hierarchies, same base matrices, same down-sampling draws consumed in the same
order -- and hands the identical degraded leaf matrices to TGIF-DB. Then it
scores TGIF and our methods against the same hidden truth, at the same noise
rungs, with the same tolerance.

SCORING IS AT THE BOUNDARY LEVEL, AND IT HAS TO BE
--------------------------------------------------
TGIF-DB emits per-bin boundary calls; it never assembles nested TAD intervals.
Our methods emit TAD intervals. There is no way to score TGIF on intervals, so
EVERYTHING is scored on boundaries: an interval (a, b) contributes the two
0-indexed boundary bins {a-1, b-1}, and predicted boundaries are matched to true
ones one-to-one within +/- ``--tolerance`` bins. Interval-level F1 is still
recorded for our methods (as ``f1_interval``) so this experiment's numbers can
be lined up against the arm's own figures, but the headline comparison is the
boundary one, because that is the only comparison TGIF can be part of.

Two boundary scorings are reported for every method:

  * ``full``   -- over the whole region. TGIF drops sparse bins (see below) and
                  is charged for the truth it cannot reach there. This is the
                  honest number: the task is "call this leaf's boundaries".
  * ``masked`` -- restricted to the bins TGIF actually evaluated, for every
                  method alike. This is the charitable number: it asks how good
                  TGIF is *where it chose to speak*.

Both are computed for all methods, so neither reading is cherry-picked.

TGIF'S SPARSITY FILTER IS THE THING TO WATCH
--------------------------------------------
TGIF drops any bin with fewer than ``-t`` of its within-window neighbours
non-zero, IN ANY task, and marks it N/A in the output. On this simulation's
count-poor matrices its own advice for sparse data (``-t 0.2``) is catastrophic:
measured on tree01/region_000, TGIF called 10 boundaries across 10 leaves at
full depth with half the region N/A, and then refused to run at all at every
noisier rung -- against F1 0.69-0.73 with the filter OFF. So the default here is
``--tgif-threshold 0``, and every parameter conclusion drawn under a nonzero
threshold should be treated as provisional (the alpha sweep's preferred value
reversed when the filter came off).

Even at 0, coverage still degrades at the bottom of the ladder -- a bin with no
non-zero neighbours is dropped at any threshold -- so ``n_bins_na`` and
``status`` are recorded at every rung. A TGIF F1 of 0 under high N/A means "TGIF
refused to look", not "TGIF looked and was wrong", and the two must not be
reported as the same thing.

STAGES
------
--stage call   run TGIF-DB on every leaf-set at every depth, cache the calls.
               Needs the compiled tgif-db (see build_tgif.sh). No Python deps
               beyond numpy.
--stage score  score TGIF from that cache, AND re-derive our own methods from
               the RobusTAD arm's cached calls so the comparison is on the same
               leaves. Needs gurobipy (the ILP) -- i.e. the snakemake-tutorial
               env, which is also where GSL lives, so both stages run there.

The split exists so a solver change only re-runs the cheap half; TGIF is ~40 s
per (region, depth) and the ILP is ~1 s.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent          # simulation/tgif_benchmark
SIM_ROOT = EXP_DIR.parent                          # simulation/
REPO_ROOT = SIM_ROOT.parent                        # repo root
SWEEP_DIR = SIM_ROOT / "data_quality_sweep"        # trees.py
RT_ARM = SIM_ROOT / "tree_informed_tad_inference_robustad"
for p in (str(REPO_ROOT), str(SIM_ROOT), str(SWEEP_DIR), str(RT_ARM), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from simulate_matrix import simulate_matrix
from tad_utilities import _match_points_within_tol
from trees import Tree, get_tree

# The RobusTAD arm is the source of truth for the simulation, the noise ladder
# and the methods being compared against. Importing it is deliberate: a copy of
# build_hierarchies() or of DEPTH_LEVELS here would silently drift and the two
# experiments would stop being about the same leaves. It imports neither
# gurobipy nor cooler at module level, so this is cheap in either stage.
import run_tree_impute_rt as rt

DEFAULT_GSL_LIB = "/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/lib"
DEFAULT_TGIF_BIN = EXP_DIR / "tgif" / "tgif-db"

TGIF_METHOD = "tgif"            # significant_boundaries_summit_only.txt
TGIF_FULL_METHOD = "tgif_wide"  # significant_boundaries.txt (whole stretches)

METHOD_LABEL = {
    TGIF_METHOD: "TGIF-DB (tree-regularized NMF)",
    TGIF_FULL_METHOD: "TGIF-DB, full stretches",
    "robustad": "RobusTAD (no tree)",
    "snap": "tree-informed ILP, unweighted",
    "wsnap": "tree-informed ILP, RobusTAD-weighted",
}


# ---------------------------------------------------------------------------
# Simulation replay
#
# This MUST stay a mirror of rt.run_call_stage's generation block. The two
# experiments are only comparable because the leaf matrices are identical, and
# they are identical only if the RNGs are seeded the same way AND consumed in
# the same order:
#   * sim_rng is shared across leaves and drawn in tree.leaves order,
#   * deg_rng is drawn depth-major then leaf-minor, and rt.downsample() does
#     NOT touch it when frac >= 1.0 -- so the 1.0x rung consumes nothing.
# Any change to that block in the arm has to be mirrored here.
# ---------------------------------------------------------------------------
def base_matrices(tree: Tree, region: int, params: config.SimParams,
                  amplitude: float) -> dict[str, np.ndarray]:
    hiers = rt.build_hierarchies(tree, region, params)
    base_params = dataclasses.replace(params, A=amplitude, noise=0.0, depth=None)
    sim_rng = np.random.default_rng(
        params.seed * 7 + rt._stable_hash(tree.name) % 100_000 + region
    )
    return {leaf: simulate_matrix(hiers[leaf], params.n_bins, base_params, sim_rng)
            for leaf in tree.leaves}


def degradation_rng(params: config.SimParams, region: int) -> np.random.Generator:
    return np.random.default_rng(params.seed * 13 + region + 7919)


# ---------------------------------------------------------------------------
# TGIF input files
# ---------------------------------------------------------------------------
def write_bed(path: Path, n_bins: int, resolution: int, chrom: str = "chrS") -> None:
    """Bin coordinates. TGIF works in bp, this simulation works in bins."""
    with path.open("w") as f:
        for i in range(n_bins):
            f.write(f"{chrom}\t{i * resolution}\t{(i + 1) * resolution}\t{i}\n")


def write_sparse_matrix(path: Path, mat: np.ndarray, max_dist: int) -> int:
    """Upper triangle, 0-indexed, non-zero only. TGIF symmetrises on read.

    Entries further than ``max_dist`` bins off the diagonal are dropped: TGIF
    ignores them anyway (both its sparsity filter and its reader cut at the
    window size), and keeping them multiplies the file size for nothing.
    """
    n = mat.shape[0]
    written = 0
    with path.open("w") as f:
        for i in range(n):
            hi = min(n, i + max_dist + 1)
            row = mat[i, i:hi]
            nz = np.nonzero(row)[0]
            for k in nz:
                f.write(f"{i}\t{i + int(k)}\t{float(row[int(k)]):.6g}\n")
            written += len(nz)
    return written


def write_tree_file(path: Path, tree: Tree, matrix_paths: dict[str, Path]) -> list[str]:
    """TGIF's tree file. The node ORDER is load-bearing, in two ways.

    ``TGIF::make_tree`` builds the first ``nTasks`` nodes as Leaf, everything up
    to the last as Branch, and the last as Root -- where ``nTasks`` is just the
    count of rows whose 4th column is not "N/A". So every leaf must come first,
    the root must come last, and (as TGIF's README says) a child must be listed
    before its parent, which reverse-BFS gives.

    There is also a latent bug worth knowing about: make_tree links a child to
    its parent only ``if (parentId > 0)``, i.e. a node whose parent is row 1 is
    silently orphaned. Listing leaves first makes row 1 a leaf, and a leaf is
    nobody's parent, so the layout below cannot hit it.

    Returns the leaf aliases in the column order TGIF will use for its output.
    """
    leaves = list(tree.leaves)
    internal_rev = [n for n in reversed(tree.nodes) if n in set(tree.internal)]

    ids: dict[str, int] = {leaf: i + 1 for i, leaf in enumerate(leaves)}
    nid = len(leaves)
    for n in internal_rev:
        nid += 1
        ids[n] = nid
    ids[tree.root] = nid + 1

    parent = tree.parent
    rows = [f"{ids[l]}\t{ids[parent[l]]}\t{l}\t{matrix_paths[l]}" for l in leaves]
    rows += [f"{ids[n]}\t{ids[parent[n]]}\t{n}\tN/A" for n in internal_rev]
    rows.append(f"{ids[tree.root]}\t-1\t{tree.root}\tN/A")
    path.write_text("\n".join(rows) + "\n")
    return leaves


# ---------------------------------------------------------------------------
# TGIF invocation + output parsing
# ---------------------------------------------------------------------------
FILTERED_LINE = re.compile(r"Filtered out (\d+)/(\d+) regions")


def run_tgif(tgif_bin: Path, tree_file: Path, bed_file: Path, out_prefix: Path,
             window_bins: int, window_bp: int, step_bp: int, threshold: float,
             alpha: float, random_state: int, gsl_lib: str,
             timeout: int) -> tuple[float, str, int | None]:
    """Run tgif-db. Returns (seconds, status, surviving bin count).

    Options go BEFORE the positionals -- see the note in build_tgif.sh.

    ``status`` is "ok", or **"too-few-bins"** when TGIF's sparsity filter left
    fewer surviving bins than one factorization window and TGIF then died
    inside GSL. It dies two different ways depending on how few survived --
    "matrix dimension n1 must be positive integer" at exactly zero, and "failed
    to allocate space for row indices" at anything under the window size, where
    its ``nzmax = N + (windowSize-1)*(2N - windowSize)`` goes negative -- but
    both are the same event: TGIF declining to analyse data this thin. Neither
    is a bug to retry, and the honest way to record it is an empty call set with
    every bin marked N/A, which is what the caller does with this status.

    Every OTHER non-zero exit is re-raised. A failure mode we have not
    characterised must never be quietly scored as "TGIF found nothing" -- that
    would turn a broken run into a baseline defeat.
    """
    out_prefix.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if gsl_lib:
        env["DYLD_LIBRARY_PATH"] = gsl_lib + ":" + env.get("DYLD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = gsl_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        str(tgif_bin),
        "-w", str(int(window_bp)), "-s", str(int(step_bp)),
        "-t", str(threshold), "-a", str(int(alpha)), "-r", str(int(random_state)),
        "-b",
        "-o", str(out_prefix) + os.sep,
        str(tree_file), str(bed_file),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    elapsed = time.time() - t0
    m = FILTERED_LINE.search(proc.stdout)
    n_usable = int(m.group(2)) - int(m.group(1)) if m else None

    summit = out_prefix / "significant_boundaries_summit_only.txt"
    if proc.returncode == 0 and summit.exists():
        return elapsed, "ok", n_usable

    if n_usable is not None and n_usable < window_bins:
        return elapsed, "too-few-bins", n_usable

    raise RuntimeError(
        f"tgif-db failed (rc={proc.returncode}, summit file "
        f"{'present' if summit.exists() else 'absent'}, "
        f"{n_usable} bins survived the -t filter against a {window_bins}-bin window).\n"
        f"cmd: {' '.join(cmd)}\n--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )


def parse_boundary_table(path: Path) -> tuple[list[str], dict[str, list[int]], list[int]]:
    """Parse a TGIF boundary table.

    Rows are bins in order; columns after the first three are the leaf aliases,
    each holding 0 / 1 / "N/A". Returns (aliases, {alias: [called bins]},
    [N/A bins]). Bins are 0-indexed, matching the bed file's 4th column.
    """
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("#").split("\t")
    aliases = [h for h in header[3:] if h.strip()]
    called: dict[str, list[int]] = {a: [] for a in aliases}
    na_bins: list[int] = []
    for idx, line in enumerate(lines[1:]):
        cells = [c for c in line.split("\t")[3:] if c.strip()]
        if not cells:
            continue
        if cells[0] == "N/A":
            na_bins.append(idx)
            continue
        for alias, cell in zip(aliases, cells):
            if cell == "1":
                called[alias].append(idx)
    return aliases, called, na_bins


# ---------------------------------------------------------------------------
# Boundary-level scoring
# ---------------------------------------------------------------------------
def intervals_to_boundaries(intervals, n_bins: int) -> set[int]:
    """TAD intervals -> boundary positions, in TGIF's coordinate system.

    A boundary is an INTERFACE between bins, not a bin. Interface ``p`` sits
    between bin p-1 and bin p, so a 0-indexed half-open domain ``[s, e)`` has
    interfaces ``s`` and ``e``. Intervals here are 1-indexed inclusive,
    ``(a, b) = (s + 1, e)``, hence ``{a - 1, b}``.

    Getting this right is not cosmetic, because TGIF speaks the same language:
    its boundary score at index i is the cosine distance between bin i-1's and
    bin i's latent columns, i.e. the interface at position i. Scoring domains by
    their FIRST and LAST BIN instead ({a-1, b-1}) would put every right-hand
    boundary one bin left of where TGIF reports it, and -- worse -- would give
    two adjacent truth boundaries (e-1 and e) for the single shared interface
    between two tiling domains. Our interval-emitting methods would reproduce
    that doubled pair and match both; TGIF, which calls the interface once,
    could only ever match one, and its recall would be halved by the convention
    alone. Interfaces collapse the pair for everybody.

    The two region edges (0 and ``n_bins``) are dropped: they are the outer
    limits of the region, every method "knows" them for free, and TGIF cannot
    report interface 0 at all (its per-window distance vector starts at 0).
    """
    out: set[int] = set()
    for a, b in intervals:
        out.add(int(a) - 1)
        out.add(int(b))
    out.discard(0)
    out.discard(n_bins)
    return out


def boundary_prf(pred: set[int], truth: set[int], tol: int) -> tuple[float, float, float]:
    """One-to-one matching within +/- tol, the same rule the arm uses on TADs."""
    matched = _match_points_within_tol(pred, truth, tol)
    precision = matched / len(pred) if pred else 0.0
    recall = matched / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_boundaries(tree: Tree, pred: dict[str, set[int]],
                     truth: dict[str, set[int]], tol: int,
                     keep: set[int] | None) -> dict[str, float]:
    """Mean per-leaf boundary P/R/F1.

    ``keep`` restricts both sides to a bin subset (the masked scoring); None
    scores the whole region.

    An EMPTY ``keep`` -- TGIF filtered the whole region away -- returns NaN
    rather than zeros. There is no truth left to recall there, so a zero would
    be a scored failure for every method alike, and pooling those zeros with
    rungs that do have coverage would drag every masked mean down for a reason
    that has nothing to do with the methods.
    """
    if keep is not None and not keep:
        return {"precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "n_pred": float("nan")}

    ps, rs, fs, npred = [], [], [], []
    for leaf in tree.leaves:
        p_set = set(pred.get(leaf, set()))
        t_set = set(truth[leaf])
        if keep is not None:
            p_set &= keep
            t_set &= keep
        p, r, f = boundary_prf(p_set, t_set, tol)
        ps.append(p)
        rs.append(r)
        fs.append(f)
        npred.append(len(p_set))
    n = max(len(fs), 1)
    return {
        "precision": sum(ps) / n,
        "recall": sum(rs) / n,
        "f1": sum(fs) / n,
        "n_pred": sum(npred) / n,
    }


# ---------------------------------------------------------------------------
# CALL stage
# ---------------------------------------------------------------------------
def run_call_stage(tree: Tree, region: int, params: config.SimParams, args) -> dict:
    bases = base_matrices(tree, region, params, args.amplitude)
    deg_rng = degradation_rng(params, region)

    work = Path(args.work_dir) / tree.name / f"region_{region:03d}"
    work.mkdir(parents=True, exist_ok=True)
    bed = work / "bins.bed"
    write_bed(bed, params.n_bins, params.resolution_bp)

    # TGIF forces the window to at least 100 bins (and the stride to half of
    # it), so anything smaller than that in bp is silently overridden. Ask for
    # the real thing directly, in bins, so the log says what actually ran.
    window_bins = max(args.window_bins, 100)
    window_bp = window_bins * params.resolution_bp
    step_bp = (window_bins // 2) * params.resolution_bp

    points = []
    for depth_label, depth_value in rt.DEPTH_LEVELS:
        # The ladder is ALWAYS walked in full, even when --depths asks for a
        # subset: deg_rng is consumed rung by rung, so skipping a rung's draws
        # would hand every later rung a different matrix than the arm saw.
        # Only the TGIF run is skipped.
        matrix_paths, nnz = {}, {}
        for leaf in tree.leaves:
            mat = rt.downsample(bases[leaf], depth_value, deg_rng)
            p = work / f"{leaf}.txt"
            nnz[leaf] = write_sparse_matrix(p, mat, window_bins)
            matrix_paths[leaf] = p
        if args.depths and depth_label not in args.depths:
            continue

        tree_file = work / "tree.txt"
        write_tree_file(tree_file, tree, matrix_paths)

        out_prefix = work / f"out_{depth_label}"
        if out_prefix.exists():
            shutil.rmtree(out_prefix)
        elapsed, status, n_usable = run_tgif(
            Path(args.tgif_bin), tree_file, bed, out_prefix,
            window_bins, window_bp, step_bp, args.tgif_threshold, args.tgif_alpha,
            args.tgif_random_state, args.gsl_lib, args.timeout,
        )
        if status == "ok":
            _, summit, na_bins = parse_boundary_table(
                out_prefix / "significant_boundaries_summit_only.txt")
            _, wide, _ = parse_boundary_table(
                out_prefix / "significant_boundaries.txt")
        else:
            # TGIF could not run at all: too little of the region survived its
            # sparsity filter. No calls anywhere, and no bin was evaluated -- so
            # the masked scoring correctly has nothing to score, and the coverage
            # figure shows the refusal for what it is. n_bins_usable keeps the
            # distinction between "nothing survived" and "a few bins survived
            # but fewer than one window" recoverable from the cache.
            summit, wide = {}, {}
            na_bins = list(range(params.n_bins))

        points.append({
            "depth_label": depth_label,
            "depth_value": depth_value,
            "status": status,
            "summit": {leaf: summit.get(leaf, []) for leaf in tree.leaves},
            "wide": {leaf: wide.get(leaf, []) for leaf in tree.leaves},
            "na_bins": na_bins,
            "n_bins_na": len(na_bins),
            "n_bins_usable": n_usable,
            "nnz_per_leaf": round(sum(nnz.values()) / max(len(nnz), 1), 1),
            "tgif_seconds": round(elapsed, 1),
        })
        print(f"{tree.name} region_{region:03d} depth={depth_label} [{status}]: "
              f"{sum(len(v) for v in summit.values())} summit boundaries over "
              f"{len(tree.leaves)} leaves, {len(na_bins)}/{params.n_bins} bins N/A, "
              f"{elapsed:.0f}s", flush=True)

        if not args.keep_work:
            shutil.rmtree(out_prefix, ignore_errors=True)

    if not args.keep_work:
        for leaf in tree.leaves:
            (work / f"{leaf}.txt").unlink(missing_ok=True)

    return {
        "tree": tree.name, "region": region, "tool": "tgif-db",
        "leaves": tree.leaves, "n_bins": params.n_bins,
        "amplitude": args.amplitude,
        "tgif_params": {
            "window_bins": window_bins, "window_bp": window_bp, "step_bp": step_bp,
            "threshold": args.tgif_threshold, "alpha": args.tgif_alpha,
            "random_state": args.tgif_random_state,
        },
        "points": points,
    }


# ---------------------------------------------------------------------------
# SCORE stage
# ---------------------------------------------------------------------------
def our_methods_at_depth(tree: Tree, pt: dict, args) -> dict[str, dict[str, set]]:
    """Re-derive the arm's predictions for one depth rung, as TAD intervals.

    Re-solving instead of reading the arm's results/ JSONs is deliberate: those
    files hold interval-level METRICS, not predictions, and boundaries cannot be
    recovered from a metric. The arm's own functions are called, at the arm's
    own defaults, so ``robustad`` and ``wsnap`` here are the same objects that
    produced its figures.
    """
    calls, tad_scores = {}, {}
    for leaf, rows in pt["leaf_calls"].items():
        calls[leaf] = {(int(r[0]), int(r[1])) for r in rows}
        tad_scores[leaf] = {(int(r[0]), int(r[1])): float(r[2]) for r in rows}
    boundary = {
        leaf: rt.BoundaryScores.from_record(rec, args.tolerance)
        for leaf, rec in pt.get("leaf_boundary", {}).items()
    }

    out: dict[str, dict[str, set]] = {"robustad": rt.method_raw(tree, calls)}
    for prefix in args.method_prefixes:
        use_imp, use_pri, wsnap = rt.METHOD_SWITCHES[prefix]
        pred, _ = rt.method_impute(
            tree, calls, tad_scores, boundary, args.impute_cost,
            args.Ct, args.Cb, args.solve_mode, args.tolerance,
            use_importance=use_imp, use_price_discount=use_pri,
            weighted_snap=wsnap,
            importance_weight=args.importance_weight,
            price_discount=args.price_discount,
            price_floor=args.price_floor,
            calibrate_importance=True,
        )
        if pred is None:
            print(f"  WARNING {prefix} @ {pt['depth_label']}: LP found no solution; "
                  f"falling back to raw calls", flush=True)
            pred = rt.method_raw(tree, calls)
        out[prefix] = pred
    return out


def run_score_stage(tree: Tree, region: int, params: config.SimParams, args) -> dict:
    hiers = rt.build_hierarchies(tree, region, params)
    truth_iv = {leaf: rt.true_intervals(hiers[leaf]) for leaf in tree.leaves}
    truth_bd = {leaf: intervals_to_boundaries(ivs, params.n_bins)
                for leaf, ivs in truth_iv.items()}

    tgif_rec = json.loads(Path(args.tgif_json).read_text())
    arm_rec = None
    if args.arm_calls_json and Path(args.arm_calls_json).exists():
        arm_rec = json.loads(Path(args.arm_calls_json).read_text())
    else:
        print(f"WARNING: no RobusTAD-arm cache at {args.arm_calls_json}; "
              f"scoring TGIF alone", flush=True)
    arm_by_depth = {p["depth_label"]: p for p in arm_rec["points"]} if arm_rec else {}

    all_bins = set(range(params.n_bins))
    rows = []
    for pt in tgif_rec["points"]:
        depth = pt["depth_label"]
        keep = all_bins - set(pt["na_bins"])

        preds_bd: dict[str, dict[str, set[int]]] = {
            TGIF_METHOD: {l: set(map(int, pt["summit"].get(l, []))) for l in tree.leaves},
            TGIF_FULL_METHOD: {l: set(map(int, pt["wide"].get(l, []))) for l in tree.leaves},
        }
        preds_iv: dict[str, dict[str, set]] = {}
        if depth in arm_by_depth:
            preds_iv = our_methods_at_depth(tree, arm_by_depth[depth], args)
            for name, pred in preds_iv.items():
                preds_bd[name] = {
                    l: intervals_to_boundaries(pred.get(l, set()), params.n_bins)
                    for l in tree.leaves}

        summary = []
        for name, pred in preds_bd.items():
            full = score_boundaries(tree, pred, truth_bd, args.tolerance, None)
            masked = score_boundaries(tree, pred, truth_bd, args.tolerance, keep)
            row = {
                "depth_label": depth, "depth_value": pt["depth_value"],
                "method": name, "method_label": METHOD_LABEL.get(name, name),
                "n_bins_na": pt["n_bins_na"], "n_bins": params.n_bins,
                "n_bins_usable": pt.get("n_bins_usable"),
                "tgif_status": pt.get("status", "ok"),
                "tgif_seconds": pt["tgif_seconds"],
                **{f"{k}_full": v for k, v in full.items()},
                **{f"{k}_masked": v for k, v in masked.items()},
            }
            if name in preds_iv:
                # Interval-level F1 for the methods that have intervals, so this
                # experiment can be lined up against the arm's own figures.
                per_leaf = [rt._prf(preds_iv[name][l], truth_iv[l], args.tolerance)
                            for l in tree.leaves]
                row["f1_interval"] = sum(x[2] for x in per_leaf) / max(len(per_leaf), 1)
            rows.append(row)
            summary.append(f"{name}={full['f1']:.3f}")
        print(f"{tree.name} region_{region:03d} depth={depth} "
              f"(N/A {pt['n_bins_na']}/{params.n_bins}): " + " ".join(summary), flush=True)

    return {
        "tree": tree.name, "region": region,
        "tolerance": args.tolerance, "n_leaves": len(tree.leaves),
        "impute_cost": args.impute_cost,
        "tgif_params": tgif_rec["tgif_params"],
        "points": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_params(args) -> config.SimParams:
    return dataclasses.replace(
        config.DEFAULT_PARAMS,
        n_bins=args.n_bins,
        n_top_tads=args.n_top_tads,
        nesting_depth=args.nesting_depth,
        min_size=args.min_size,
        min_score_node_tads=args.min_score_node_tads,
        events_per_edge=args.events_per_edge,
        tolerance=args.tolerance,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    d = config.DEFAULT_PARAMS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True)
    p.add_argument("--region-index", type=int, required=True)
    p.add_argument("--stage", choices=["call", "score"], required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=EXP_DIR / "work")
    p.add_argument("--keep-work", action="store_true",
                   help="keep the per-depth TGIF inputs and raw outputs (~40 MB "
                        "per region); by default only the parsed cache survives")

    # tree / simulation -- MUST match the RobusTAD arm's Snakefile COMMON string
    p.add_argument("--n-nodes", type=int, default=16)
    p.add_argument("--tree-seed", type=int, default=20240101)
    p.add_argument("--amplitude", type=float, default=rt.DEFAULT_AMPLITUDE)
    p.add_argument("--n-bins", type=int, default=500)
    p.add_argument("--n-top-tads", type=int, default=10)
    p.add_argument("--nesting-depth", type=int, default=2)
    p.add_argument("--min-size", type=int, default=8)
    p.add_argument("--min-score-node-tads", type=int, default=6)
    p.add_argument("--events-per-edge", type=int, default=1)
    p.add_argument("--tolerance", type=int, default=d.tolerance)
    p.add_argument("--seed", type=int, default=d.seed)

    # TGIF
    p.add_argument("--tgif-bin", type=Path, default=DEFAULT_TGIF_BIN)
    p.add_argument("--gsl-lib", default=DEFAULT_GSL_LIB,
                   help="prepended to DYLD_LIBRARY_PATH for the tgif-db child")
    p.add_argument("--window-bins", type=int, default=100,
                   help="factorization window, in BINS. TGIF forces a minimum of "
                        "100 and a stride of half the window, so smaller values "
                        "are silently ignored")
    p.add_argument("--tgif-threshold", type=float, default=0.0,
                   help="TGIF's -t sparsity filter: a bin is dropped (N/A) if "
                        "fewer than this fraction of its within-window neighbours "
                        "are non-zero IN ANY leaf. TGIF's own default is 0.5 and "
                        "its README suggests 0.2 for sparse input; on these "
                        "count-poor matrices BOTH cripple it (see the module "
                        "docstring), so 0 -- the filter off -- is the default here")
    p.add_argument("--tgif-alpha", type=float, default=1e6,
                   help="TGIF's tree-regularization strength (-a). This is the "
                        "knob that decides how much the tree matters to TGIF, so "
                        "it is the one worth sweeping")
    p.add_argument("--tgif-random-state", type=int, default=1010)
    p.add_argument("--depths", default="",
                   help="comma-separated depth labels to actually run TGIF on "
                        "(e.g. '1.0x,0.1x'); empty means the whole ladder. For "
                        "cheap pilots only -- the ladder's RNG is still consumed "
                        "in full, so the matrices are unchanged, but the cache "
                        "this writes is PARTIAL and must not be mixed into "
                        "results/ alongside full ones")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--tgif-json", type=Path,
                   help="CALL-stage cache to score (required for --stage score)")

    # our methods (defaults mirror the RobusTAD arm's Snakefile)
    p.add_argument("--arm-calls-json", type=Path,
                   help="the RobusTAD arm's cached calls for this (tree, region); "
                        "without it only TGIF is scored")
    p.add_argument("--methods", default="snap,wsnap",
                   help="which of the arm's method prefixes to re-derive")
    p.add_argument("--impute-cost", type=float, default=0.5,
                   help="the arm's headline impute_cost")
    p.add_argument("--importance-weight", type=float, default=0.1)
    p.add_argument("--price-discount", type=float, default=0.6)
    p.add_argument("--price-floor", type=float, default=0.25)
    p.add_argument("--Ct", type=float, default=1.0)
    p.add_argument("--Cb", type=float, default=1.0)
    p.add_argument("--solve-mode", dest="solve_mode", default="convex")

    args = p.parse_args()
    args.depths = [x.strip() for x in str(args.depths).split(",") if x.strip()]
    known = {label for label, _ in rt.DEPTH_LEVELS}
    bad = [d for d in args.depths if d not in known]
    if bad:
        raise SystemExit(f"unknown depth label(s) {bad}; choose from {sorted(known)}")
    args.method_prefixes = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    unknown = [m for m in args.method_prefixes if m not in rt.METHOD_SWITCHES]
    if unknown:
        raise SystemExit(f"unknown method prefix(es) {unknown}; "
                         f"choose from {sorted(rt.METHOD_SWITCHES)}")
    if args.stage == "score" and not args.tgif_json:
        raise SystemExit("--stage score needs --tgif-json")
    if args.stage == "call" and not Path(args.tgif_bin).exists():
        raise SystemExit(f"no tgif-db at {args.tgif_bin}; run ./build_tgif.sh first")
    return args


def main() -> None:
    args = parse_args()
    params = build_params(args)
    tree = get_tree(args.tree, args.n_nodes, args.tree_seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.stage == "call":
        rec = run_call_stage(tree, args.region_index, params, args)
    else:
        rec = run_score_stage(tree, args.region_index, params, args)

    out = args.outdir / f"region_{args.region_index:03d}.json"
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
