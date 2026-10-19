"""Run a TAD caller on one dense matrix and return its scored calls.

Both callers return a ``dict[(start, end) -> score]`` mapping each called TAD
interval to the caller's own confidence score (OnTAD's TADscore column;
RobusTAD's per-TAD score). The interval keys use the *same* coordinate
convention the shared generator writes truth in (``hierarchy.write_tad``):
1-indexed inclusive, i.e. a 0-indexed half-open domain ``[s, e)`` is reported as
``(s + 1, e)``. Keeping calls and truth in one convention lets the
reconstruction and scoring compare them directly, and the score rides along for
the weighted reconstruction methods (Sankoff). Callers that produce no usable
score for an interval report ``0.0``.

OnTAD needs only its compiled binary. RobusTAD needs a cooler-capable
interpreter, so its heavy imports (``cooler`` and the sibling ``run_robustad``
module) are done lazily inside ``call_robustad`` -- an OnTAD-only sweep never
imports them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

SWEEP_DIR = Path(__file__).resolve().parent
SIM_ROOT = SWEEP_DIR.parent            # simulation/
ROBUSTAD_ANALYSIS_DIR = SIM_ROOT / "using_robustad"


def write_matrix(mat: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, mat, fmt="%d", delimiter="\t")


# ---------------------------------------------------------------------------
# OnTAD
# ---------------------------------------------------------------------------
# OnTAD writes each call as: start end level meanscore TADscore. Column index 4
# is the TADscore -- the reliability weight OnTAD assigns the call, used directly
# as the leaf weight by the Sankoff reconstruction.
ONTAD_SCORE_COL = 4


def _parse_ontad_tad(path: Path) -> dict[tuple[int, int], float]:
    """Parse an OnTAD ``.tad`` file to ``{(start, end): TADscore}``.

    Skips the whole-region level-0 row. Calls without a score column fall back
    to ``0.0``.
    """
    scores: dict[tuple[int, int], float] = {}
    with path.open() as fh:
        for line_no, line in enumerate(fh):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            start, end = int(fields[0]), int(fields[1])
            level = int(fields[2]) if len(fields) >= 3 and fields[2].lstrip("-").isdigit() else None
            if line_no == 0 and level == 0:
                continue
            score = float(fields[ONTAD_SCORE_COL]) if len(fields) > ONTAD_SCORE_COL else 0.0
            scores[(start, end)] = score
    return scores


def call_ontad(
    matrix_path: Path,
    out_prefix: Path,
    ontad_bin: Path,
    params: dict,
    timeout: int,
) -> dict[tuple[int, int], float]:
    """Run OnTAD on ``matrix_path`` and return ``{(start, end): TADscore}``."""
    cleaned = matrix_path.with_suffix(matrix_path.suffix + ".cleaned")
    with matrix_path.open() as src, cleaned.open("w") as dst:
        for line in src:
            stripped = " ".join(line.split())
            if stripped:
                dst.write(stripped + "\n")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ontad_bin), str(cleaned),
        "-o", str(out_prefix),
        "-penalty", str(params.get("penalty", 0.1)),
        "-maxsz", str(params.get("maxsz", 200)),
        "-minsz", str(params.get("minsz", 3)),
        "-lsize", str(params.get("lsize", 5)),
        "-ldiff", str(params.get("ldiff", 0.5)),
    ]
    if params.get("log2", True):
        cmd.append("-log2")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    cleaned.unlink(missing_ok=True)
    tad_path = out_prefix.with_suffix(".tad")
    if result.returncode != 0 or not tad_path.exists():
        # A caller that fails on a badly degraded matrix scores as "no calls",
        # which is the honest outcome for that quality level.
        return {}
    return _parse_ontad_tad(tad_path)


# ---------------------------------------------------------------------------
# RobusTAD
# ---------------------------------------------------------------------------
def call_robustad(
    matrix_path: Path,
    work_dir: Path,
    resolution: int,
    params: dict,
    timeout: int,
) -> dict[tuple[int, int], float]:
    """Run RobusTAD (boundary + assembly) and return ``{(start, end): score}``.

    Reuses the matrix->cooler and subprocess helpers from the sibling
    ``using_robustad/run_robustad.py`` so caller settings stay identical to the
    main analysis. Must run under an interpreter that has ``cooler`` + RobusTAD.
    """
    if str(ROBUSTAD_ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ROBUSTAD_ANALYSIS_DIR))
    import pandas as pd
    import run_robustad as rt  # noqa: E402  (lazy: needs the cooler env)

    work_dir.mkdir(parents=True, exist_ok=True)
    cool_file = work_dir / "m.cool"
    boundary_prefix = work_dir / "m_boundary"
    tad_prefix = work_dir / "m_tads"

    rt.matrix_to_cooler(matrix_path, cool_file, resolution)
    env = rt.robustad_env(work_dir / ".mplcache")
    try:
        boundary_bed = rt.run_boundary(
            cool_file, boundary_prefix, resolution,
            params["minW_kb"], params["maxW_kb"], params["ratio"], params["alpha"],
            env, timeout,
        )
        rt.run_assembly(
            cool_file, boundary_bed, tad_prefix, resolution,
            params["mintad_bp"], params["maxtad_bp"], params["ratio"], params["delta"],
            env, timeout,
        )
    except subprocess.CalledProcessError:
        return {}
    finally:
        cool_file.unlink(missing_ok=True)

    tads_bed = work_dir / "m_tads_all.bed"
    if not tads_bed.exists():
        return {}
    with_bins = rt.convert_tads(tads_bed, resolution)
    df = pd.read_csv(with_bins, sep="\t")
    if df.empty or "start_bin" not in df or "end_bin" not in df:
        return {}

    # start_bin is the 0-indexed first bin; end_bin the 0-indexed last bin
    # (inclusive). The half-open domain is [start_bin, end_bin + 1), which in the
    # truth 1-indexed-inclusive convention is (start_bin + 1, end_bin + 1). The
    # per-TAD "score" column (RobusTAD's confidence) rides along for Sankoff; keep
    # the strongest score per interval if RobusTAD reports duplicates.
    has_score = "score" in df.columns
    scores: dict[tuple[int, int], float] = {}
    for row in df.itertuples(index=False):
        interval = (int(row.start_bin) + 1, int(row.end_bin) + 1)
        score = float(getattr(row, "score")) if has_score else 0.0
        if score > scores.get(interval, float("-inf")):
            scores[interval] = score
    return scores
