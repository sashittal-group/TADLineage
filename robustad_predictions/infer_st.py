#!/usr/bin/env python3
"""Add ST to robustad_predictions with the existing panels and settings.

build_ontad_panel_and_infer.py produced robustad_predictions/<NODE>/ for the nine
cells with a GSE152918 .mcool. ST had no .mcool; GSE152918_ST.hic was downloaded
from GEO and converted with hic2cool 0.8.3 like the other nine
(refhic_predictions/converted_mcool/GSE152918_ST.mcool).

ST is inferred with exactly the same steps as the other targets, reusing what that
run saved rather than rebuilding anything (so the other cells' calls are untouched):
the OnTAD-biased panel ontad_panel/ontad_panel_<res>.pkl (built from the nine
cells), the RobusTAD windows of ontad_panel/summary_<res>.json, RobusTAD lmcc +
assembly, and the same calibration to ST's own OnTAD calls (chromosomes with < 90%
agreement fall back to OnTAD-derived TADs with nearest-match RobusTAD scores).

Writes robustad_predictions/ST/chr<C>/<res>/ and hemato_cells/using_robustad/ST/
chr<C>/<res>/tads.tad, and ontad_panel/summary_ST_<res>.json.

Run with the refhic env (needs cooler):
    /Users/PSLabMonisha/miniforge3/envs/refhic/bin/python robustad_predictions/infer_st.py
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hemato_cells" / "using_robustad"))

import build_ontad_panel_and_infer as b  # noqa: E402
import build_robustad_tad_files as tadfiles  # noqa: E402

SAMPLE = "ST"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolutions", nargs="*", default=["2mb", "1mb", "500kb", "250kb"],
                    choices=sorted(b.RES_LABEL_TO_BP))
    args = ap.parse_args()

    env = b.robustad_env()
    b.ensure_robustad_config()
    for res_label in args.resolutions:
        saved = json.loads((b.PANEL_ROOT / f"summary_{res_label}.json").read_text())
        panel_path = Path(saved["panel"])
        params = saved["windows"]
        print(f"{SAMPLE} {res_label}: panel {panel_path.name}, minW {params['minW_kb']} kb, "
              f"maxW {params['maxW_kb']} kb", flush=True)
        output_index = b.infer_with_panel(SAMPLE, res_label, panel_path, params, env)
        calibration = b.calibrate_outputs_to_ontad(SAMPLE, res_label, b.RES_LABEL_TO_BP[res_label], output_index)
        evaluation = b.evaluate_sample_resolution(SAMPLE, res_label, output_index)
        (b.PANEL_ROOT / f"summary_{SAMPLE}_{res_label}.json").write_text(json.dumps(
            {"panel": str(panel_path), "windows": params,
             "outputs": output_index, "calibration": calibration, "evaluation": evaluation}, indent=2))
        n_cal = sum(c["calibrated"] for c in calibration.values())
        print(f"  replaced by OnTAD-derived TADs on {n_cal} of {len(calibration)} chromosomes", flush=True)
        for chrom in tadfiles.CHRS:
            if not tadfiles.convert_one(SAMPLE, chrom, res_label, "all"):
                print(f"  missing RobusTAD output for chr{chrom} {res_label}", flush=True)


if __name__ == "__main__":
    main()
