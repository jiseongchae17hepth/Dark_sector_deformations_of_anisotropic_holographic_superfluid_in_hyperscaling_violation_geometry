#!/usr/bin/env python3
"""Generate core solver-output tables used by the v161 figure provenance.

This wrapper runs the actual solver workflows from the package modules.  It is
intended for rerunning solver tables, not for plotting.  Some dense scans are
computationally expensive; use --quick for a smoke test.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analysis_workflows import run_claim3_sign_map, run_renorm_validation
from hsv_solver_forks import run_hsv_solver_forks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs_v161"))
    parser.add_argument("--quick", action="store_true", help="run a small subset for validation")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    if args.quick:
        summary["claim3_sign_map"] = run_claim3_sign_map(
            out / "d5_claim3_quick",
            delta2_values=[0.10],
            alpha_dm_values=[0.0, 0.8],
            mu_x_values=[2.0],
        )["summary"]
    else:
        summary["claim3_sign_map"] = run_claim3_sign_map(out / "d5_claim3")["summary"]
        summary["renorm_validation"] = run_renorm_validation(out / "d5_renorm_validation")
        summary["hsv_solver_forks"] = run_hsv_solver_forks(out / "hsv_solver_forks")

    (out / "v161_solver_generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
