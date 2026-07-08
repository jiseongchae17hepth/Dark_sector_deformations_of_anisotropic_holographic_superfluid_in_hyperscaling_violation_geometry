import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hee_workflows import analyze_d4_hee_response, run_d5_control_hee, write_claim4_note


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    hee_dir = output_root / "hee_closure"
    d5_controls = run_d5_control_hee(hee_dir / "d5_controls", output_root / "d5_claim3_refined" / "claim3_raw_branches_refined.csv")
    d4_analysis = analyze_d4_hee_response(
        hee_dir / "d4_analysis",
        output_root / "hsv_solver_forks" / "d4_analytic_island" / "d4_hee_scan.csv",
        output_root / "hsv_solver_forks" / "d4_analytic_island" / "d4_hee_alpha_response.csv",
        output_root / "hsv_solver_forks" / "d4_analytic_island" / "d4_hee_width_fits.csv",
    )
    write_claim4_note(hee_dir / "claim4_status.md", d4_analysis, d5_controls)
    summary = {
        "d5_controls": d5_controls["summary"],
        "d4_analysis": d4_analysis["summary"],
    }
    (hee_dir / "hee_closure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
