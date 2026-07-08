import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_workflows import csv_dump, json_dump


EXTREMAL_RE = re.compile(r"mu_max=([0-9.eE+-]+)")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim3-json", default=str(ROOT / "outputs" / "d5_claim3" / "claim3_sign_map.json"))
    parser.add_argument("--repair-jsons", nargs="+", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "d5_claim3_refined"))
    args = parser.parse_args()

    original = load_json(Path(args.claim3_json))
    fit_rows = {row["point_key"]: dict(row) for row in original["fit_rows"]}
    repaired_rows = []
    for item in args.repair_jsons:
        payload = load_json(Path(item))
        repaired_rows.extend(payload["fit_rows"])
    for row in repaired_rows:
        fit_rows[row["point_key"]] = dict(row)

    extremality_rows = []
    for row in fit_rows.values():
        msg = str(row.get("message", ""))
        if msg.startswith("no compact onset root before extremality bound"):
            row["closure_mode"] = "extremality_limited_no_linear_onset"
            row["classification"] = "no_linear_onset_before_extremality_bound"
            match = EXTREMAL_RE.search(msg)
            row["mu_extremal_max"] = float(match.group(1)) if match else None
            extremality_rows.append({
                "point_key": row["point_key"],
                "delta2": row["delta2"],
                "alpha_dm": row["alpha_dm"],
                "mu_X": row["mu_X"],
                "classification": row["classification"],
                "mu_extremal_max": row["mu_extremal_max"],
            })

    refined_fit_rows = sorted(
        fit_rows.values(),
        key=lambda row: (float(row["delta2"]), float(row["alpha_dm"]), float(row["mu_X"])),
    )

    summary = {
        "total_points": len(refined_fit_rows),
        "claim3_fit_success_points": sum(1 for row in refined_fit_rows if row.get("fit_success")),
        "positive_c2_points": sum(1 for row in refined_fit_rows if row.get("c2_sign") == "positive"),
        "near_zero_c2_points": sum(1 for row in refined_fit_rows if row.get("c2_sign") == "near_zero"),
        "negative_c2_points": sum(1 for row in refined_fit_rows if row.get("c2_sign") == "negative"),
        "extremality_limited_points": len(extremality_rows),
        "fully_classified_points": sum(
            1 for row in refined_fit_rows
            if row.get("fit_success") or row.get("classification") == "no_linear_onset_before_extremality_bound"
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dump(output_dir / "claim3_sign_map_refined.csv", refined_fit_rows)
    csv_dump(output_dir / "extremality_limited_points.csv", extremality_rows)
    json_dump(output_dir / "claim3_sign_map_refined.json", {
        "summary": summary,
        "fit_rows": refined_fit_rows,
        "extremality_rows": extremality_rows,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
