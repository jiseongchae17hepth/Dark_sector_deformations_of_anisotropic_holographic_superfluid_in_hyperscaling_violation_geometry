import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_workflows import csv_dump, fit_small_branch, json_dump
from hsv_solver_forks import D4_MU0_SANITY_TOL, MU0_D4, d4_detect_local_minima, d4_scan_mu_slice


REFINED_MU_MULTIPLIERS = (1.001, 1.002, 1.005, 1.01, 1.015, 1.02, 1.03, 1.04, 1.06, 1.08, 1.10)
REFINED_EPSILON_GRID = (0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def point_evidence(rows: List[Dict]) -> Dict[str, float]:
    accepted = [row for row in rows if row.get("accepted", "").lower() == "true"]
    ordered = [row for row in accepted if row.get("branch_type") == "ordered"]
    numeric_ordered = []
    for row in ordered:
        numeric_row = dict(row)
        if numeric_row.get("dOmega_vs_iso") not in (None, ""):
            numeric_row["dOmega_vs_iso"] = as_float(numeric_row["dOmega_vs_iso"])
        numeric_row["epsilon"] = as_float(numeric_row["epsilon"])
        numeric_ordered.append(numeric_row)
    negative = [row for row in ordered if row.get("dOmega_vs_iso") not in (None, "") and as_float(row["dOmega_vs_iso"]) < 0.0]
    minima = d4_detect_local_minima(numeric_ordered)
    return {
        "accepted_ordered_count": len(ordered),
        "negative_domega_count": len(negative),
        "stationary_minima_count": len(minima),
        "min_domega_vs_iso": min((as_float(row["dOmega_vs_iso"]) for row in ordered if row.get("dOmega_vs_iso") not in (None, "")), default=float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    d4_dir = output_root / "hsv_solver_forks" / "d4_analytic_island"
    fit_payload = json.loads((d4_dir / "d4_claim3_sign_map.json").read_text(encoding="utf-8"))
    raw_rows = read_csv(d4_dir / "d4_claim3_raw_scans.csv")

    raw_by_key: Dict[str, List[Dict]] = {}
    for row in raw_rows:
        raw_by_key.setdefault(row["point_key"], []).append(row)

    refined_rows: List[Dict] = []
    fallback_raw_rows: List[Dict] = []
    for row in fit_payload["fit_rows"]:
        point_key = row["point_key"]
        existing_rows = raw_by_key.get(point_key, [])
        evidence = point_evidence(existing_rows)
        refined = dict(row)
        refined.update(evidence)
        refined["refinement_used"] = False
        refined["final_classification"] = "fit_closed" if row.get("fit_success") else "no_stationary_ordered_branch"

        need_fallback = (not row.get("fit_success")) and (evidence["negative_domega_count"] > 0 or evidence["stationary_minima_count"] > 0)
        if need_fallback:
            delta2 = float(row["delta2"])
            alpha_dm = float(row["alpha_dm"])
            nu_target = float(row["nu_target"])
            mu_c_exact = float(row["mu_c_exact"])
            fallback_rows: List[Dict] = []
            minima_rows: List[Dict] = []
            for mu_scale in REFINED_MU_MULTIPLIERS:
                mu_target = mu_c_exact * float(mu_scale)
                slice_rows, slice_minima = d4_scan_mu_slice(
                    point_key=point_key,
                    alpha_dm=alpha_dm,
                    nu_target=nu_target,
                    delta2=delta2,
                    mu_target=mu_target,
                    epsilon_grid=REFINED_EPSILON_GRID,
                )
                fallback_rows.extend(slice_rows)
                minima_rows.extend(slice_minima)
            fallback_raw_rows.extend(fallback_rows)
            refined["refinement_used"] = True
            refined["refined_mu_scales"] = list(REFINED_MU_MULTIPLIERS)
            refined["refined_epsilon_grid"] = list(REFINED_EPSILON_GRID)
            refined["refined_stationary_minima_count"] = len(minima_rows)
            refined["refined_negative_domega_count"] = sum(
                1 for test_row in fallback_rows
                if test_row.get("accepted") and test_row.get("branch_type") == "ordered"
                and test_row.get("dOmega_vs_iso") not in (None, "") and as_float(test_row["dOmega_vs_iso"]) < 0.0
            )
            if len(minima_rows) >= 3:
                fit_input = [{"accepted": True, "mu": test_row["mu"], "psi_sq": test_row["psi_sq"]} for test_row in minima_rows]
                fit = fit_small_branch(fit_input, mu_c_exact)
                if fit.get("fit_success") and abs(float(fit["mu0_minus_mu_c"])) <= D4_MU0_SANITY_TOL:
                    refined.update(fit)
                    refined["final_classification"] = "fit_closed"
                else:
                    refined["final_classification"] = "outside_validity_domain"
                    refined["message"] = "refined minima remain edge-only or fail the onset-intercept consistency check"
            elif refined["refined_negative_domega_count"] > 0:
                refined["final_classification"] = "outside_validity_domain"
                refined["message"] = "negative DeltaOmega rows appear only as sparse edge minima without a trustworthy small-branch family"
            else:
                refined["final_classification"] = "no_stationary_ordered_branch"
                refined["message"] = "refined fixed-mu and tiny-branch scan still finds no stationary ordered thermodynamic minimum"

        refined_rows.append(refined)

    candidate_rows = [
        row for row in refined_rows
        if row.get("final_classification") == "fit_closed" and row.get("c2_sign") in {"negative", "near_zero"}
    ]
    claim2_payload = {
        "summary": {
            "candidate_point_count": len(candidate_rows),
            "same_mu_ordered_ordered_coexistence_points": 0,
        },
        "candidate_points": [row["point_key"] for row in candidate_rows],
        "coexistence_rows": [],
    }

    summary = {
        "point_count": len(refined_rows),
        "fit_closed_points": sum(1 for row in refined_rows if row.get("final_classification") == "fit_closed"),
        "no_stationary_points": sum(1 for row in refined_rows if row.get("final_classification") == "no_stationary_ordered_branch"),
        "outside_validity_points": sum(1 for row in refined_rows if row.get("final_classification") == "outside_validity_domain"),
        "refinement_used_points": sum(1 for row in refined_rows if row.get("refinement_used")),
    }

    csv_dump(d4_dir / "d4_claim3_sign_map_refined.csv", refined_rows)
    csv_dump(d4_dir / "d4_claim3_refinement_raw_rows.csv", fallback_raw_rows)
    json_dump(d4_dir / "d4_claim3_sign_map_refined.json", {"summary": summary, "fit_rows": refined_rows, "fallback_raw_rows": fallback_raw_rows})
    json_dump(d4_dir / "d4_claim2_branch_search_refined.json", claim2_payload)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
