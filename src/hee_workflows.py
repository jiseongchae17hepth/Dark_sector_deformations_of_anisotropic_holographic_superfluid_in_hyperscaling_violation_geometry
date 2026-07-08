import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from analysis_workflows import csv_dump, ensure_dir, json_dump
from full_solver_renorm import ModelParams, bundle_branch, solve_branch_by_wh


D5_HEE_WIDTHS: Tuple[float, ...] = (0.20, 0.30, 0.40, 0.50)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def d5_hee_strip_observables(branch: Dict, width_target: float, r_upper: float = 12.0) -> Dict[str, float]:
    """Strip entropies on the D=5 anisotropic anchor.

    Parallel means the finite strip direction is the x-direction, i.e. the same
    direction as the vector order. Perpendicular means the finite direction is
    one of the y/z directions.
    """

    sol = branch["solution"]

    def metric(r: float) -> Tuple[float, float]:
        vals = np.asarray(sol.sol(float(r)), dtype=float)
        N = float(vals[0])
        f = float(vals[2])
        return N, f

    def width_integrand_parallel(r: float, r_star: float) -> float:
        N, f = metric(r)
        return (f * f) / (r * math.sqrt(N) * math.sqrt((r ** 6) / (r_star ** 6) - 1.0))

    def area_integrand_parallel(r: float, r_star: float) -> float:
        N, f = metric(r)
        return (r ** 5) * (f * f) / (r_star ** 3 * math.sqrt(N) * math.sqrt((r ** 6) / (r_star ** 6) - 1.0))

    def width_integrand_perp(r: float, r_star: float) -> float:
        N, f = metric(r)
        return 1.0 / (r * f * math.sqrt(N) * math.sqrt((r ** 6) / (r_star ** 6) - 1.0))

    def area_integrand_perp(r: float, r_star: float) -> float:
        N, f = metric(r)
        return (r ** 5) / (r_star ** 3 * f * math.sqrt(N) * math.sqrt((r ** 6) / (r_star ** 6) - 1.0))

    def width_from_rstar(r_star: float, direction: str) -> float:
        integrand = width_integrand_parallel if direction == "parallel" else width_integrand_perp
        value, _ = quad(integrand, r_star + 1.0e-6, r_upper, args=(r_star,), limit=300, epsabs=1.0e-10, epsrel=1.0e-9)
        return 2.0 * value

    def area_from_rstar(r_star: float, direction: str) -> float:
        integrand = area_integrand_parallel if direction == "parallel" else area_integrand_perp
        value, _ = quad(integrand, r_star + 1.0e-6, r_upper, args=(r_star,), limit=300, epsabs=1.0e-10, epsrel=1.0e-9)
        return 2.0 * value

    try:
        r_star_parallel = brentq(lambda rs: width_from_rstar(rs, "parallel") - width_target, 1.001, r_upper - 0.2)
        r_star_perp = brentq(lambda rs: width_from_rstar(rs, "perp") - width_target, 1.001, r_upper - 0.2)
        s_parallel = area_from_rstar(r_star_parallel, "parallel")
        s_perp = area_from_rstar(r_star_perp, "perp")
        return {
            "success": True,
            "width": float(width_target),
            "r_star_parallel": float(r_star_parallel),
            "r_star_perp": float(r_star_perp),
            "S_parallel": float(s_parallel),
            "S_perp": float(s_perp),
            "O12_EE": float(s_perp - s_parallel),
        }
    except Exception as exc:
        return {
            "success": False,
            "width": float(width_target),
            "r_star_parallel": float("nan"),
            "r_star_perp": float("nan"),
            "S_parallel": float("nan"),
            "S_perp": float("nan"),
            "O12_EE": float("nan"),
            "message": repr(exc),
        }


def _select_d5_control_rows(raw_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    targets = [
        ("d0.10_a0.0_nu0.0", 0.02),
        ("d0.10_a0.8_nu1.0", 0.02),
        ("d0.20_a1.2_nu1.5", 0.02),
    ]
    selected: List[Dict[str, str]] = []
    for point_key, max_w0 in targets:
        block = [
            row for row in raw_rows
            if row["point_key"] == point_key
            and row.get("accepted", "").lower() == "true"
            and _as_float(row["w0"]) <= max_w0 + 1.0e-12
        ]
        block = sorted(block, key=lambda row: _as_float(row["w0"]))
        if block:
            selected.append(block[0])
    return selected


def run_d5_control_hee(output_dir: Path, raw_branch_csv: Path) -> Dict:
    output_dir = ensure_dir(output_dir)
    raw_rows = _read_csv_rows(raw_branch_csv)
    selected_rows = _select_d5_control_rows(raw_rows)
    results: List[Dict] = []
    branch_summaries: List[Dict] = []

    iso_cases = [
        {"label": "d5_iso_control", "mu": 4.0, "alpha_dm": 0.0, "delta2": 0.10, "mu_X": 0.0, "branch_type": "iso"},
    ]

    for case in iso_cases:
        branch = bundle_branch(case["mu"], case["alpha_dm"], delta2=case["delta2"], nu_target=case["mu_X"], branch_type="iso")
        for width in D5_HEE_WIDTHS:
            obs = d5_hee_strip_observables(branch, width)
            row = {
                **case,
                "w0": 0.0,
                "success": bool(obs["success"]),
                "width": float(width),
                "S_parallel": obs["S_parallel"],
                "S_perp": obs["S_perp"],
                "O12_EE": obs["O12_EE"],
                "r_star_parallel": obs["r_star_parallel"],
                "r_star_perp": obs["r_star_perp"],
            }
            results.append(row)

    for selected in selected_rows:
        p = ModelParams(alpha_dm=_as_float(selected["alpha_dm"]), delta2=_as_float(selected["delta2"]), nu_target=_as_float(selected["mu_X"]))
        branch = solve_branch_by_wh(_as_float(selected["w0"]), p)
        label = selected["point_key"]
        if not branch.get("success"):
            branch_summaries.append({"label": label, "success": False, "message": branch.get("message")})
            continue
        branch_summaries.append({
            "label": label,
            "success": True,
            "mu": float(branch["mu"]),
            "w0": float(selected["w0"]),
            "temperature": float(branch["temperature"]),
            "vev": float(branch["vev"]),
        })
        for width in D5_HEE_WIDTHS:
            obs = d5_hee_strip_observables(branch, width)
            results.append({
                "label": label,
                "mu": float(branch["mu"]),
                "alpha_dm": float(selected["alpha_dm"]),
                "delta2": float(selected["delta2"]),
                "mu_X": float(selected["mu_X"]),
                "branch_type": "ordered",
                "w0": float(selected["w0"]),
                "temperature": float(branch["temperature"]),
                "vev": float(branch["vev"]),
                "success": bool(obs["success"]),
                "width": float(width),
                "S_parallel": obs["S_parallel"],
                "S_perp": obs["S_perp"],
                "O12_EE": obs["O12_EE"],
                "r_star_parallel": obs["r_star_parallel"],
                "r_star_perp": obs["r_star_perp"],
            })

    csv_dump(output_dir / "d5_hee_controls.csv", results)
    summary = {
        "branch_count": len(branch_summaries) + len(iso_cases),
        "width_row_count": len(results),
        "successful_width_rows": sum(1 for row in results if row.get("success")),
        "isotropic_zero_check_max_abs_o12": max(
            abs(float(row["O12_EE"]))
            for row in results
            if row.get("label") == "d5_iso_control" and math.isfinite(float(row["O12_EE"]))
        ),
    }
    payload = {"summary": summary, "branch_summaries": branch_summaries, "rows": results}
    json_dump(output_dir / "d5_hee_controls.json", payload)
    return payload


def analyze_d4_hee_response(output_dir: Path, d4_hee_scan_csv: Path, d4_response_csv: Path, d4_width_fit_csv: Path) -> Dict:
    output_dir = ensure_dir(output_dir)
    scan_rows = _read_csv_rows(d4_hee_scan_csv)
    response_rows = _read_csv_rows(d4_response_csv)
    width_fit_rows = _read_csv_rows(d4_width_fit_csv)

    response_fit_rows: List[Dict] = []
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
    for row in response_rows:
        key = (row["delta2"], row["nu_ratio_to_mu0"], row["epsilon"], row["width"])
        grouped.setdefault(key, []).append(row)

    supported_count = 0
    for key, block in grouped.items():
        usable = sorted(block, key=lambda row: _as_float(row["alpha_dm_sq_over4"]))
        xs = np.asarray([_as_float(row["alpha_dm_sq_over4"]) for row in usable], dtype=float)
        ys = np.asarray([_as_float(row["delta_O12_EE"]) for row in usable], dtype=float)
        denom = float(np.dot(xs, xs))
        slope = float(np.dot(xs, ys) / denom) if denom > 0.0 else 0.0
        pred = slope * xs
        rmse = float(np.sqrt(np.mean((ys - pred) ** 2)))
        scale = float(np.max(np.abs(ys))) if len(ys) else 0.0
        rel_rmse = float(rmse / scale) if scale > 1.0e-18 else 0.0
        supported = bool(rel_rmse <= 0.12)
        supported_count += int(supported)
        response_fit_rows.append({
            "delta2": float(key[0]),
            "nu_ratio_to_mu0": float(key[1]),
            "epsilon": float(key[2]),
            "width": float(key[3]),
            "fit_point_count": len(usable),
            "alpha_sq_slope": slope,
            "rmse": rmse,
            "relative_rmse": rel_rmse,
            "quadratic_response_supported": supported,
        })

    collapse_rows: List[Dict] = []
    collapse_grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in response_rows:
        if abs(_as_float(row["alpha_dm_sq_over4"])) <= 1.0e-12:
            continue
        key = (row["delta2"], row["nu_ratio_to_mu0"], row["epsilon"])
        collapse_grouped.setdefault(key, []).append(row)

    for key, block in collapse_grouped.items():
        by_width: Dict[float, List[float]] = {}
        for row in block:
            alpha_sq = _as_float(row["alpha_dm_sq_over4"])
            if alpha_sq <= 0.0:
                continue
            by_width.setdefault(_as_float(row["width"]), []).append(_as_float(row["delta_O12_EE"]) / alpha_sq)
        for width, values in sorted(by_width.items()):
            values_arr = np.asarray(values, dtype=float)
            collapse_rows.append({
                "delta2": float(key[0]),
                "nu_ratio_to_mu0": float(key[1]),
                "epsilon": float(key[2]),
                "width": float(width),
                "collapse_mean": float(np.mean(values_arr)),
                "collapse_std": float(np.std(values_arr)),
                "collapse_rel_std": float(np.std(values_arr) / max(np.mean(np.abs(values_arr)), 1.0e-18)),
            })

    isotropic_rows = [
        row for row in scan_rows
        if row.get("accepted", "").lower() == "true"
        and abs(_as_float(row["epsilon"])) <= 1.0e-12
    ]
    summary = {
        "response_fit_count": len(response_fit_rows),
        "quadratic_response_supported_count": supported_count,
        "quadratic_response_supported_fraction": float(supported_count / max(len(response_fit_rows), 1)),
        "isotropic_zero_check_max_abs_o12": max(abs(_as_float(row["O12_EE"])) for row in isotropic_rows) if isotropic_rows else None,
        "small_width_fit_count": len(width_fit_rows),
        "median_reported_small_width_exponent": float(np.median([_as_float(row["small_width_exponent"]) for row in width_fit_rows])) if width_fit_rows else None,
    }

    csv_dump(output_dir / "d4_hee_response_fits.csv", response_fit_rows)
    csv_dump(output_dir / "d4_hee_collapse_rows.csv", collapse_rows)
    payload = {"summary": summary, "response_fit_rows": response_fit_rows, "collapse_rows": collapse_rows}
    json_dump(output_dir / "d4_hee_response_analysis.json", payload)
    return payload


def write_claim4_note(output_path: Path, d4_analysis: Dict, d5_controls: Dict) -> None:
    summary = d4_analysis["summary"]
    d5_summary = d5_controls["summary"]
    lines = [
        "# Claim 4 / HEE Closure Note",
        "",
        "## What Is Closed",
        f"- Corrected D=4 effective O12_EE data now exist on {summary['response_fit_count']} response-fit slices and {summary['small_width_fit_count']} width-fit slices.",
        f"- The isotropic limit is numerically clean in the D=4 effective scan: max |O12_EE| at epsilon=0 is {summary['isotropic_zero_check_max_abs_o12']}.",
        f"- D5 control HEE data were computed on {d5_summary['width_row_count']} width rows, with isotropic max |O12_EE| = {d5_summary['isotropic_zero_check_max_abs_o12']}.",
        "",
        "## What The Data Support",
        f"- On the corrected D=4 effective island, the leading dark-sector response is reasonably captured by an alpha_dm^2 fit on {summary['quadratic_response_supported_count']} / {summary['response_fit_count']} tested slices.",
        "",
        "## What Is Not Closed",
        "- The global HEE claim is still open.",
        "- The current D=4 effective small-width exponent extracted from the stored corrected data does not robustly reproduce a clean W^4 law by itself.",
        "- There is still no corrected nonlinear D=3 dark-HSV branch/HEE closure, and no full corrected global HEE phase diagram across D=5 + HSV branches.",
        "",
        "## Final Claim 4 Reading",
        "- Claim 4 is still open with a precise obstruction: corrected D4 effective O12_EE data and D5 control data exist, but the full corrected global branch closure needed to promote the HEE scaling claim is still missing.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
