import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from full_solver_renorm import (
    ModelParams,
    bundle_branch,
    estimate_mu_c,
    grand_potential_from_boundary_data,
    solve_anisotropic,
    solve_branch_by_wh,
    validate_renormalization,
)


D5_DELTA2_VALUES: List[float] = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
D5_ALPHA_DM_VALUES: List[float] = [0.0, 0.4, 0.8, 1.2, 1.6]
D5_MU_X_VALUES: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0]

DEFAULT_SMALL_BRANCH_W0: List[float] = [0.02, 0.04, 0.06, 0.08]
FALLBACK_SMALL_BRANCH_W0: List[float] = [0.01, 0.03, 0.05, 0.10]
DENSE_SMALL_BRANCH_W0: List[float] = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]

SOURCE_TOL = 1.0e-6
NEAR_ZERO_C2_TOL = 5.0e-3
NONTRIVIAL_VEV_TOL = 1.0e-4


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dump(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def csv_dump(path: Path, rows: Sequence[Dict]) -> None:
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def point_key(delta2: float, alpha_dm: float, mu_x: float) -> str:
    return f"d{delta2:.2f}_a{alpha_dm:.1f}_nu{mu_x:.1f}"


def finite_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def profile_from_onset(onset: Dict) -> Optional[Dict[str, List[float]]]:
    if not onset.get("success"):
        return None
    return {
        "u": [float(v) for v in onset["u"]],
        "w": [float(v) for v in onset["w"]],
        "w_u": [float(v) for v in onset["w_u"]],
    }


def nearest_completed_records(target: Dict, completed_records: Sequence[Dict], limit: int = 4) -> List[Dict]:
    usable = []
    for record in completed_records:
        if not record.get("onset_success"):
            continue
        dist = (
            abs(record["delta2"] - target["delta2"]) / 0.05
            + abs(record["alpha_dm"] - target["alpha_dm"]) / 0.4
            + abs(record["mu_X"] - target["mu_X"]) / 0.5
        )
        usable.append((dist, record))
    usable.sort(key=lambda item: item[0])
    return [record for _, record in usable[:limit]]


def onset_brackets(point: Dict, neighbors: Sequence[Dict]) -> List[Tuple[float, float]]:
    brackets: List[Tuple[float, float]] = []
    centers = [record["mu_c"] for record in neighbors if record.get("mu_c") is not None]
    if centers:
        center = statistics.median(centers[: min(3, len(centers))])
        for width in [0.06, 0.12, 0.25, 0.50]:
            brackets.append((max(1.20, center - width), center + width))

    baseline = 4.0 / math.sqrt(3.0)
    guess = baseline - 3.0 * (point["delta2"] - 0.10) - 0.45 * point["alpha_dm"] - 0.10 * point["mu_X"]
    guess = max(1.50, guess)
    for width in [0.25, 0.50, 1.00]:
        brackets.append((max(1.20, guess - width), guess + width))

    brackets.append((1.50, 4.00))

    deduped: List[Tuple[float, float]] = []
    seen = set()
    for left, right in brackets:
        key = (round(left, 6), round(right, 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((left, right))
    return deduped


def exact_isotropic_mu_max(p: ModelParams) -> Optional[float]:
    disc = 3.0 / p.delta2 - p.alpha_tilde * p.nu_target * p.nu_target
    if disc <= 0.0:
        return None
    return float(-p.mix * p.nu_target + math.sqrt(disc))


def solve_onset_for_point(p: ModelParams, point: Dict, completed_records: Sequence[Dict]) -> Dict:
    neighbors = nearest_completed_records(point, completed_records)
    candidate_profiles = []
    for record in neighbors[:2]:
        if record.get("onset_profile") is not None:
            candidate_profiles.append(record["onset_profile"])
    candidate_profiles.append(None)

    neighbor_center = None
    neighbor_mu = [record["mu_c"] for record in neighbors if record.get("mu_c") is not None]
    if neighbor_mu:
        neighbor_center = statistics.median(neighbor_mu)
    mu_extremal_max = exact_isotropic_mu_max(p)

    def onset_is_physical(result: Dict) -> bool:
        mu_c = result.get("mu_c")
        source = result.get("source")
        if mu_c is None or not math.isfinite(float(mu_c)) or float(mu_c) <= 0.0:
            return False
        if source is not None and math.isfinite(float(source)) and abs(float(source)) > 1.0e-4:
            return False
        if neighbor_center is not None and abs(float(mu_c) - neighbor_center) > 1.25:
            return False
        if mu_extremal_max is not None and float(mu_c) >= mu_extremal_max:
            return False
        return True

    last_failure = {"success": False, "message": "no onset attempt"}
    for guess_profile in candidate_profiles:
        for bracket in onset_brackets(point, neighbors)[:5]:
            if mu_extremal_max is not None:
                capped = (bracket[0], min(bracket[1], mu_extremal_max - 1.0e-3))
                if capped[0] >= capped[1]:
                    continue
                bracket = capped
            out = estimate_mu_c(bracket, p, guess_profile=guess_profile)
            if out.get("success") and onset_is_physical(out):
                return out
            last_failure = out

    if mu_extremal_max is not None:
        grid = np.linspace(1.0e-3, max(1.0e-3, mu_extremal_max - 1.0e-3), 8)
        source_values = []
        from full_solver_renorm import w_source_for_mu  # local import to avoid unused global dependency

        for mu_trial in grid:
            source = w_source_for_mu(float(mu_trial), p)
            if math.isfinite(source):
                source_values.append(float(source))
        if source_values and all(value > 0.0 for value in source_values):
            return {
                "success": False,
                "message": f"no compact onset root before extremality bound mu_max={mu_extremal_max:.6f}",
                "mu_extremal_max": mu_extremal_max,
            }
    return last_failure


def branch_guess_candidates(
    p: ModelParams,
    mu_c: float,
    point: Dict,
    completed_records: Sequence[Dict],
    last_local_guess: Optional[np.ndarray],
) -> List[np.ndarray]:
    guesses: List[np.ndarray] = []
    if last_local_guess is not None:
        guesses.append(np.asarray(last_local_guess, dtype=float))

    for record in nearest_completed_records(point, completed_records):
        neighbor_guess = record.get("branch_seed")
        if neighbor_guess is not None:
            guesses.append(np.asarray(neighbor_guess, dtype=float))

    guesses.append(np.array([0.0, 0.0, 2.0 * mu_c, 2.0 * p.nu_target], dtype=float))
    guesses.append(np.array([0.0, 0.0, 2.0 * (mu_c + 0.05), 2.0 * p.nu_target], dtype=float))

    deduped: List[np.ndarray] = []
    seen = set()
    for guess in guesses:
        key = tuple(round(float(v), 10) for v in guess.tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(guess)
    return deduped


def accept_small_branch(branch: Dict, mu_c: float, p: ModelParams) -> bool:
    if not branch.get("success"):
        return False
    source = finite_float(branch.get("source"))
    residual = finite_float(branch.get("residual_inf"))
    mu = finite_float(branch.get("mu"))
    vev = finite_float(branch.get("vev"))
    if source is None or residual is None or mu is None or vev is None:
        return False
    if abs(source) > SOURCE_TOL or residual > p.root_residual_tol:
        return False
    if abs(vev) <= 1.0e-8 or abs(vev) >= 20.0:
        return False
    if abs(mu - mu_c) > 0.20:
        return False
    return True


def branch_row_from_result(branch: Dict, p: ModelParams, point: Dict, mu_c: float, w0: float) -> Dict:
    row = {
        "point_key": point["point_key"],
        "delta2": point["delta2"],
        "alpha_dm": point["alpha_dm"],
        "mu_X": point["mu_X"],
        "mu_c": mu_c,
        "w0": float(w0),
        "success": bool(branch.get("success")),
        "message": branch.get("message", ""),
        "residual_inf": finite_float(branch.get("residual_inf")),
        "temperature": finite_float(branch.get("temperature")),
        "mu": finite_float(branch.get("mu")),
        "source": finite_float(branch.get("source")),
        "vev": finite_float(branch.get("vev")),
    }
    if branch.get("success"):
        thermo = grand_potential_from_boundary_data(branch, p, mu=float(branch["mu"]))
        row.update({
            "rho": thermo["rho"],
            "rho_d": thermo["rho_d"],
            "epsilon": thermo["epsilon"],
            "p_bar": thermo["p_bar"],
            "Omega_ren": thermo["Omega_ren"],
            "minus_p_bar": thermo["minus_p_bar"],
        })
    else:
        row.update({
            "rho": None,
            "rho_d": None,
            "epsilon": None,
            "p_bar": None,
            "Omega_ren": None,
            "minus_p_bar": None,
        })

    mu_val = row["mu"]
    vev_val = row["vev"]
    if mu_val is not None and vev_val is not None:
        psi = abs(vev_val)
        row["psi_abs"] = psi
        row["psi_sq"] = psi * psi
        row["mu_minus_mu_c"] = mu_val - mu_c
    else:
        row["psi_abs"] = None
        row["psi_sq"] = None
        row["mu_minus_mu_c"] = None
    row["accepted"] = accept_small_branch(branch, mu_c, p)
    return row


def solve_small_branch_family(
    p: ModelParams,
    point: Dict,
    mu_c: float,
    completed_records: Sequence[Dict],
    w0_values: Optional[Iterable[float]] = None,
) -> Tuple[List[Dict], Optional[np.ndarray]]:
    w0_values = list(w0_values or DEFAULT_SMALL_BRANCH_W0)
    rows: List[Dict] = []
    last_local_guess: Optional[np.ndarray] = None
    best_seed: Optional[np.ndarray] = None

    def run_w0_grid(candidate_w0: Sequence[float]) -> None:
        nonlocal last_local_guess, best_seed
        for w0 in candidate_w0:
            branch: Dict = {"success": False, "message": "no branch guess attempted"}
            for guess in branch_guess_candidates(p, mu_c, point, completed_records, last_local_guess):
                out = solve_branch_by_wh(float(w0), p, guess=guess)
                if out.get("success"):
                    branch = out
                    last_local_guess = np.asarray(out["params"][:4], dtype=float)
                    if best_seed is None:
                        best_seed = last_local_guess.copy()
                    break
                branch = out
            rows.append(branch_row_from_result(branch, p, point, mu_c, float(w0)))

    run_w0_grid(w0_values)
    if sum(1 for row in rows if row["accepted"]) < 3:
        existing_w0 = {round(row["w0"], 8) for row in rows}
        fallback_w0 = [w0 for w0 in FALLBACK_SMALL_BRANCH_W0 if round(w0, 8) not in existing_w0]
        run_w0_grid(fallback_w0)
    return rows, best_seed


def fit_small_branch(point_rows: Sequence[Dict], mu_c: float) -> Dict:
    accepted_rows = [row for row in point_rows if row.get("accepted")]
    if len(accepted_rows) < 3:
        return {
            "fit_success": False,
            "message": "fewer than three accepted small-branch points",
            "accepted_points": len(accepted_rows),
            "mu_c": mu_c,
        }

    psi_sq = np.asarray([row["psi_sq"] for row in accepted_rows], dtype=float)
    mu_vals = np.asarray([row["mu"] for row in accepted_rows], dtype=float)
    design = np.column_stack([np.ones_like(psi_sq), psi_sq, psi_sq * psi_sq])
    coeffs, _, _, _ = np.linalg.lstsq(design, mu_vals, rcond=None)
    fit_vals = design @ coeffs
    residuals = mu_vals - fit_vals
    mu0, c2, c4 = [float(v) for v in coeffs.tolist()]
    c2_sign = "near_zero"
    if c2 > NEAR_ZERO_C2_TOL:
        c2_sign = "positive"
    elif c2 < -NEAR_ZERO_C2_TOL:
        c2_sign = "negative"

    return {
        "fit_success": True,
        "message": "",
        "accepted_points": len(accepted_rows),
        "mu_c": float(mu_c),
        "mu0": mu0,
        "c2": c2,
        "c4": c4,
        "mu0_minus_mu_c": float(mu0 - mu_c),
        "rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "psi_sq_min": float(np.min(psi_sq)),
        "psi_sq_max": float(np.max(psi_sq)),
        "c2_sign": c2_sign,
        "claim2_candidate": bool(c2 <= 0.0 or abs(c2) <= NEAR_ZERO_C2_TOL),
    }


def run_renorm_validation(output_dir: Path) -> Dict:
    output_dir = ensure_dir(output_dir)
    summary = validate_renormalization()
    json_dump(output_dir / "renorm_validation.json", summary)

    flat_rows: List[Dict] = []
    for block, values in summary.items():
        row = {"check": block}
        row.update(values)
        flat_rows.append(row)
    csv_dump(output_dir / "renorm_validation.csv", flat_rows)
    return summary


def run_claim3_sign_map(
    output_dir: Path,
    delta2_values: Optional[Sequence[float]] = None,
    alpha_dm_values: Optional[Sequence[float]] = None,
    mu_x_values: Optional[Sequence[float]] = None,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    delta2_values = list(delta2_values or D5_DELTA2_VALUES)
    alpha_dm_values = list(alpha_dm_values or D5_ALPHA_DM_VALUES)
    mu_x_values = list(mu_x_values or D5_MU_X_VALUES)

    raw_rows: List[Dict] = []
    fit_rows: List[Dict] = []
    completed_records: List[Dict] = []

    for delta2 in delta2_values:
        for alpha_dm in alpha_dm_values:
            for mu_x in mu_x_values:
                point = {
                    "point_key": point_key(delta2, alpha_dm, mu_x),
                    "delta2": float(delta2),
                    "alpha_dm": float(alpha_dm),
                    "mu_X": float(mu_x),
                }
                p = ModelParams(alpha_dm=alpha_dm, delta2=delta2, nu_target=mu_x)
                onset = solve_onset_for_point(p, point, completed_records)
                point_record = dict(point)
                point_record["onset_success"] = bool(onset.get("success"))
                point_record["onset_profile"] = profile_from_onset(onset)
                point_record["branch_seed"] = None

                if not onset.get("success"):
                    fit_row = dict(point)
                    fit_row.update({
                        "fit_success": False,
                        "message": onset.get("message", "onset solve failed"),
                        "accepted_points": 0,
                        "mu_c": None,
                        "mu0": None,
                        "c2": None,
                        "c4": None,
                        "mu0_minus_mu_c": None,
                        "rmse": None,
                        "max_abs_residual": None,
                        "psi_sq_min": None,
                        "psi_sq_max": None,
                        "c2_sign": "unresolved",
                        "claim2_candidate": False,
                    })
                    fit_rows.append(fit_row)
                    completed_records.append(point_record)
                    continue

                mu_c = float(onset["mu_c"])
                point_record["mu_c"] = mu_c
                point_rows, branch_seed = solve_small_branch_family(p, point, mu_c, completed_records)
                point_record["branch_seed"] = branch_seed.tolist() if branch_seed is not None else None
                raw_rows.extend(point_rows)

                fit_row = dict(point)
                fit_row.update(fit_small_branch(point_rows, mu_c))
                fit_rows.append(fit_row)
                completed_records.append(point_record)

                csv_dump(output_dir / "claim3_raw_branches.csv", raw_rows)
                csv_dump(output_dir / "claim3_sign_map.csv", fit_rows)
                json_dump(
                    output_dir / "claim3_partial.json",
                    {
                        "grid": {
                            "delta2_values": delta2_values,
                            "alpha_dm_values": alpha_dm_values,
                            "mu_x_values": mu_x_values,
                        },
                        "fit_rows": fit_rows,
                        "completed_records": [
                            {
                                key: value
                                for key, value in record.items()
                                if key not in {"onset_profile", "branch_seed"}
                            }
                            for record in completed_records
                        ],
                    },
                )

    result = {
        "grid": {
            "delta2_values": delta2_values,
            "alpha_dm_values": alpha_dm_values,
            "mu_x_values": mu_x_values,
        },
        "fit_rows": fit_rows,
        "raw_rows": raw_rows,
        "summary": {
            "total_points": len(fit_rows),
            "fit_success_points": sum(1 for row in fit_rows if row["fit_success"]),
            "claim2_candidates": sum(1 for row in fit_rows if row.get("claim2_candidate")),
            "negative_c2_points": sum(1 for row in fit_rows if row.get("c2_sign") == "negative"),
            "near_zero_c2_points": sum(1 for row in fit_rows if row.get("c2_sign") == "near_zero"),
            "positive_c2_points": sum(1 for row in fit_rows if row.get("c2_sign") == "positive"),
        },
    }

    csv_dump(output_dir / "claim3_raw_branches.csv", raw_rows)
    csv_dump(output_dir / "claim3_sign_map.csv", fit_rows)
    json_dump(output_dir / "claim3_sign_map.json", result)
    return result


def dedupe_ordered_branches(rows: Sequence[Dict]) -> List[Dict]:
    unique: List[Dict] = []
    for row in rows:
        vev = abs(float(row["vev"]))
        omega = float(row["Omega_ren"])
        duplicate = False
        for existing in unique:
            if abs(vev - abs(float(existing["vev"]))) < 1.0e-3 and abs(omega - float(existing["Omega_ren"])) < 1.0e-5:
                duplicate = True
                break
        if not duplicate:
            unique.append(row)
    return unique


def select_claim2_candidates(sign_map_rows: Sequence[Dict]) -> List[Dict]:
    return [
        row for row in sign_map_rows
        if row.get("fit_success") and row.get("claim2_candidate")
    ]


def branch_search_rows_for_point(raw_rows: Sequence[Dict], point: Dict) -> List[float]:
    point_rows = [
        row for row in raw_rows
        if row["point_key"] == point["point_key"] and row.get("accepted")
    ]
    mu_values = sorted({round(float(row["mu"]), 6) for row in point_rows if row.get("mu") is not None})
    for extra in [point["mu_c"], point.get("mu0")]:
        if extra is not None:
            mu_values.append(round(float(extra), 6))
    return sorted(set(mu_values))


def run_claim2_search(output_dir: Path, claim3_result: Dict) -> Dict:
    output_dir = ensure_dir(output_dir)
    sign_map_rows = list(claim3_result["fit_rows"])
    raw_rows = list(claim3_result["raw_rows"])
    candidates = select_claim2_candidates(sign_map_rows)

    search_rows: List[Dict] = []
    coexistence_rows: List[Dict] = []

    for candidate in candidates:
        p = ModelParams(
            alpha_dm=float(candidate["alpha_dm"]),
            delta2=float(candidate["delta2"]),
            nu_target=float(candidate["mu_X"]),
        )
        mu_values = branch_search_rows_for_point(raw_rows, candidate)
        seed_values = [0.05, 0.10, 0.20, 0.40, 0.80, 1.50, 2.50, 4.00]

        for mu in mu_values:
            iso = bundle_branch(mu, p.alpha_dm, delta2=p.delta2, nu_target=p.nu_target, branch_type="iso")
            if iso.get("success"):
                search_rows.append({
                    "point_key": candidate["point_key"],
                    "delta2": candidate["delta2"],
                    "alpha_dm": candidate["alpha_dm"],
                    "mu_X": candidate["mu_X"],
                    "mu": mu,
                    "branch_kind": "trivial",
                    "seed_w0": 0.0,
                    "success": True,
                    "source": 0.0,
                    "residual_inf": 0.0,
                    "vev": 0.0,
                    "Omega_ren": iso["Omega_ren"],
                })

            ordered_rows: List[Dict] = []
            for seed in seed_values:
                guess = np.array([0.0, 0.0, 2.0 * mu, 2.0 * p.nu_target, seed], dtype=float)
                out = solve_anisotropic(mu, p, guess=guess)
                row = {
                    "point_key": candidate["point_key"],
                    "delta2": candidate["delta2"],
                    "alpha_dm": candidate["alpha_dm"],
                    "mu_X": candidate["mu_X"],
                    "mu": mu,
                    "branch_kind": "ordered",
                    "seed_w0": seed,
                    "success": bool(out.get("success")),
                    "source": finite_float(out.get("source")),
                    "residual_inf": finite_float(out.get("residual_inf")),
                    "vev": finite_float(out.get("vev")),
                    "Omega_ren": None,
                }
                if out.get("success"):
                    thermo = grand_potential_from_boundary_data(out, p, mu=mu)
                    row["Omega_ren"] = thermo["Omega_ren"]
                    row["accepted"] = bool(
                        abs(float(out["vev"])) > 1.0e-8
                        and abs(float(out["source"])) <= SOURCE_TOL
                        and float(out["residual_inf"]) <= p.root_residual_tol
                    )
                    if row["accepted"]:
                        ordered_rows.append(row)
                else:
                    row["accepted"] = False
                search_rows.append(row)

            unique_ordered = dedupe_ordered_branches([row for row in ordered_rows if row["accepted"]])
            if len(unique_ordered) >= 2:
                coexistence_rows.append({
                    "point_key": candidate["point_key"],
                    "delta2": candidate["delta2"],
                    "alpha_dm": candidate["alpha_dm"],
                    "mu_X": candidate["mu_X"],
                    "mu": mu,
                    "ordered_branch_count": len(unique_ordered),
                    "ordered_vev_values": [float(row["vev"]) for row in unique_ordered],
                    "ordered_Omega_values": [float(row["Omega_ren"]) for row in unique_ordered],
                })

    result = {
        "candidate_points": candidates,
        "search_rows": search_rows,
        "coexistence_rows": coexistence_rows,
        "summary": {
            "candidate_point_count": len(candidates),
            "candidate_mu_count": len({(row["point_key"], row["mu"]) for row in search_rows if row["branch_kind"] == "ordered"}),
            "ordered_ordered_coexistence_count": len(coexistence_rows),
        },
    }
    csv_dump(output_dir / "claim2_branch_search.csv", search_rows)
    json_dump(output_dir / "claim2_branch_search.json", result)
    csv_dump(output_dir / "claim2_coexistence_candidates.csv", coexistence_rows)
    return result


def point_dict_from_row(row: Dict) -> Dict:
    return {
        "point_key": row["point_key"],
        "delta2": float(row["delta2"]),
        "alpha_dm": float(row["alpha_dm"]),
        "mu_X": float(row["mu_X"]),
    }


def fit_row_distance(a: Dict, b: Dict) -> float:
    return (
        abs(float(a["delta2"]) - float(b["delta2"])) / 0.05
        + abs(float(a["alpha_dm"]) - float(b["alpha_dm"])) / 0.4
        + abs(float(a["mu_X"]) - float(b["mu_X"])) / 0.5
    )


def neighbor_fit_rows(target_row: Dict, fit_rows: Sequence[Dict], require_fit_success: bool = True, limit: int = 6) -> List[Dict]:
    candidates: List[Tuple[float, Dict]] = []
    for row in fit_rows:
        if row["point_key"] == target_row["point_key"]:
            continue
        if require_fit_success and not row.get("fit_success"):
            continue
        candidates.append((fit_row_distance(target_row, row), row))
    candidates.sort(key=lambda item: item[0])
    return [row for _, row in candidates[:limit]]


def preferred_seed_rows_for_target(target_row: Dict, fit_rows: Sequence[Dict], limit: int = 3) -> List[Dict]:
    target_delta = float(target_row["delta2"])
    target_alpha = float(target_row["alpha_dm"])
    target_mu_x = float(target_row["mu_X"])

    same_delta_alpha = [
        row for row in fit_rows
        if row["point_key"] != target_row["point_key"]
        and row.get("fit_success")
        and float(row["delta2"]) == target_delta
        and float(row["alpha_dm"]) == target_alpha
    ]
    same_delta_alpha.sort(key=lambda row: abs(float(row["mu_X"]) - target_mu_x))
    if same_delta_alpha:
        picked = same_delta_alpha[:1]
        for row in neighbor_fit_rows(target_row, fit_rows, require_fit_success=True, limit=limit + 2):
            if row["point_key"] in {item["point_key"] for item in picked}:
                continue
            picked.append(row)
            if len(picked) >= limit:
                break
        return picked

    return neighbor_fit_rows(target_row, fit_rows, require_fit_success=True, limit=limit)


def reconstruct_resolved_seed(fit_row: Dict) -> Optional[Dict]:
    point = point_dict_from_row(fit_row)
    p = ModelParams(alpha_dm=point["alpha_dm"], delta2=point["delta2"], nu_target=point["mu_X"])
    mu_anchor = finite_float(fit_row.get("mu_c"))
    if mu_anchor is None:
        mu_anchor = finite_float(fit_row.get("mu0"))
    if mu_anchor is None:
        return None

    base_guess = np.array([0.0, 0.0, 2.0 * mu_anchor, 2.0 * p.nu_target], dtype=float)
    for w0 in [0.005, 0.01, 0.02]:
        out = solve_branch_by_wh(w0, p, guess=base_guess)
        if not out.get("success"):
            continue
        mu_val = finite_float(out.get("mu"))
        if mu_val is None or mu_val <= 0.0 or abs(mu_val - mu_anchor) > 0.15:
            continue
        return {
            "point_key": fit_row["point_key"],
            "seed4": np.asarray(out["params"][:4], dtype=float),
            "mu_anchor": mu_anchor,
            "w0_anchor": float(w0),
        }
    return None


def dense_small_branch_repair(target_row: Dict, fit_rows: Sequence[Dict]) -> Dict:
    point = point_dict_from_row(target_row)
    p = ModelParams(alpha_dm=point["alpha_dm"], delta2=point["delta2"], nu_target=point["mu_X"])
    mu_c = float(target_row["mu_c"])
    repaired_rows: List[Dict] = []
    seed_bundles = []
    for neighbor in preferred_seed_rows_for_target(target_row, fit_rows, limit=4):
        seed = reconstruct_resolved_seed(neighbor)
        if seed is not None:
            seed_bundles.append(seed)

    last_local_guess: Optional[np.ndarray] = None
    for w0 in DENSE_SMALL_BRANCH_W0:
        trial_candidates: List[Tuple[float, Dict, Optional[np.ndarray]]] = []
        if last_local_guess is not None:
            out = solve_branch_by_wh(w0, p, guess=last_local_guess)
            row = branch_row_from_result(out, p, point, mu_c, w0)
            if row["accepted"] and row["mu"] is not None and row["mu"] > 0.0:
                trial_candidates.append((abs(float(row["mu"]) - mu_c), row, np.asarray(out["params"][:4], dtype=float)))

        for seed in seed_bundles:
            seed4 = np.asarray(seed["seed4"], dtype=float)
            for scale in [0.90, 0.95, 1.00, 1.05]:
                guess = seed4.copy()
                guess[2] *= scale
                guess[3] = 2.0 * p.nu_target * scale
                out = solve_branch_by_wh(w0, p, guess=guess)
                row = branch_row_from_result(out, p, point, mu_c, w0)
                if row["accepted"] and row["mu"] is not None and row["mu"] > 0.0:
                    trial_candidates.append((abs(float(row["mu"]) - mu_c), row, np.asarray(out["params"][:4], dtype=float)))

        for scale in [0.90, 0.95, 1.00, 1.05, 1.10]:
            guess = np.array([0.0, 0.0, 2.0 * mu_c * scale, 2.0 * p.nu_target * scale], dtype=float)
            out = solve_branch_by_wh(w0, p, guess=guess)
            row = branch_row_from_result(out, p, point, mu_c, w0)
            if row["accepted"] and row["mu"] is not None and row["mu"] > 0.0:
                trial_candidates.append((abs(float(row["mu"]) - mu_c), row, np.asarray(out["params"][:4], dtype=float)))

        if trial_candidates:
            trial_candidates.sort(key=lambda item: item[0])
            _, best_row, best_guess = trial_candidates[0]
            repaired_rows.append(best_row)
            last_local_guess = best_guess

    fit_row = dict(target_row)
    fit_row.update(fit_small_branch(repaired_rows, mu_c))
    fit_row["closure_mode"] = "dense_neighbor_seed_repair"
    return {
        "fit_row": fit_row,
        "raw_rows": repaired_rows,
        "seed_sources": [seed["point_key"] for seed in seed_bundles],
    }


def ordered_branch_is_nontrivial(branch: Dict, p: ModelParams, mu: float) -> Optional[Dict]:
    if not branch.get("success"):
        return None
    source = finite_float(branch.get("source"))
    residual = finite_float(branch.get("residual_inf"))
    vev = finite_float(branch.get("vev"))
    if source is None or residual is None or vev is None:
        return None
    if abs(source) > SOURCE_TOL or residual > p.root_residual_tol or abs(vev) < NONTRIVIAL_VEV_TOL:
        return None
    thermo = grand_potential_from_boundary_data(branch, p, mu=mu)
    return {
        "vev": float(vev),
        "source": float(source),
        "residual_inf": float(residual),
        "Omega_ren": float(thermo["Omega_ren"]),
        "rho": float(thermo["rho"]),
        "rho_d": float(thermo["rho_d"]),
    }


def search_extremality_limited_point(target_row: Dict, fit_rows: Sequence[Dict]) -> Dict:
    point = point_dict_from_row(target_row)
    p = ModelParams(alpha_dm=point["alpha_dm"], delta2=point["delta2"], nu_target=point["mu_X"])
    mu_max = exact_isotropic_mu_max(p)
    seed_bundles = []
    for neighbor in preferred_seed_rows_for_target(target_row, fit_rows, limit=1):
        seed = reconstruct_resolved_seed(neighbor)
        if seed is not None:
            seed_bundles.append(seed)

    raw_rows: List[Dict] = []
    positive_mu_probe = []
    for w0 in [0.02, 0.10]:
        found = False
        for seed in seed_bundles:
            seed4 = np.asarray(seed["seed4"], dtype=float)
            for scale in [0.90, 0.95]:
                guess = seed4.copy()
                guess[2] *= scale
                guess[3] = 2.0 * p.nu_target * scale
                out = solve_branch_by_wh(w0, p, guess=guess)
                row = {
                    "point_key": point["point_key"],
                    "delta2": point["delta2"],
                    "alpha_dm": point["alpha_dm"],
                    "mu_X": point["mu_X"],
                    "scan_type": "w0_probe",
                    "w0": float(w0),
                    "seed_source": seed["point_key"],
                    "scale": float(scale),
                    "success": bool(out.get("success")),
                    "mu": finite_float(out.get("mu")),
                    "vev": finite_float(out.get("vev")),
                    "source": finite_float(out.get("source")),
                    "residual_inf": finite_float(out.get("residual_inf")),
                    "Omega_ren": None,
                }
                nontrivial = None
                if out.get("success") and row["mu"] is not None and row["mu"] > 0.0:
                    nontrivial = ordered_branch_is_nontrivial(out, p, float(row["mu"]))
                    if nontrivial is not None:
                        row.update(nontrivial)
                        positive_mu_probe.append(row)
                        found = True
                        raw_rows.append(row)
                        break
                raw_rows.append(row)
            if found:
                break

    unique_ordered_rows = dedupe_ordered_branches([row for row in positive_mu_probe if row.get("Omega_ren") is not None])
    classification = "extremality_limited_no_w0_family_found"
    if unique_ordered_rows:
        classification = "extremality_limited_positive_mu_w0_family_found"

    closed_row = dict(target_row)
    closed_row["closure_mode"] = "extremality_branch_probe"
    closed_row["classification"] = classification
    closed_row["mu_extremal_max"] = mu_max
    closed_row["ordered_branch_found"] = bool(unique_ordered_rows)
    closed_row["ordered_ordered_coexistence_found"] = False
    closed_row["fit_success"] = False
    return {
        "fit_row": closed_row,
        "search_rows": raw_rows,
        "ordered_rows": unique_ordered_rows,
        "coexistence_rows": [],
        "seed_sources": [seed["point_key"] for seed in seed_bundles],
    }


def close_unresolved_d5(
    output_dir: Path,
    claim3_result: Dict,
    target_delta2_values: Optional[Sequence[float]] = None,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    original_fit_rows = list(claim3_result["fit_rows"])
    fit_rows_by_key = {row["point_key"]: dict(row) for row in original_fit_rows}
    sparse_repairs = []
    extremality_repairs = []
    repaired_raw_rows: List[Dict] = []
    extremality_search_rows: List[Dict] = []
    target_delta2_values = None if target_delta2_values is None else {round(float(v), 8) for v in target_delta2_values}

    def selected(row: Dict) -> bool:
        if target_delta2_values is None:
            return True
        return round(float(row["delta2"]), 8) in target_delta2_values

    sparse_rows = [
        row for row in original_fit_rows
        if row.get("message") == "fewer than three accepted small-branch points" and selected(row)
    ]
    for row in sparse_rows:
        repair = dense_small_branch_repair(row, list(fit_rows_by_key.values()))
        fit_rows_by_key[row["point_key"]] = repair["fit_row"]
        repaired_raw_rows.extend(repair["raw_rows"])
        sparse_repairs.append({
            "point_key": row["point_key"],
            "accepted_points": repair["fit_row"].get("accepted_points"),
            "fit_success": repair["fit_row"].get("fit_success"),
            "mu_c": repair["fit_row"].get("mu_c"),
            "mu0": repair["fit_row"].get("mu0"),
            "c2": repair["fit_row"].get("c2"),
            "seed_sources": repair["seed_sources"],
        })

    extremality_rows = [
        fit_rows_by_key[row["point_key"]]
        for row in original_fit_rows
        if str(row.get("message", "")).startswith("no compact onset root before extremality bound") and selected(row)
    ]
    for row in extremality_rows:
        repair = search_extremality_limited_point(row, list(fit_rows_by_key.values()))
        fit_rows_by_key[row["point_key"]] = repair["fit_row"]
        extremality_search_rows.extend(repair["search_rows"])
        extremality_repairs.append({
            "point_key": row["point_key"],
            "classification": repair["fit_row"]["classification"],
            "mu_extremal_max": repair["fit_row"]["mu_extremal_max"],
            "ordered_branch_found": repair["fit_row"]["ordered_branch_found"],
            "ordered_ordered_coexistence_found": repair["fit_row"]["ordered_ordered_coexistence_found"],
            "seed_sources": repair["seed_sources"],
        })

    refined_fit_rows = sorted(
        fit_rows_by_key.values(),
        key=lambda row: (float(row["delta2"]), float(row["alpha_dm"]), float(row["mu_X"])),
    )
    summary = {
        "total_points": len(refined_fit_rows),
        "claim3_fit_success_points": sum(1 for row in refined_fit_rows if row.get("fit_success")),
        "positive_c2_points": sum(1 for row in refined_fit_rows if row.get("c2_sign") == "positive"),
        "near_zero_c2_points": sum(1 for row in refined_fit_rows if row.get("c2_sign") == "near_zero"),
        "negative_c2_points": sum(1 for row in refined_fit_rows if row.get("c2_sign") == "negative"),
        "sparse_repairs_total": len(sparse_repairs),
        "sparse_repairs_closed": sum(1 for row in sparse_repairs if row.get("fit_success")),
        "extremality_points_total": len(extremality_repairs),
        "extremality_points_with_ordered_branch": sum(1 for row in extremality_repairs if row.get("ordered_branch_found")),
        "extremality_points_with_ordered_ordered_coexistence": sum(
            1 for row in extremality_repairs if row.get("ordered_ordered_coexistence_found")
        ),
        "fully_classified_points": sum(
            1
            for row in refined_fit_rows
            if row.get("fit_success") or str(row.get("classification", "")).startswith("extremality_limited_")
        ),
    }

    csv_dump(output_dir / "claim3_sparse_repairs.csv", sparse_repairs)
    csv_dump(output_dir / "claim3_sparse_repaired_raw_rows.csv", repaired_raw_rows)
    csv_dump(output_dir / "extremality_branch_probe.csv", extremality_search_rows)
    csv_dump(output_dir / "extremality_branch_probe_summary.csv", extremality_repairs)
    json_dump(output_dir / "d5_closure_refined.json", {
        "summary": summary,
        "fit_rows": refined_fit_rows,
        "sparse_repairs": sparse_repairs,
        "extremality_repairs": extremality_repairs,
    })
    return {
        "summary": summary,
        "fit_rows": refined_fit_rows,
        "sparse_repairs": sparse_repairs,
        "extremality_repairs": extremality_repairs,
        "repaired_raw_rows": repaired_raw_rows,
        "extremality_search_rows": extremality_search_rows,
    }


def write_status_summary(
    output_path: Path,
    renorm_summary: Dict,
    claim3_result: Dict,
    claim2_result: Dict,
    hsv_status: Optional[Dict] = None,
) -> None:
    fit_rows = claim3_result["fit_rows"]
    resolved_rows = [row for row in fit_rows if row.get("fit_success")]
    unresolved_rows = [row for row in fit_rows if not row.get("fit_success")]
    positive_rows = [row for row in resolved_rows if row.get("c2_sign") == "positive"]
    negative_rows = [row for row in resolved_rows if row.get("c2_sign") == "negative"]
    near_zero_rows = [row for row in resolved_rows if row.get("c2_sign") == "near_zero"]
    extremality_limited = [
        row for row in unresolved_rows
        if str(row.get("message", "")).startswith("no compact onset root before extremality bound")
    ]
    sparse_branch_rows = [
        row for row in unresolved_rows
        if row.get("message") == "fewer than three accepted small-branch points"
    ]

    lines = [
        "# Reanalysis status",
        "",
        "## What is closed",
        f"- The AdS5 onset helper now uses an exact isotropic background plus compact-coordinate shooting/BVP extraction.",
        f"- The corrected thermodynamic scheme is enforced: only boundary-data Omega_ren is used.",
        f"- Cutoff-stability validation was re-run and saved to JSON/CSV under the renorm validation output.",
        "",
        "## Claim 3",
        f"- Total D=5 grid points requested: {claim3_result['summary']['total_points']}.",
        f"- Points with usable small-branch fits: {claim3_result['summary']['fit_success_points']}.",
        f"- Unresolved points: {len(unresolved_rows)}.",
        f"- Positive c2 points: {len(positive_rows)}.",
        f"- Near-zero c2 points: {len(near_zero_rows)}.",
        f"- Negative c2 points: {len(negative_rows)}.",
    ]

    if resolved_rows:
        mu_gap_max = max(abs(float(row["mu0_minus_mu_c"])) for row in resolved_rows if row.get("mu0_minus_mu_c") is not None)
        lines.append(f"- Max |mu0 - mu_c| across resolved fits: {mu_gap_max:.6e}.")
    if extremality_limited:
        lines.append(f"- Extremality-limited points with no linear onset root before the isotropic bound: {len(extremality_limited)}.")
    if sparse_branch_rows:
        lines.append(f"- Points where onset exists but fewer than three clean small-branch points were accepted: {len(sparse_branch_rows)}.")
    if claim2_result["summary"]["candidate_point_count"] == 0:
        lines.append("- No D=5 point triggered the Claim 2 search criterion (c2 <= 0 or nearly zero).")
    else:
        lines.append(f"- Claim 2 search candidates: {claim2_result['summary']['candidate_point_count']}.")
        lines.append(f"- Ordered-ordered coexistence demonstrations found: {claim2_result['summary']['ordered_ordered_coexistence_count']}.")

    lines.extend([
        "",
        "## Physical reading",
        "- In the corrected D=5 scheme, c2 > 0 means the sampled small ordered branch remains supercritical wherever the fit resolves cleanly.",
        "- Claim 2 stays open unless two distinct nontrivial ordered branches at the same mu are directly shown. The search output records whether that ever happens.",
        "",
        "## HSV branch",
    ])

    if hsv_status is None:
        lines.append("- No corrected HSV thermodynamic closure is claimed in this run.")
    else:
        for item in hsv_status.get("lines", []):
            lines.append(f"- {item}")

    lines.append("")
    lines.append("## Cutoff validation snapshot")
    for block, values in renorm_summary.items():
        lines.append(f"- {block}: {json.dumps(values, sort_keys=True)}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
