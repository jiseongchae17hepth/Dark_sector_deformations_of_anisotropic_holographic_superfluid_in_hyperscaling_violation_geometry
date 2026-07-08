import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis_workflows import (
    csv_dump,
    dedupe_ordered_branches,
    ensure_dir,
    exact_isotropic_mu_max,
    finite_float,
    json_dump,
    ordered_branch_is_nontrivial,
    point_dict_from_row,
    preferred_seed_rows_for_target,
    reconstruct_resolved_seed,
)
from full_solver_renorm import ModelParams, bundle_branch, solve_anisotropic, solve_branch_by_wh


EXTREMALITY_SEED_W0: List[float] = [0.01, 0.02, 0.04, 0.08, 0.15, 0.30, 0.60, 0.90]
EXTREMALITY_DIRECT_W0: List[float] = [0.08, 0.20, 0.50]
EXTREMALITY_GUESS_SCALES: List[float] = [0.85, 0.95, 1.05]
EXTREMALITY_MU_FRACTIONS: List[float] = [0.68, 0.80, 0.88, 0.94, 0.975, 0.992]
EXTREMALITY_SEED_BRANCH_LIMIT = 3
EXTREMALITY_ACCEPTED_VEV_TOL = 1.0e-4


def is_extremality_limited_row(row: Dict) -> bool:
    classification = str(row.get("classification", ""))
    message = str(row.get("message", ""))
    return (
        classification == "no_linear_onset_before_extremality_bound"
        or message.startswith("no compact onset root before extremality bound")
    )


def _seed_guess_from_seed4(seed4: np.ndarray, w0: float) -> np.ndarray:
    return np.array([seed4[0], seed4[1], seed4[2], seed4[3], w0], dtype=float)


def _scale_seed4_charges(seed4: np.ndarray, scale: float, p: ModelParams) -> np.ndarray:
    guess4 = np.asarray(seed4, dtype=float).copy()
    guess4[2] *= scale
    guess4[3] = 2.0 * p.nu_target * scale
    return guess4


def _charge_rescaled_guess(params5: np.ndarray, mu_target: float, mu_ref: float, p: ModelParams) -> np.ndarray:
    scale = 1.0
    if mu_ref > 1.0e-8:
        scale = min(1.25, max(0.75, mu_target / mu_ref))
    guess = np.asarray(params5, dtype=float).copy()
    guess[2] *= scale
    guess[3] = 2.0 * p.nu_target * scale
    return guess


def _attempt_row(
    point: Dict,
    p: ModelParams,
    stage: str,
    seed_source: str,
    mu_target: Optional[float],
    family_id: Optional[str],
    basin_label: str,
    branch: Dict,
    guess: np.ndarray,
) -> Tuple[Dict, Optional[Dict]]:
    row = {
        "point_key": point["point_key"],
        "delta2": point["delta2"],
        "alpha_dm": point["alpha_dm"],
        "mu_X": point["mu_X"],
        "stage": stage,
        "seed_source": seed_source,
        "family_id": family_id,
        "basin_label": basin_label,
        "mu_target": finite_float(mu_target),
        "success": bool(branch.get("success")),
        "message": branch.get("message", ""),
        "guess_params": [float(v) for v in np.asarray(guess, dtype=float).tolist()],
        "mu_solved": finite_float(branch.get("mu")),
        "temperature": finite_float(branch.get("temperature")),
        "source": finite_float(branch.get("source")),
        "residual_inf": finite_float(branch.get("residual_inf")),
        "vev": finite_float(branch.get("vev")),
        "Omega_ren": None,
        "rho": None,
        "rho_d": None,
        "accepted_ordered": False,
        "w0_solved": None,
        "params5": None,
    }
    if branch.get("success"):
        mu_eval = finite_float(branch.get("mu"))
        ordered = ordered_branch_is_nontrivial(branch, p, mu_eval) if mu_eval is not None else None
        row["w0_solved"] = finite_float(branch.get("w0"))
        row["params5"] = [float(v) for v in np.asarray(branch.get("params", []), dtype=float).tolist()]
        if ordered is not None and abs(float(ordered["vev"])) >= EXTREMALITY_ACCEPTED_VEV_TOL:
            row.update(ordered)
            row["accepted_ordered"] = True
            return row, ordered
    return row, None


def _append_unique_seed_branch(seed_rows: List[Dict], row: Dict) -> None:
    mu_val = float(row["mu_solved"])
    vev_val = abs(float(row["vev"]))
    omega_val = float(row["Omega_ren"])
    for existing in seed_rows:
        if (
            abs(mu_val - float(existing["mu_solved"])) < 1.0e-3
            and abs(vev_val - abs(float(existing["vev"]))) < 1.0e-3
            and abs(omega_val - float(existing["Omega_ren"])) < 1.0e-5
        ):
            return
    seed_rows.append(row)


def harvest_extremality_seed_branches(target_row: Dict, fit_rows: Sequence[Dict]) -> Dict:
    point = point_dict_from_row(target_row)
    p = ModelParams(alpha_dm=point["alpha_dm"], delta2=point["delta2"], nu_target=point["mu_X"])
    mu_max = exact_isotropic_mu_max(p)
    if mu_max is None:
        raise ValueError(f"Could not compute isotropic extremality bound for {point['point_key']}")

    attempt_rows: List[Dict] = []
    unique_seed_rows: List[Dict] = []
    resolved_seeds = []
    for neighbor in preferred_seed_rows_for_target(target_row, fit_rows, limit=4):
        seed = reconstruct_resolved_seed(neighbor)
        if seed is not None:
            resolved_seeds.append(seed)

    candidate_seed4: List[Tuple[str, np.ndarray]] = []
    for seed in resolved_seeds:
        seed4 = np.asarray(seed["seed4"], dtype=float)
        for scale in EXTREMALITY_GUESS_SCALES:
            candidate_seed4.append((seed["point_key"], _scale_seed4_charges(seed4, scale, p)))

    for frac in [0.72, 0.86, 0.96]:
        mu_anchor = frac * mu_max
        direct4 = np.array([0.0, 0.0, 2.0 * mu_anchor, 2.0 * p.nu_target * frac], dtype=float)
        candidate_seed4.append((f"direct_mu_fraction_{frac:.2f}", direct4))

    for seed_source, guess4 in candidate_seed4:
        for w0 in EXTREMALITY_SEED_W0:
            guess5 = _seed_guess_from_seed4(guess4, w0)
            out = solve_branch_by_wh(float(w0), p, guess=guess4)
            row, _ = _attempt_row(
                point=point,
                p=p,
                stage="seed_harvest",
                seed_source=seed_source,
                mu_target=finite_float(out.get("mu")),
                family_id=None,
                basin_label=f"w0={w0:.3f}",
                branch=out,
                guess=guess5,
            )
            attempt_rows.append(row)
            if row["accepted_ordered"] and row["mu_solved"] is not None:
                _append_unique_seed_branch(unique_seed_rows, row)
                if len(unique_seed_rows) >= EXTREMALITY_SEED_BRANCH_LIMIT:
                    break
        if len(unique_seed_rows) >= EXTREMALITY_SEED_BRANCH_LIMIT:
            break

    for idx, row in enumerate(unique_seed_rows, start=1):
        row["family_id"] = f"family_{idx}"
    return {
        "seed_rows": unique_seed_rows,
        "attempt_rows": attempt_rows,
        "mu_extremal_max": mu_max,
    }


def build_extremality_mu_targets(mu_max: float, seed_rows: Sequence[Dict]) -> List[float]:
    values = {round(mu_max * frac, 6) for frac in EXTREMALITY_MU_FRACTIONS}
    for row in seed_rows:
        mu_seed = float(row["mu_solved"])
        for delta in [-0.03, 0.0, 0.03]:
            if mu_seed + delta > 0.0:
                values.add(round(mu_seed + delta, 6))
    return sorted(v for v in values if v > 0.0)


def _ordered_seed_guess_bank(target_row: Dict, fit_rows: Sequence[Dict], mu_target: float) -> List[Tuple[str, np.ndarray]]:
    point = point_dict_from_row(target_row)
    p = ModelParams(alpha_dm=point["alpha_dm"], delta2=point["delta2"], nu_target=point["mu_X"])
    guesses: List[Tuple[str, np.ndarray]] = []
    for neighbor in preferred_seed_rows_for_target(target_row, fit_rows, limit=3):
        seed = reconstruct_resolved_seed(neighbor)
        if seed is None:
            continue
        seed4 = np.asarray(seed["seed4"], dtype=float)
        mu_anchor = max(1.0e-8, float(seed["mu_anchor"]))
        charge_scale = min(1.30, max(0.70, mu_target / mu_anchor))
        scaled4 = _scale_seed4_charges(seed4, charge_scale, p)
        for w0 in EXTREMALITY_DIRECT_W0:
            guesses.append((seed["point_key"], _seed_guess_from_seed4(scaled4, w0)))
    for w0 in EXTREMALITY_DIRECT_W0:
        guesses.append((f"direct_mu_{mu_target:.4f}", np.array([0.0, 0.0, 2.0 * mu_target, 2.0 * p.nu_target, w0], dtype=float)))

    deduped: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for source, guess in guesses:
        key = tuple(round(float(v), 8) for v in np.asarray(guess, dtype=float).tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((source, guess))
    return deduped


def continue_seed_family(target_row: Dict, seed_row: Dict, mu_targets: Sequence[float]) -> List[Dict]:
    point = point_dict_from_row(target_row)
    p = ModelParams(alpha_dm=point["alpha_dm"], delta2=point["delta2"], nu_target=point["mu_X"])
    seed_mu = float(seed_row["mu_solved"])
    seed_params = np.asarray(seed_row["params5"], dtype=float)
    family_id = seed_row["family_id"]
    rows: List[Dict] = []

    lower = sorted([mu for mu in mu_targets if mu < seed_mu], reverse=True)
    upper = sorted([mu for mu in mu_targets if mu >= seed_mu])

    def run_direction(targets: Sequence[float]) -> None:
        current_guess = seed_params.copy()
        current_mu = seed_mu
        for mu_target in targets:
            if abs(mu_target - seed_mu) < 5.0e-7:
                seed_copy = dict(seed_row)
                seed_copy["stage"] = "family_continuation"
                seed_copy["mu_target"] = mu_target
                rows.append(seed_copy)
                continue
            trial_guesses = [
                _charge_rescaled_guess(current_guess, mu_target, current_mu, p),
                _charge_rescaled_guess(seed_params, mu_target, seed_mu, p),
            ]
            accepted_row = None
            for idx, guess in enumerate(trial_guesses, start=1):
                out = solve_anisotropic(mu_target, p, guess=guess)
                row, _ = _attempt_row(
                    point=point,
                    p=p,
                    stage="family_continuation",
                    seed_source=seed_row["seed_source"],
                    mu_target=mu_target,
                    family_id=family_id,
                    basin_label=f"continuation_guess_{idx}",
                    branch=out,
                    guess=guess,
                )
                rows.append(row)
                if row["accepted_ordered"]:
                    accepted_row = row
                    current_guess = np.asarray(out["params"], dtype=float)
                    current_mu = mu_target
                    break
            if accepted_row is None:
                current_guess = seed_params.copy()
                current_mu = seed_mu

    run_direction(upper)
    run_direction(lower)
    return rows


def direct_fixed_mu_scan(target_row: Dict, fit_rows: Sequence[Dict], mu_targets: Sequence[float], seed_rows: Sequence[Dict]) -> List[Dict]:
    point = point_dict_from_row(target_row)
    p = ModelParams(alpha_dm=point["alpha_dm"], delta2=point["delta2"], nu_target=point["mu_X"])
    rows: List[Dict] = []
    for mu_target in mu_targets:
        guess_bank: List[Tuple[str, np.ndarray]] = []
        for seed_row in seed_rows:
            guess_bank.append((seed_row["family_id"], _charge_rescaled_guess(np.asarray(seed_row["params5"], dtype=float), mu_target, float(seed_row["mu_solved"]), p)))
        guess_bank.extend(_ordered_seed_guess_bank(target_row, fit_rows, mu_target))
        deduped = []
        seen = set()
        for seed_source, guess in guess_bank:
            key = tuple(round(float(v), 8) for v in np.asarray(guess, dtype=float).tolist())
            if key in seen:
                continue
            seen.add(key)
            deduped.append((seed_source, guess))
        for seed_source, guess in deduped[:8]:
            out = solve_anisotropic(mu_target, p, guess=guess)
            row, _ = _attempt_row(
                point=point,
                p=p,
                stage="direct_fixed_mu",
                seed_source=seed_source,
                mu_target=mu_target,
                family_id=None,
                basin_label="direct_fixed_mu",
                branch=out,
                guess=guess,
            )
            rows.append(row)
    return rows


def _trivial_row(point: Dict, mu_target: float, mu_max: float) -> Dict:
    row = {
        "point_key": point["point_key"],
        "delta2": point["delta2"],
        "alpha_dm": point["alpha_dm"],
        "mu_X": point["mu_X"],
        "stage": "trivial_branch",
        "seed_source": "exact_isotropic",
        "family_id": None,
        "basin_label": "exact_isotropic",
        "mu_target": float(mu_target),
        "success": False,
        "message": "trivial branch above isotropic extremality bound",
        "guess_params": None,
        "mu_solved": float(mu_target),
        "temperature": None,
        "source": 0.0,
        "residual_inf": 0.0,
        "vev": 0.0,
        "Omega_ren": None,
        "rho": None,
        "rho_d": None,
        "accepted_ordered": False,
        "w0_solved": 0.0,
        "params5": None,
        "physical_trivial_branch": False,
        "mu_extremal_max": float(mu_max),
    }
    if mu_target >= mu_max - 1.0e-4:
        return row
    iso = bundle_branch(mu_target, point["alpha_dm"], delta2=point["delta2"], nu_target=point["mu_X"], branch_type="iso")
    if iso.get("success") and finite_float(iso.get("temperature")) is not None and float(iso["temperature"]) > 0.0:
        row.update({
            "success": True,
            "message": "",
            "temperature": float(iso["temperature"]),
            "Omega_ren": float(iso["Omega_ren"]),
            "rho": float(iso["rho"]),
            "rho_d": float(iso["rho_d"]),
            "physical_trivial_branch": True,
        })
    return row


def summarise_extremality_point(target_row: Dict, mu_targets: Sequence[float], mu_max: float, rows: Sequence[Dict]) -> Tuple[Dict, List[Dict], List[Dict]]:
    point = point_dict_from_row(target_row)
    per_mu_rows: List[Dict] = []
    coexistence_rows: List[Dict] = []
    max_ordered = 0
    mu_with_ordered = 0
    mu_with_coexistence = 0
    ordered_below_ext = 0
    ordered_above_ext = 0

    for mu_target in mu_targets:
        same_mu = [row for row in rows if row.get("accepted_ordered") and row.get("mu_target") is not None and abs(float(row["mu_target"]) - mu_target) < 5.0e-7]
        unique_ordered = dedupe_ordered_branches(same_mu)
        ordered_count = len(unique_ordered)
        trivial_exists = bool(mu_target < mu_max - 1.0e-4)
        if ordered_count > 0:
            mu_with_ordered += 1
            if mu_target < mu_max - 1.0e-4:
                ordered_below_ext += 1
            else:
                ordered_above_ext += 1
        if ordered_count >= 2:
            mu_with_coexistence += 1
            coexistence_rows.append({
                "point_key": point["point_key"],
                "delta2": point["delta2"],
                "alpha_dm": point["alpha_dm"],
                "mu_X": point["mu_X"],
                "mu": float(mu_target),
                "ordered_branch_count": ordered_count,
                "ordered_vev_values": [float(row["vev"]) for row in unique_ordered],
                "ordered_Omega_values": [float(row["Omega_ren"]) for row in unique_ordered],
            })
        max_ordered = max(max_ordered, ordered_count)
        per_mu_rows.append({
            "point_key": point["point_key"],
            "delta2": point["delta2"],
            "alpha_dm": point["alpha_dm"],
            "mu_X": point["mu_X"],
            "mu": float(mu_target),
            "mu_extremal_max": float(mu_max),
            "trivial_branch_exists": trivial_exists,
            "ordered_branch_count": ordered_count,
            "ordered_vev_values": [float(row["vev"]) for row in unique_ordered],
            "ordered_Omega_values": [float(row["Omega_ren"]) for row in unique_ordered],
        })

    classification = "no_clean_ordered_branch_found"
    if max_ordered >= 2:
        classification = "ordered_ordered_coexistence_found"
    elif mu_with_ordered > 0:
        classification = "single_ordered_branch_only"

    point_summary = {
        "point_key": point["point_key"],
        "delta2": point["delta2"],
        "alpha_dm": point["alpha_dm"],
        "mu_X": point["mu_X"],
        "mu_extremal_max": float(mu_max),
        "mu_scan_count": len(mu_targets),
        "mu_with_ordered_branch": mu_with_ordered,
        "mu_with_ordered_branch_below_extremality": ordered_below_ext,
        "mu_with_ordered_branch_above_extremality": ordered_above_ext,
        "mu_with_ordered_ordered_coexistence": mu_with_coexistence,
        "max_ordered_branches_same_mu": max_ordered,
        "classification": classification,
        "claim2_status": "positive_closure" if max_ordered >= 2 else "negative_evidence_only",
    }
    return point_summary, per_mu_rows, coexistence_rows


def write_extremality_report(output_path: Path, summary: Dict, point_rows: Sequence[Dict]) -> None:
    lines = [
        "# D5 Extremality-Regime Report",
        "",
        f"- Extremality-limited points scanned: {summary['point_count']}.",
        f"- Points with any clean ordered branch in the targeted fixed-mu search: {summary['points_with_ordered_branch']}.",
        f"- Points with two distinct clean ordered branches at the same mu: {summary['points_with_ordered_ordered_coexistence']}.",
        f"- Maximum number of distinct ordered branches at the same mu seen anywhere: {summary['max_ordered_branches_same_mu']}.",
        "",
        "## Point outcomes",
    ]
    for row in point_rows:
        lines.append(
            "- "
            f"{row['point_key']}: {row['classification']}, "
            f"mu_ext,max={row['mu_extremal_max']:.6f}, "
            f"mu-with-ordered={row['mu_with_ordered_branch']}, "
            f"mu-with-ordered-ordered={row['mu_with_ordered_ordered_coexistence']}."
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_extremality_regime_search(output_dir: Path, refined_claim3_result: Dict) -> Dict:
    output_dir = ensure_dir(output_dir)
    fit_rows = list(refined_claim3_result["fit_rows"])
    target_rows = [row for row in fit_rows if is_extremality_limited_row(row)]

    seed_harvest_rows: List[Dict] = []
    family_rows: List[Dict] = []
    direct_rows: List[Dict] = []
    trivial_rows: List[Dict] = []
    point_rows: List[Dict] = []
    mu_rows: List[Dict] = []
    coexistence_rows: List[Dict] = []

    for target_row in target_rows:
        harvest = harvest_extremality_seed_branches(target_row, fit_rows)
        seed_rows = harvest["seed_rows"]
        mu_max = float(harvest["mu_extremal_max"])
        seed_harvest_rows.extend(harvest["attempt_rows"])

        mu_targets = build_extremality_mu_targets(mu_max, seed_rows)
        if not mu_targets:
            mu_targets = [round(mu_max * frac, 6) for frac in EXTREMALITY_MU_FRACTIONS]

        point = point_dict_from_row(target_row)
        for mu_target in mu_targets:
            trivial_rows.append(_trivial_row(point, mu_target, mu_max))

        for seed_row in seed_rows:
            family_rows.extend(continue_seed_family(target_row, seed_row, mu_targets))
        direct_rows.extend(direct_fixed_mu_scan(target_row, fit_rows, mu_targets, seed_rows))

        all_rows = [row for row in family_rows + direct_rows if row["point_key"] == point["point_key"]]
        point_summary, per_mu, per_mu_coexistence = summarise_extremality_point(target_row, mu_targets, mu_max, all_rows)
        point_summary["seed_branch_count"] = len(seed_rows)
        point_rows.append(point_summary)
        mu_rows.extend(per_mu)
        coexistence_rows.extend(per_mu_coexistence)

        json_dump(output_dir / f"{point['point_key']}_partial.json", {
            "point": point_summary,
            "seed_rows": seed_rows,
            "mu_rows": per_mu,
            "coexistence_rows": per_mu_coexistence,
        })

    point_rows.sort(key=lambda row: (float(row["delta2"]), float(row["alpha_dm"]), float(row["mu_X"])))
    summary = {
        "point_count": len(point_rows),
        "points_with_ordered_branch": sum(1 for row in point_rows if row["mu_with_ordered_branch"] > 0),
        "points_with_ordered_ordered_coexistence": sum(1 for row in point_rows if row["mu_with_ordered_ordered_coexistence"] > 0),
        "max_ordered_branches_same_mu": max((int(row["max_ordered_branches_same_mu"]) for row in point_rows), default=0),
    }

    csv_dump(output_dir / "extremality_seed_harvest.csv", seed_harvest_rows)
    csv_dump(output_dir / "extremality_family_continuation.csv", family_rows)
    csv_dump(output_dir / "extremality_direct_fixed_mu.csv", direct_rows)
    csv_dump(output_dir / "extremality_trivial_branches.csv", trivial_rows)
    csv_dump(output_dir / "extremality_point_summary.csv", point_rows)
    csv_dump(output_dir / "extremality_mu_summary.csv", mu_rows)
    csv_dump(output_dir / "extremality_coexistence_rows.csv", coexistence_rows)
    json_dump(output_dir / "extremality_report.json", {
        "summary": summary,
        "point_rows": point_rows,
        "mu_rows": mu_rows,
        "coexistence_rows": coexistence_rows,
    })
    write_extremality_report(output_dir / "extremality_report.md", summary, point_rows)
    return {
        "summary": summary,
        "point_rows": point_rows,
        "mu_rows": mu_rows,
        "coexistence_rows": coexistence_rows,
    }
