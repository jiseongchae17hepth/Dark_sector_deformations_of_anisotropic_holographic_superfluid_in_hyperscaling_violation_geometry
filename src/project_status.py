from pathlib import Path
from typing import Dict, Optional


def write_project_status_table(
    output_path: Path,
    renorm_summary: Dict,
    claim3_refined: Dict,
    extremality_result: Dict,
    hsv_result: Dict,
    policy_result: Optional[Dict] = None,
    d4_refined_result: Optional[Dict] = None,
    d4_claim2_refined_result: Optional[Dict] = None,
    hee_closure_result: Optional[Dict] = None,
) -> None:
    d5_fit_summary = claim3_refined["summary"]
    ext_summary = extremality_result["summary"]
    hsv_summary = hsv_result.get("summary", {})
    d4_validation = hsv_result.get("d4_validation", {}).get("summary", {})
    d4_analytic_control = hsv_result.get("d4_analytic_control", {}).get("summary", {})
    d4_claim3 = (d4_refined_result or hsv_result.get("d4_claim3", {})).get("summary", {})
    d4_claim2 = (d4_claim2_refined_result or hsv_result.get("d4_claim2", {})).get("summary", {})
    d4_hee = hsv_result.get("d4_hee", {}).get("summary", {})
    d3_summary = hsv_result.get("d3", {}).get("summary", {})
    policy_summary = (policy_result or {}).get("summary", {})
    hee_summary = hee_closure_result or {}
    hee_d4 = hee_summary.get("d4_analysis", {})
    hee_d5 = hee_summary.get("d5_controls", {})

    lines = [
        "# Project Claim Status",
        "",
        "| Claim | Status | Current reading |",
        "| --- | --- | --- |",
        "| Claim 1 | positive closure (revised) | The revised piecewise deformation law is the closed statement: continuous deformation of Sigma_c is kept, but the sign of the onset shift depends on mu_X relative to 4. |",
        "| Claim 2 | still open | D=5 gives strong negative evidence and the corrected D=4 HSV classification now resolves all 18 scanned points into 17 `no stationary ordered branch` cases plus 1 `outside validity domain` case, with 0 same-mu ordered-ordered coexistence; the global obstruction is the missing corrected nonlinear D=3 dark-HSV branch solver. |",
        "| Claim 3 | still open | In corrected D=5, all 144 fit-closed onset-connected points have c2 > 0. The corrected D=4 HSV classification now resolves all 18 scanned points but yields 0 trustworthy small-branch fits, so the remaining global obstruction is again the missing corrected nonlinear D=3 dark-HSV branch solver. |",
        "| Claim 4 | still open | Actual corrected D=4 effective `O12^EE` data and D5 control HEE data now exist, and the D4 effective dark response is mostly compatible with an `alpha_dm^2` leading correction on the tested slices, but a corrected full/global HEE closure is still missing. |",
        "",
        "## D5 Snapshot",
        f"- Corrected onset-connected fit-closed points: {d5_fit_summary['claim3_fit_success_points']} / {d5_fit_summary['total_points']}.",
        f"- Positive c2 points in that resolved D=5 window: {d5_fit_summary['positive_c2_points']}.",
        f"- Extremality-limited points scanned in the dedicated project: {ext_summary['point_count']}.",
        f"- Extremality points with any clean ordered branch: {ext_summary['points_with_ordered_branch']}.",
        f"- Extremality points with ordered-ordered coexistence at the same mu: {ext_summary['points_with_ordered_ordered_coexistence']}.",
        "- In the dedicated extremality scan, the clean ordered branches that were found sit only above the isotropic normal-branch extremality bound.",
        "",
        "## HSV Snapshot",
        f"- D=4 corrected thermodynamics closed on the stored effective validation set: {hsv_summary.get('d4_corrected_thermodynamics_closed')}.",
        f"- D=4 corrected thermo validation cases: {d4_validation.get('case_count')}, cutoff-stable cases: {d4_validation.get('cutoff_stable_case_count')}.",
        f"- D=4 perturbative analytic-control max cutoff spread: {d4_analytic_control.get('max_omega_spread')}.",
        f"- D=4 Claim 3 trustworthy fit-closed points: {d4_claim3.get('fit_closed_points', d4_claim3.get('fit_success_points'))} / {d4_claim3.get('point_count')}.",
        f"- D=4 points with no stationary ordered minimum on the scanned corrected slices: {d4_claim3.get('no_stationary_points', d4_claim3.get('no_stationary_branch_points'))}.",
        f"- D=4 points classified as outside the current validity domain after fallback refinement: {d4_claim3.get('outside_validity_points', 0)}.",
        f"- D=4 same-mu ordered-ordered coexistence points: {d4_claim2.get('same_mu_ordered_ordered_coexistence_points')}.",
        f"- D=4 actual O12_EE width rows exported: {d4_hee.get('accepted_width_rows')} / {d4_hee.get('width_row_count')}.",
        f"- D=4 small-width HEE fit count: {hee_d4.get('small_width_fit_count', d4_hee.get('width_fit_count'))}.",
        f"- D=3 analytic thermodynamic diagnostics closed: {d3_summary.get('corrected_thermodynamics_closed_for_analytic_backgrounds')}.",
        f"- D=3 perturbative analytic-control max cutoff spread: {d3_summary.get('max_omega_spread')}.",
        f"- D=3 nonlinear dark-HSV branch solver closed: {d3_summary.get('nonlinear_dark_branch_solver_closed')}.",
    ]
    for item in hsv_result.get("lines", []):
        lines.append(f"- {item}")
    if hee_d4:
        lines.extend([
            "",
            "## HEE Snapshot",
            f"- D=4 effective HEE response-fit slices analyzed: {hee_d4.get('response_fit_count')}.",
            f"- D=4 slices compatible with an alpha_dm^2 leading response: {hee_d4.get('quadratic_response_supported_count')} ({hee_d4.get('quadratic_response_supported_fraction')}).",
            f"- D=4 isotropic-limit max |O12_EE|: {hee_d4.get('isotropic_zero_check_max_abs_o12')}.",
        ])
    if hee_d5:
        lines.append(f"- D5 control HEE width rows computed: {hee_d5.get('width_row_count')}, isotropic max |O12_EE|: {hee_d5.get('isotropic_zero_check_max_abs_o12')}.")
    if policy_summary:
        lines.extend([
            "",
            "## (alpha,z) Policy",
            f"- Analytic anchors/islands tracked explicitly in code: {policy_summary.get('anchor_count')}.",
            f"- Literature baseline box sample points tracked explicitly in code: {policy_summary.get('baseline_box_count')}.",
            f"- Baseline-box points satisfying NEC inside that policy table: {policy_summary.get('baseline_box_nec_allowed_count')}.",
            f"- Current final physical-subset points with full closure flags: {policy_summary.get('final_physical_subset_count')}.",
            f"- Policy statement: {policy_summary.get('policy_statement')}",
        ])
    lines.extend([
        "",
        "## Cutoff Validation Snapshot",
    ])
    for block, values in renorm_summary.items():
        lines.append(f"- {block}: {values}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
