from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from analysis_workflows import csv_dump, ensure_dir, json_dump


LITERATURE_ALPHA_VALUES: Sequence[float] = tuple(float(v) for v in range(0, 5))
LITERATURE_Z_VALUES: Sequence[float] = tuple(float(v) for v in range(1, 5))


@dataclass
class AlphaZAnchor:
    label: str
    dimension: int
    alpha_hv: float
    z: float
    family: str
    note: str


ANALYTIC_ANCHORS: Sequence[AlphaZAnchor] = (
    AlphaZAnchor("D5 AdS anchor", 5, 0.0, 1.0, "analytic_anchor", "Corrected D=5 backreacted anchor already scanned."),
    AlphaZAnchor("D4 HSV island", 4, 0.5, 1.0, "analytic_island", "Exact reduced onset law plus corrected effective thermodynamics."),
    AlphaZAnchor("D3 HSV island", 3, 2.0, 1.0, "analytic_island", "Perturbative analytic island only; nonlinear dark branch still missing."),
)


def nec_allowed(alpha_hv: float, z: float) -> bool:
    return (alpha_hv + 1.0) * (alpha_hv + z - 1.0) >= 0.0


def probe_sl_window(alpha_hv: float, z: float) -> bool:
    return (
        5.0 * z + 3.0 * alpha_hv > 1.0
        and 3.0 * alpha_hv + z > -3.0
        and 4.0 * z + 6.0 * alpha_hv > -2.0
        and z + alpha_hv > -1.0
    )


def _anchor_status(anchor: AlphaZAnchor, hsv_result: Optional[Dict]) -> Dict[str, object]:
    d4_closed = None
    d3_analytic_closed = None
    d3_nonlinear_closed = None
    if hsv_result:
        d4_closed = bool(hsv_result.get("summary", {}).get("d4_corrected_thermodynamics_closed"))
        d3_analytic_closed = bool(hsv_result.get("summary", {}).get("d3_analytic_thermodynamics_closed"))
        d3_nonlinear_closed = bool(hsv_result.get("summary", {}).get("d3_nonlinear_branch_solver_closed"))

    if anchor.dimension == 5:
        return {
            "regular_known": True,
            "extremality_consistent": True,
            "corrected_thermodynamics_closed": True,
            "nonlinear_branch_solver_closed": True,
            "final_physical_subset": True,
        }
    if anchor.dimension == 4:
        return {
            "regular_known": True,
            "extremality_consistent": True,
            "corrected_thermodynamics_closed": bool(d4_closed),
            "nonlinear_branch_solver_closed": False,
            "final_physical_subset": False,
        }
    return {
        "regular_known": True,
        "extremality_consistent": False,
        "corrected_thermodynamics_closed": bool(d3_analytic_closed),
        "nonlinear_branch_solver_closed": bool(d3_nonlinear_closed),
        "final_physical_subset": False,
    }


def build_alpha_z_policy_rows(hsv_result: Optional[Dict] = None) -> List[Dict]:
    rows: List[Dict] = []
    for anchor in ANALYTIC_ANCHORS:
        status = _anchor_status(anchor, hsv_result)
        rows.append({
            "row_type": "anchor",
            "label": anchor.label,
            "dimension": anchor.dimension,
            "alpha_hv": anchor.alpha_hv,
            "z": anchor.z,
            "family": anchor.family,
            "nec_allowed": nec_allowed(anchor.alpha_hv, anchor.z),
            "probe_sl_window": probe_sl_window(anchor.alpha_hv, anchor.z),
            **status,
            "note": anchor.note,
        })

    for alpha_hv in LITERATURE_ALPHA_VALUES:
        for z in LITERATURE_Z_VALUES:
            rows.append({
                "row_type": "baseline_box",
                "label": f"baseline_alpha{alpha_hv:.1f}_z{z:.1f}",
                "dimension": "",
                "alpha_hv": float(alpha_hv),
                "z": float(z),
                "family": "literature_box",
                "nec_allowed": nec_allowed(float(alpha_hv), float(z)),
                "probe_sl_window": probe_sl_window(float(alpha_hv), float(z)),
                "regular_known": None,
                "extremality_consistent": None,
                "corrected_thermodynamics_closed": False,
                "nonlinear_branch_solver_closed": False,
                "final_physical_subset": False,
                "note": "Baseline probe-literature box. Promotion to the final physical subset requires regularity, extremality consistency, and corrected thermodynamic closure.",
            })
    return rows


def build_alpha_z_policy_summary(rows: Sequence[Dict]) -> Dict[str, object]:
    anchors = [row for row in rows if row["row_type"] == "anchor"]
    baseline = [row for row in rows if row["row_type"] == "baseline_box"]
    return {
        "anchor_count": len(anchors),
        "baseline_box_count": len(baseline),
        "baseline_box_nec_allowed_count": sum(1 for row in baseline if row["nec_allowed"]),
        "baseline_box_probe_window_count": sum(1 for row in baseline if row["probe_sl_window"]),
        "final_physical_subset_count": sum(1 for row in anchors if row["final_physical_subset"]),
        "policy_statement": "The project now treats the (alpha,z) exploration as a three-stage policy: analytic anchors/islands first, the literature baseline box second, and only NEC-allowed + regular + extremality-consistent + corrected-thermodynamics-closed points as the final physical subset.",
    }


def write_alpha_z_policy_outputs(output_dir: Path, hsv_result: Optional[Dict] = None) -> Dict:
    output_dir = ensure_dir(output_dir)
    rows = build_alpha_z_policy_rows(hsv_result=hsv_result)
    summary = build_alpha_z_policy_summary(rows)
    csv_dump(output_dir / "alpha_z_policy.csv", rows)
    json_dump(output_dir / "alpha_z_policy.json", {"summary": summary, "rows": rows, "anchors": [asdict(anchor) for anchor in ANALYTIC_ANCHORS]})
    return {"summary": summary, "rows": rows}
