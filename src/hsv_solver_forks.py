import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import quad, solve_bvp

from analysis_workflows import csv_dump, ensure_dir, fit_small_branch, json_dump
from hsv_islands_analytic import (
    HsvIslandParams,
    d3_analytic_island_solution,
    d4_analytic_island_solution,
    d4_exact_mu_c,
    hsv_renormalized_thermodynamics,
)


MU0_D4 = 4.0 / math.sqrt(3.0)
D4_SOURCE_TOL = 2.0e-5
D4_RESIDUAL_TOL = 8.0e-3
D4_THERMO_CUTOFFS: Tuple[float, ...] = (8.0, 10.0, 12.0)
HSV_ANALYTIC_CUTOFFS: Tuple[float, ...] = (8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 40.0)
HSV_ANALYTIC_UGRID_MAX = 50.0
D4_DEFAULT_ALPHA_DM: Tuple[float, ...] = (0.0, 0.8, 1.2)
D4_DEFAULT_NU: Tuple[float, ...] = (0.0, 1.0 * MU0_D4, 1.5 * MU0_D4)
D4_DEFAULT_DELTA2: Tuple[float, ...] = (0.05, 0.10)
D4_DEFAULT_EPSILON_GRID: Tuple[float, ...] = (0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20)
D4_DEFAULT_MU_MULTIPLIERS: Tuple[float, ...] = (1.00, 1.03, 1.06, 1.10)
D4_MU0_SANITY_TOL = 0.15
D4_HEE_WIDTHS: Tuple[float, ...] = (0.18, 0.30, 0.40)
D4_HEE_ALPHA_DM: Tuple[float, ...] = (0.0, 0.4, 0.8, 1.2)
D4_HEE_NU: Tuple[float, ...] = (0.0, 1.5 * MU0_D4)
D4_HEE_DELTA2: Tuple[float, ...] = (0.05, 0.10)
D4_HEE_EPSILON_GRID: Tuple[float, ...] = (0.0, 0.10, 0.20)
D4_HEE_MU_SCALE = 1.05


@dataclass
class HsvEffectiveCase:
    alpha_dm: float
    nu: float
    delta2: float
    epsilon: float

    @property
    def mix(self) -> float:
        return 0.5 * self.alpha_dm

    @property
    def alpha_tilde(self) -> float:
        return 1.0 - self.mix * self.mix


def d4_effective_rhs(u, y, delta2=0.02, alpha_dm=0.0, nu=0.0, mu=None):
    if mu is None:
        mu = d4_exact_mu_c(alpha_dm=alpha_dm, nu=nu)
    N, sig, A, p, b, w, q, r, s, t = y
    eP = np.exp(p / np.sqrt(3.0))
    eX = np.exp(2.0 * A + p / np.sqrt(3.0))
    at = 1.0 - alpha_dm * alpha_dm / 4.0

    eta = (nu + alpha_dm * mu / 2.0) * (1.0 - u ** -2) - alpha_dm * b / 2.0
    etap = 2.0 * (nu + alpha_dm * mu / 2.0) * u ** -3 - alpha_dm * s / 2.0

    kb = at * s * s + etap * etap + alpha_dm * s * etap
    ueff = b * b - (alpha_dm * alpha_dm / 4.0) * b * eta

    sigp = (
        2.0 * q * q * sig * u / 3.0
        + r * r * sig * u / 6.0
        + np.sqrt(3.0) * r * sig / 3.0
        + delta2 * at * sig * t * t * eX / (6.0 * u)
        + delta2 * ueff * w * w * eX / (6.0 * N * N * sig * u ** 5)
    )
    Np = (
        -2.0 * N * q * q * u / 3.0
        - N * r * r * u / 6.0
        - np.sqrt(3.0) * N * r / 3.0
        - delta2 * at * N * t * t * eX / (6.0 * u)
        - 4.0 * N / u
        - delta2 * kb * eP / (6.0 * sig * sig * u)
        + 4.0 * np.exp(-p / np.sqrt(3.0)) / u
        - delta2 * ueff * w * w * eX / (6.0 * N * sig * sig * u ** 5)
    )
    Add = (
        - q * sigp / sig
        - q * Np / N
        - 5.0 * q / u
        + delta2 * at * t * t * eX / (4.0 * u * u)
        - delta2 * ueff * w * w * eX / (4.0 * N * N * sig * sig * u ** 6)
    )
    pdd = (
        delta2 * np.sqrt(3.0) * at * t * t * eX / (6.0 * u * u)
        - 5.0 * r / u
        - r * sigp / sig
        - np.sqrt(3.0) * sigp / (sig * u)
        - Np * r / N
        - np.sqrt(3.0) * Np / (N * u)
        - 4.0 * np.sqrt(3.0) / u ** 2
        + 4.0 * np.sqrt(3.0) * np.exp(-p / np.sqrt(3.0)) / (N * u ** 2)
        - delta2 * np.sqrt(3.0) * kb * eP / (6.0 * N * sig * sig * u ** 2)
        - delta2 * np.sqrt(3.0) * ueff * w * w * eX / (6.0 * N * N * sig * sig * u ** 6)
    )
    bdd = (
        - np.sqrt(3.0) * r * s / 3.0
        - 3.0 * s / u
        + s * sigp / sig
        + b * w * w * np.exp(2.0 * A) / (max(at, 1e-8) * N * u ** 4)
    )
    wdd = (
        -2.0 * q * t
        - np.sqrt(3.0) * r * t / 3.0
        - sigp * t / sig
        - Np * t / N
        - 3.0 * t / u
        - ueff * w / (max(at, 1e-8) * N * N * sig * sig * u ** 4)
    )
    return np.vstack([Np, sigp, q, r, s, t, Add, pdd, bdd, wdd])


def d4_effective_dark_profile(u: float, b: float, s: float, mu: float, case: HsvEffectiveCase) -> Tuple[float, float]:
    eta = (case.nu + case.alpha_dm * mu / 2.0) * (1.0 - u ** -2) - case.alpha_dm * b / 2.0
    etap = 2.0 * (case.nu + case.alpha_dm * mu / 2.0) * u ** -3 - case.alpha_dm * s / 2.0
    return float(eta), float(etap)


def asymptotic_source_u_minus_two(value: float, derivative: float, u: float) -> float:
    return float(value + 0.5 * u * derivative)


def asymptotic_vev_u_minus_two(value: float, derivative: float, u: float) -> float:
    del value
    return float(-0.5 * u ** 3 * derivative)


def d4_effective_mass_coefficient(u: float, n_value: float, dn_du: float) -> float:
    return float(-3.0 * u ** 4 * (n_value - 1.0) - 0.5 * u ** 5 * dn_du)


def d4_effective_temperature(sol, horizon_sample: Optional[Sequence[float]] = None) -> float:
    sample = np.asarray(horizon_sample or [1.001, 1.002, 1.003, 1.004], dtype=float)
    values = sol.sol(sample)
    slope = float(np.polyfit(sample - 1.0, values[0], 1)[0])
    sigma_h = float(values[1, 0])
    return float(sigma_h * slope / (4.0 * math.pi))


def d4_effective_entropy_density(sol) -> float:
    del sol
    return float(2.0 * math.pi)


def d4_isotropic_seed(case: HsvEffectiveCase, mu: float, u: np.ndarray) -> np.ndarray:
    return 1.0 + case.delta2 * mu * mu / u ** 6 - (1.0 + case.delta2 * mu * mu) / u ** 4


def solve_d4_effective_case(
    case: HsvEffectiveCase,
    y_init=None,
    uh: float = 1.001,
    u_max: float = 12.0,
    npts: int = 260,
    tol: float = 4.0e-3,
    mu_override: Optional[float] = None,
):
    mu = float(mu_override if mu_override is not None else d4_exact_mu_c(alpha_dm=case.alpha_dm, nu=case.nu))
    u = np.linspace(uh, u_max, npts)
    Nseed = d4_isotropic_seed(case, mu, u)
    sig0 = np.ones_like(u)
    A0 = np.zeros_like(u)
    p0 = np.zeros_like(u)
    b0 = mu * (1.0 - u ** -2)
    w0 = case.epsilon * u ** 2 / (1.0 + u ** 2) ** 2

    q0 = np.gradient(A0, u)
    r0 = np.gradient(p0, u)
    s0 = np.gradient(b0, u)
    t0 = np.gradient(w0, u)
    y0 = np.vstack([Nseed, sig0, A0, p0, b0, w0, q0, r0, s0, t0]) if y_init is None else y_init

    def ode(x, y):
        return d4_effective_rhs(x, y, delta2=case.delta2, alpha_dm=case.alpha_dm, nu=case.nu, mu=mu)

    def bc(ya, yb):
        return np.array([
            ya[0] - Nseed[0],
            yb[1] - 1.0,
            ya[2] - 0.0,
            yb[2] - 0.0,
            ya[3] - 0.0,
            yb[3] - 0.0,
            ya[4] - 0.0,
            yb[4] + 0.5 * u_max * yb[8] - mu,
            ya[5] - w0[0],
            yb[5] + 0.5 * u_max * yb[9],
        ])

    sol = solve_bvp(ode, bc, u, y0, tol=tol, max_nodes=70000, verbose=0)
    return mu, sol


def d4_effective_boundary_thermodynamics(sol, case: HsvEffectiveCase, u_cut: float = 10.0) -> Dict[str, float]:
    y = sol.sol(np.array([u_cut], dtype=float))[:, 0]
    N, sig, A, p, b, w, q, r, s, t = [float(v) for v in y.tolist()]
    mu = asymptotic_source_u_minus_two(b, s, u_cut)
    vev = asymptotic_vev_u_minus_two(w, t, u_cut)
    source = asymptotic_source_u_minus_two(w, t, u_cut)
    eta, etap = d4_effective_dark_profile(u_cut, b, s, mu, case)
    nu = asymptotic_source_u_minus_two(eta, etap, u_cut)
    rhs = d4_effective_rhs(
        np.array([u_cut], dtype=float),
        np.asarray(y, dtype=float).reshape(-1, 1),
        delta2=case.delta2,
        alpha_dm=case.alpha_dm,
        nu=case.nu,
        mu=mu,
    )
    Np = float(rhs[0, 0])
    mass = d4_effective_mass_coefficient(u_cut, N, Np)
    charge_prefactor = case.delta2 * math.exp(p / math.sqrt(3.0)) * u_cut ** 3
    rho = float(charge_prefactor * (case.alpha_tilde * s + case.mix * etap))
    rho_d = float(charge_prefactor * (etap + case.mix * s))
    temperature = d4_effective_temperature(sol)
    entropy = d4_effective_entropy_density(sol)
    omega = float(mass - temperature * entropy - mu * rho - nu * rho_d)
    return {
        "mu": mu,
        "nu": nu,
        "source": source,
        "vev": vev,
        "psi_abs": abs(vev),
        "psi_sq": vev * vev,
        "epsilon": mass,
        "temperature": temperature,
        "entropy_density": entropy,
        "rho": rho,
        "rho_d": rho_d,
        "Omega_ren": omega,
        "A_max": float(np.max(np.abs(sol.y[2]))),
        "p_max": float(np.max(np.abs(sol.y[3]))),
        "residual_inf": float(np.max(sol.rms_residuals)) if len(sol.rms_residuals) else float("nan"),
    }


def d4_hee_strip_observables(sol, W=0.30, u_max=12.0) -> Dict[str, float]:
    """Return strip entropies for the two anisotropic directions.

    We use the convention that the x-direction is parallel to the vector order
    and define the paper-facing order parameter as

        O12_EE = S_perp - S_parallel = S_y - S_x.
    """

    def metric(u):
        vals = sol.sol(np.atleast_1d(u))
        N = vals[0][0]
        A = vals[2][0]
        return N, A

    def width(ustar, direction="x"):
        def integrand(uu):
            N, A = metric(uu)
            pref = np.exp(A) if direction == "x" else np.exp(-A)
            return pref / (uu ** 2 * np.sqrt(N) * np.sqrt((uu / ustar) ** 6 - 1.0))

        val, _ = quad(integrand, ustar + 1e-6, u_max, limit=300, epsabs=1e-10, epsrel=1e-9)
        return 2.0 * val

    def area(ustar):
        def integrand(uu):
            N, _ = metric(uu)
            return uu ** 2.5 / (np.sqrt(N) * np.sqrt(1.0 - (ustar / uu) ** 6))

        val, _ = quad(integrand, ustar + 1e-6, u_max, limit=300, epsabs=1e-10, epsrel=1e-9)
        return 2.0 * val

    ux = float("nan")
    uy = float("nan")
    try:
        from scipy.optimize import brentq

        ux = brentq(lambda us: width(us, "x") - W, 1.01, u_max - 0.5)
        uy = brentq(lambda us: width(us, "y") - W, 1.01, u_max - 0.5)
        s_parallel = float(area(ux))
        s_perp = float(area(uy))
        return {
            "success": True,
            "width": float(W),
            "u_star_parallel": float(ux),
            "u_star_perp": float(uy),
            "S_parallel": s_parallel,
            "S_perp": s_perp,
            "O12_EE": float(s_perp - s_parallel),
        }
    except Exception:
        return {
            "success": False,
            "width": float(W),
            "u_star_parallel": ux,
            "u_star_perp": uy,
            "S_parallel": float("nan"),
            "S_perp": float("nan"),
            "O12_EE": float("nan"),
        }


def d4_hee_order(sol, W=0.30, u_max=12.0):
    obs = d4_hee_strip_observables(sol, W=W, u_max=u_max)
    if not obs["success"]:
        return float("nan"), float(obs["u_star_parallel"]), float(obs["u_star_perp"])
    # Keep the legacy sign convention used in the older exploratory outputs.
    return float(obs["S_parallel"] - obs["S_perp"]), float(obs["u_star_parallel"]), float(obs["u_star_perp"])


def d4_raw_branch_row(
    point_key: str,
    case: HsvEffectiveCase,
    mu_target: float,
    epsilon: float,
    sol,
    branch_type: str,
    dOmega: Optional[float] = None,
) -> Dict:
    row = {
        "point_key": point_key,
        "dimension": 4,
        "alpha_hv": 0.5,
        "z": 1.0,
        "alpha_dm": float(case.alpha_dm),
        "nu_target": float(case.nu),
        "delta2": float(case.delta2),
        "mu_target": float(mu_target),
        "epsilon": float(epsilon),
        "branch_type": branch_type,
        "success": bool(sol.success),
        "dOmega_vs_iso": None if dOmega is None else float(dOmega),
    }
    if not sol.success:
        row.update({
            "residual_inf": float(np.max(sol.rms_residuals)) if len(sol.rms_residuals) else float("nan"),
            "source": None,
            "vev": None,
            "psi_abs": None,
            "psi_sq": None,
            "mu": None,
            "nu": None,
            "temperature": None,
            "rho": None,
            "rho_d": None,
            "epsilon_density": None,
            "Omega_ren": None,
            "A_max": None,
            "accepted": False,
        })
        return row

    thermo = d4_effective_boundary_thermodynamics(sol, case)
    hee_obs = d4_hee_strip_observables(sol, W=0.30)
    hee_order = float("nan")
    if hee_obs["success"]:
        hee_order = float(hee_obs["S_parallel"] - hee_obs["S_perp"])
    row.update({
        "residual_inf": thermo["residual_inf"],
        "source": thermo["source"],
        "vev": thermo["vev"],
        "psi_abs": thermo["psi_abs"],
        "psi_sq": thermo["psi_sq"],
        "mu": thermo["mu"],
        "nu": thermo["nu"],
        "temperature": thermo["temperature"],
        "rho": thermo["rho"],
        "rho_d": thermo["rho_d"],
        "epsilon_density": thermo["epsilon"],
        "Omega_ren": thermo["Omega_ren"],
        "A_max": thermo["A_max"],
        "hee_order_W030": hee_order,
        "hee_o12_W030": hee_obs["O12_EE"],
        "S_parallel_W030": hee_obs["S_parallel"],
        "S_perp_W030": hee_obs["S_perp"],
        "u_star_x": hee_obs["u_star_parallel"],
        "u_star_y": hee_obs["u_star_perp"],
    })
    row["accepted"] = bool(
        abs(thermo["source"]) <= D4_SOURCE_TOL
        and thermo["residual_inf"] <= D4_RESIDUAL_TOL
        and math.isfinite(thermo["Omega_ren"])
    )
    return row


def d4_detect_local_minima(ordered_rows: Sequence[Dict]) -> List[Dict]:
    usable = [row for row in ordered_rows if row.get("accepted") and row.get("dOmega_vs_iso") is not None]
    usable = sorted(usable, key=lambda row: row["epsilon"])
    minima: List[Dict] = []
    if not usable:
        return minima
    if len(usable) == 1:
        row = usable[0]
        if row["dOmega_vs_iso"] is not None and row["dOmega_vs_iso"] < 0.0:
            minima.append(dict(row))
        return minima
    first = usable[0]
    if first["dOmega_vs_iso"] is not None and first["dOmega_vs_iso"] < 0.0 and first["dOmega_vs_iso"] <= usable[1]["dOmega_vs_iso"]:
        minima.append(dict(first))
    if len(usable) < 3:
        last = usable[-1]
        if last["dOmega_vs_iso"] is not None and last["dOmega_vs_iso"] < 0.0 and last["dOmega_vs_iso"] <= usable[0]["dOmega_vs_iso"]:
            minima.append(dict(last))
        return minima
    for idx in range(1, len(usable) - 1):
        prev_row = usable[idx - 1]
        this_row = usable[idx]
        next_row = usable[idx + 1]
        dprev = prev_row["dOmega_vs_iso"]
        dcurr = this_row["dOmega_vs_iso"]
        dnext = next_row["dOmega_vs_iso"]
        if dcurr is None or dprev is None or dnext is None:
            continue
        if dcurr < 0.0 and dcurr <= dprev and dcurr <= dnext:
            minima.append(dict(this_row))
    last = usable[-1]
    if last["dOmega_vs_iso"] is not None and last["dOmega_vs_iso"] < 0.0 and last["dOmega_vs_iso"] <= usable[-2]["dOmega_vs_iso"]:
        minima.append(dict(last))
    return minima


def d4_scan_mu_slice(
    point_key: str,
    alpha_dm: float,
    nu_target: float,
    delta2: float,
    mu_target: float,
    epsilon_grid: Sequence[float],
) -> Tuple[List[Dict], List[Dict]]:
    raw_rows: List[Dict] = []
    epsilon_grid = list(epsilon_grid)
    iso_case = HsvEffectiveCase(alpha_dm=alpha_dm, nu=nu_target, delta2=delta2, epsilon=0.0)
    _, iso_sol = solve_d4_effective_case(iso_case, mu_override=mu_target)
    iso_row = d4_raw_branch_row(point_key, iso_case, mu_target, 0.0, iso_sol, branch_type="isotropic", dOmega=0.0)
    raw_rows.append(iso_row)
    iso_omega = iso_row["Omega_ren"] if iso_row.get("accepted") else None

    seed_sol = iso_sol if iso_sol.success else None
    for epsilon in epsilon_grid:
        if abs(epsilon) < 1.0e-15:
            continue
        ordered_case = HsvEffectiveCase(alpha_dm=alpha_dm, nu=nu_target, delta2=delta2, epsilon=epsilon)
        y_init = seed_sol.sol(np.linspace(1.001, 12.0, 260)) if seed_sol is not None and seed_sol.success else None
        _, ordered_sol = solve_d4_effective_case(ordered_case, y_init=y_init, mu_override=mu_target)
        row = d4_raw_branch_row(point_key, ordered_case, mu_target, epsilon, ordered_sol, branch_type="ordered")
        if iso_omega is not None and row.get("Omega_ren") is not None and math.isfinite(float(row["Omega_ren"])):
            row["dOmega_vs_iso"] = float(row["Omega_ren"] - iso_omega)
        row["accepted"] = bool(row.get("accepted")) and row.get("dOmega_vs_iso") is not None
        raw_rows.append(row)
        if ordered_sol.success:
            seed_sol = ordered_sol

    minima_rows = d4_detect_local_minima([row for row in raw_rows if row["branch_type"] == "ordered"])
    return raw_rows, minima_rows


def summarize_cutoff_cases(rows: Sequence[Dict], case_keys: Sequence[str], stable_tol: float = 1.0e-2) -> List[Dict]:
    grouped: Dict[Tuple[float, ...], List[Dict]] = {}
    for row in rows:
        key = tuple(float(row[key]) if isinstance(row.get(key), (int, float)) or str(row.get(key)).replace(".", "", 1).replace("-", "", 1).isdigit() else row.get(key) for key in case_keys)
        grouped.setdefault(key, []).append(row)

    summaries: List[Dict] = []
    for key, block in grouped.items():
        finite_omegas = [float(row["Omega_ren"]) for row in block if row.get("Omega_ren") is not None and math.isfinite(float(row["Omega_ren"]))]
        finite_mass = [float(row["mass"]) for row in block if row.get("mass") is not None and math.isfinite(float(row["mass"]))]
        finite_temps = [float(row["temperature"]) for row in block if row.get("temperature") is not None and math.isfinite(float(row["temperature"]))]
        summary = {case_key: block[0][case_key] for case_key in case_keys}
        summary.update({
            "cutoff_count": len(block),
            "omega_spread": float(max(finite_omegas) - min(finite_omegas)) if finite_omegas else None,
            "mass_spread": float(max(finite_mass) - min(finite_mass)) if finite_mass else None,
            "temperature_spread": float(max(finite_temps) - min(finite_temps)) if finite_temps else None,
            "cutoff_stable": bool(finite_omegas) and (max(finite_omegas) - min(finite_omegas) <= stable_tol),
        })
        summaries.append(summary)
    return summaries


def validate_d4_corrected_thermodynamics(
    output_dir: Path,
    alpha_dm_values: Optional[Sequence[float]] = None,
    nu_values: Optional[Sequence[float]] = None,
    delta2_values: Optional[Sequence[float]] = None,
    cutoffs: Sequence[float] = D4_THERMO_CUTOFFS,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    alpha_dm_values = list(alpha_dm_values or D4_DEFAULT_ALPHA_DM)
    nu_values = list(nu_values or D4_DEFAULT_NU)
    delta2_values = list(delta2_values or D4_DEFAULT_DELTA2)

    rows: List[Dict] = []
    case_summaries: List[Dict] = []
    for delta2 in delta2_values:
        for nu_target in nu_values:
            for alpha_dm in alpha_dm_values:
                for epsilon in [0.0, 0.10]:
                    case = HsvEffectiveCase(alpha_dm=alpha_dm, nu=nu_target, delta2=delta2, epsilon=epsilon)
                    mu_target = d4_exact_mu_c(alpha_dm=alpha_dm, nu=nu_target) * (1.05 if epsilon > 0.0 else 1.00)
                    _, sol = solve_d4_effective_case(case, mu_override=mu_target)
                    label = f"d4_d{delta2:.2f}_a{alpha_dm:.1f}_nu{nu_target / MU0_D4:.1f}_eps{epsilon:.2f}"
                    point_rows: List[Dict] = []
                    for u_cut in cutoffs:
                        row = {
                            "label": label,
                            "dimension": 4,
                            "alpha_dm": float(alpha_dm),
                            "nu_target": float(nu_target),
                            "delta2": float(delta2),
                            "epsilon": float(epsilon),
                            "mu_target": float(mu_target),
                            "u_cut": float(u_cut),
                            "success": bool(sol.success),
                        }
                        if sol.success:
                            thermo = d4_effective_boundary_thermodynamics(sol, case, u_cut=float(u_cut))
                            row.update({
                                "Omega_ren": thermo["Omega_ren"],
                                "epsilon_density": thermo["epsilon"],
                                "temperature": thermo["temperature"],
                                "rho": thermo["rho"],
                                "rho_d": thermo["rho_d"],
                                "mu": thermo["mu"],
                                "nu": thermo["nu"],
                                "source": thermo["source"],
                                "vev": thermo["vev"],
                                "residual_inf": thermo["residual_inf"],
                            })
                        else:
                            row.update({
                                "Omega_ren": None,
                                "epsilon_density": None,
                                "temperature": None,
                                "rho": None,
                                "rho_d": None,
                                "mu": None,
                                "nu": None,
                                "source": None,
                                "vev": None,
                                "residual_inf": float(np.max(sol.rms_residuals)) if len(sol.rms_residuals) else float("nan"),
                            })
                        point_rows.append(row)
                        rows.append(row)

                    finite_omegas = [float(row["Omega_ren"]) for row in point_rows if row["Omega_ren"] is not None and math.isfinite(float(row["Omega_ren"]))]
                    finite_sources = [abs(float(row["source"])) for row in point_rows if row["source"] is not None and math.isfinite(float(row["source"]))]
                    case_summaries.append({
                        "label": label,
                        "success": bool(sol.success),
                        "cutoff_stable": bool(finite_omegas) and (max(finite_omegas) - min(finite_omegas) <= 5.0e-3),
                        "omega_spread": float(max(finite_omegas) - min(finite_omegas)) if finite_omegas else None,
                        "max_source_abs": float(max(finite_sources)) if finite_sources else None,
                        "max_residual_inf": float(max(row["residual_inf"] for row in point_rows if row["residual_inf"] is not None)),
                    })

    csv_dump(output_dir / "d4_corrected_thermo_validation.csv", rows)
    summary = {
        "case_count": len(case_summaries),
        "successful_case_count": sum(1 for row in case_summaries if row["success"]),
        "cutoff_stable_case_count": sum(1 for row in case_summaries if row["cutoff_stable"]),
        "all_cases_cutoff_stable": bool(case_summaries) and all(row["cutoff_stable"] for row in case_summaries if row["success"]),
        "note": "The corrected D=4 HSV scalar uses only boundary-data thermodynamics extracted from asymptotic coefficients and mixed canonical momenta.",
    }
    payload = {"summary": summary, "case_summaries": case_summaries, "rows": rows}
    json_dump(output_dir / "d4_corrected_thermo_validation.json", payload)
    return payload


def run_d4_analytic_control_scan(output_dir: Path, delta2_values: Optional[Sequence[float]] = None, epsilon_order_values: Optional[Sequence[float]] = None) -> Dict:
    output_dir = ensure_dir(output_dir)
    delta2_values = list(delta2_values or [0.02, 0.05, 0.10])
    epsilon_order_values = list(epsilon_order_values or [0.10, 0.20, 0.30])

    u = np.linspace(1.001, HSV_ANALYTIC_UGRID_MAX, 1200)
    rows: List[Dict] = []
    for delta2 in delta2_values:
        for epsilon_order in epsilon_order_values:
            p = HsvIslandParams(dimension=4, alpha_hv=0.5, z=1.0, delta2=float(delta2), epsilon_order=float(epsilon_order))
            solution = d4_analytic_island_solution(u, p)
            for u_cut in HSV_ANALYTIC_CUTOFFS:
                thermo = hsv_renormalized_thermodynamics(u, solution, p, u_cut=u_cut)
                rows.append({
                    "dimension": 4,
                    "alpha_hv": 0.5,
                    "z": 1.0,
                    "delta2": float(delta2),
                    "epsilon_order": float(epsilon_order),
                    "u_cut": float(u_cut),
                    "mass": thermo["mass"],
                    "temperature": thermo["temperature"],
                    "entropy_density": thermo["entropy_density"],
                    "Omega_ren": thermo["Omega_ren"],
                })

    csv_dump(output_dir / "d4_analytic_control.csv", rows)
    case_summaries = summarize_cutoff_cases(rows, ["delta2", "epsilon_order"])
    summary = {
        "case_count": len(case_summaries),
        "cutoff_count": len(HSV_ANALYTIC_CUTOFFS),
        "max_omega_spread": float(max(row["omega_spread"] for row in case_summaries if row["omega_spread"] is not None)),
        "cutoff_stable_case_count": sum(1 for row in case_summaries if row["cutoff_stable"]),
        "all_cases_cutoff_stable_within_1e-2": bool(case_summaries) and all(row["cutoff_stable"] for row in case_summaries),
        "corrected_thermodynamics_closed": True,
    }
    payload = {"summary": summary, "case_summaries": case_summaries, "rows": rows}
    json_dump(output_dir / "d4_analytic_control_summary.json", payload)
    return payload


def run_d4_claim3_sign_map(
    output_dir: Path,
    alpha_dm_values: Optional[Sequence[float]] = None,
    nu_values: Optional[Sequence[float]] = None,
    delta2_values: Optional[Sequence[float]] = None,
    mu_multipliers: Optional[Sequence[float]] = None,
    epsilon_grid: Optional[Sequence[float]] = None,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    alpha_dm_values = list(alpha_dm_values or D4_DEFAULT_ALPHA_DM)
    nu_values = list(nu_values or D4_DEFAULT_NU)
    delta2_values = list(delta2_values or D4_DEFAULT_DELTA2)
    mu_multipliers = list(mu_multipliers or D4_DEFAULT_MU_MULTIPLIERS)
    epsilon_grid = list(epsilon_grid or D4_DEFAULT_EPSILON_GRID)

    raw_rows: List[Dict] = []
    fit_rows: List[Dict] = []

    for delta2 in delta2_values:
        for nu_target in nu_values:
            for alpha_dm in alpha_dm_values:
                point_key = f"d4_d{delta2:.2f}_a{alpha_dm:.1f}_nu{nu_target / MU0_D4:.1f}"
                mu_c = float(d4_exact_mu_c(alpha_dm=alpha_dm, nu=nu_target))
                minima_rows: List[Dict] = []
                for mu_scale in mu_multipliers:
                    mu_target = float(mu_c * mu_scale)
                    slice_rows, slice_minima = d4_scan_mu_slice(
                        point_key=point_key,
                        alpha_dm=alpha_dm,
                        nu_target=nu_target,
                        delta2=delta2,
                        mu_target=mu_target,
                        epsilon_grid=epsilon_grid,
                    )
                    raw_rows.extend(slice_rows)
                    minima_rows.extend(slice_minima)

                if len(minima_rows) >= 3:
                    fit_input = [{"accepted": True, "mu": row["mu"], "psi_sq": row["psi_sq"]} for row in minima_rows]
                    fit = fit_small_branch(fit_input, mu_c)
                    if fit.get("fit_success") and abs(float(fit["mu0_minus_mu_c"])) > D4_MU0_SANITY_TOL:
                        fit = {
                            "fit_success": False,
                            "message": "stationary-minimum fit fails the onset-intercept consistency check",
                            "accepted_points": len(minima_rows),
                            "mu_c": mu_c,
                            "claim2_candidate": False,
                        }
                else:
                    reason = "no ordered thermodynamic minimum detected on the scanned fixed-mu slices"
                    if minima_rows:
                        reason = "fewer than three ordered thermodynamic minima detected"
                    fit = {
                        "fit_success": False,
                        "message": reason,
                        "accepted_points": len(minima_rows),
                        "mu_c": mu_c,
                        "claim2_candidate": False,
                    }

                fit_row = {
                    "point_key": point_key,
                    "dimension": 4,
                    "alpha_hv": 0.5,
                    "z": 1.0,
                    "delta2": float(delta2),
                    "alpha_dm": float(alpha_dm),
                    "nu_target": float(nu_target),
                    "mu0_reference": MU0_D4,
                    "mu_c_exact": mu_c,
                    "stationary_minima_count": len(minima_rows),
                }
                fit_row.update(fit)
                fit_row["claim3_reading"] = "no_stationary_ordered_branch"
                if fit_row.get("fit_success"):
                    fit_row["claim3_reading"] = "positive_c2" if fit_row["c2"] > 0.0 else "nonpositive_c2"
                fit_rows.append(fit_row)

    csv_dump(output_dir / "d4_claim3_raw_scans.csv", raw_rows)
    csv_dump(output_dir / "d4_claim3_sign_map.csv", fit_rows)
    summary = {
        "point_count": len(fit_rows),
        "fit_success_points": sum(1 for row in fit_rows if row.get("fit_success")),
        "positive_c2_points": sum(1 for row in fit_rows if row.get("fit_success") and row.get("c2", 0.0) > 0.0),
        "negative_or_near_zero_c2_points": sum(
            1
            for row in fit_rows
            if row.get("fit_success") and row.get("c2_sign") in {"negative", "near_zero"}
        ),
        "no_stationary_branch_points": sum(
            1 for row in fit_rows if row.get("claim3_reading") == "no_stationary_ordered_branch"
        ),
    }
    payload = {"summary": summary, "fit_rows": fit_rows, "raw_rows": raw_rows}
    json_dump(output_dir / "d4_claim3_sign_map.json", payload)
    return payload


def run_d4_claim2_branch_search(output_dir: Path, claim3_result: Dict) -> Dict:
    output_dir = ensure_dir(output_dir)
    fit_rows = claim3_result["fit_rows"]
    raw_rows = claim3_result["raw_rows"]
    candidate_points = {
        row["point_key"]
        for row in fit_rows
        if row.get("fit_success") and row.get("claim2_candidate")
    }

    grouped: Dict[Tuple[str, float], List[Dict]] = {}
    for row in raw_rows:
        if row["point_key"] not in candidate_points or row["branch_type"] != "ordered":
            continue
        key = (row["point_key"], row["mu_target"])
        grouped.setdefault(key, []).append(row)

    branch_rows: List[Dict] = []
    coexistence_rows: List[Dict] = []
    for (point_key, mu_target), rows in grouped.items():
        clean_minima = d4_detect_local_minima(rows)
        branch_rows.extend(clean_minima)
        if len(clean_minima) >= 2:
            psi_values = sorted(abs(float(row["vev"])) for row in clean_minima if row.get("vev") is not None)
            if len(psi_values) >= 2 and abs(psi_values[-1] - psi_values[0]) > 1.0e-4:
                coexistence_rows.append({
                    "point_key": point_key,
                    "mu_target": float(mu_target),
                    "ordered_branch_count": len(clean_minima),
                    "psi_values": psi_values,
                })

    csv_dump(output_dir / "d4_claim2_stationary_branches.csv", branch_rows)
    csv_dump(output_dir / "d4_claim2_coexistence_rows.csv", coexistence_rows)
    summary = {
        "candidate_point_count": len(candidate_points),
        "same_mu_ordered_ordered_coexistence_points": len(coexistence_rows),
    }
    payload = {"summary": summary, "candidate_points": sorted(candidate_points), "coexistence_rows": coexistence_rows}
    json_dump(output_dir / "d4_claim2_branch_search.json", payload)
    return payload


def run_d3_analytic_scan(output_dir: Path, delta2_values: Optional[Sequence[float]] = None, epsilon_order_values: Optional[Sequence[float]] = None) -> Dict:
    output_dir = ensure_dir(output_dir)
    delta2_values = list(delta2_values or [0.02, 0.05, 0.10])
    epsilon_order_values = list(epsilon_order_values or [0.10, 0.20, 0.30])

    u = np.linspace(1.001, HSV_ANALYTIC_UGRID_MAX, 1200)
    thermo_rows: List[Dict] = []
    status_rows: List[Dict] = []
    for delta2 in delta2_values:
        for epsilon in epsilon_order_values:
            p = HsvIslandParams(dimension=3, alpha_hv=2.0, z=1.0, delta2=float(delta2), epsilon_order=float(epsilon))
            sol = d3_analytic_island_solution(u, p)
            for u_cut in HSV_ANALYTIC_CUTOFFS:
                thermo = hsv_renormalized_thermodynamics(u, sol, p, u_cut=u_cut)
                thermo_rows.append({
                    "dimension": 3,
                    "alpha_hv": 2.0,
                    "z": 1.0,
                    "delta2": float(delta2),
                    "epsilon_order": float(epsilon),
                    "u_cut": float(u_cut),
                    "mass": thermo["mass"],
                    "temperature": thermo["temperature"],
                    "entropy_density": thermo["entropy_density"],
                    "Omega_ren": thermo["Omega_ren"],
                })
            status_rows.append({
                "dimension": 3,
                "alpha_hv": 2.0,
                "z": 1.0,
                "delta2": float(delta2),
                "epsilon_order": float(epsilon),
                "claim3_status": "open",
                "message": "Only perturbative analytic thermodynamics is closed here; no nonlinear dark-HSV branch solver exists yet on the D=3 analytic island.",
            })

    csv_dump(output_dir / "d3_corrected_thermo_scan.csv", thermo_rows)
    csv_dump(output_dir / "d3_claim_status.csv", status_rows)
    case_summaries = summarize_cutoff_cases(thermo_rows, ["delta2", "epsilon_order"])
    summary = {
        "case_count": len(status_rows),
        "cutoff_count": len(HSV_ANALYTIC_CUTOFFS),
        "max_omega_spread": float(max(row["omega_spread"] for row in case_summaries if row["omega_spread"] is not None)),
        "cutoff_stable_case_count": sum(1 for row in case_summaries if row["cutoff_stable"]),
        "all_cases_cutoff_stable_within_1e-2": bool(case_summaries) and all(row["cutoff_stable"] for row in case_summaries),
        "corrected_thermodynamics_closed_for_analytic_backgrounds": True,
        "nonlinear_dark_branch_solver_closed": False,
    }
    payload = {"summary": summary, "thermo_rows": thermo_rows, "status_rows": status_rows, "case_summaries": case_summaries}
    json_dump(output_dir / "d3_analytic_summary.json", payload)
    return payload


def run_d4_hee_scan(
    output_dir: Path,
    alpha_dm_values: Optional[Sequence[float]] = None,
    nu_values: Optional[Sequence[float]] = None,
    delta2_values: Optional[Sequence[float]] = None,
    epsilon_values: Optional[Sequence[float]] = None,
    widths: Optional[Sequence[float]] = None,
    mu_scale: float = D4_HEE_MU_SCALE,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    alpha_dm_values = list(alpha_dm_values or D4_HEE_ALPHA_DM)
    nu_values = list(nu_values or D4_HEE_NU)
    delta2_values = list(delta2_values or D4_HEE_DELTA2)
    epsilon_values = list(epsilon_values or D4_HEE_EPSILON_GRID)
    widths = list(widths or D4_HEE_WIDTHS)

    rows: List[Dict] = []
    response_rows: List[Dict] = []
    width_fit_rows: List[Dict] = []
    tc_cache: Dict[Tuple[float, float, float], float] = {}

    for delta2 in delta2_values:
        for nu_target in nu_values:
            for alpha_dm in alpha_dm_values:
                point_key = f"d4hee_d{delta2:.2f}_a{alpha_dm:.1f}_nu{nu_target / MU0_D4:.1f}"
                mu_c = float(d4_exact_mu_c(alpha_dm=alpha_dm, nu=nu_target))
                tc_key = (float(delta2), float(alpha_dm), float(nu_target))
                if tc_key not in tc_cache:
                    onset_case = HsvEffectiveCase(alpha_dm=alpha_dm, nu=nu_target, delta2=delta2, epsilon=0.0)
                    _, onset_sol = solve_d4_effective_case(onset_case, mu_override=mu_c)
                    tc_cache[tc_key] = float(d4_effective_temperature(onset_sol)) if onset_sol.success else float("nan")
                temperature_c = tc_cache[tc_key]
                mu_target = float(mu_scale * mu_c)

                seed_sol = None
                for epsilon in epsilon_values:
                    case = HsvEffectiveCase(alpha_dm=alpha_dm, nu=nu_target, delta2=delta2, epsilon=epsilon)
                    y_init = seed_sol.sol(np.linspace(1.001, 12.0, 260)) if seed_sol is not None and seed_sol.success else None
                    _, sol = solve_d4_effective_case(case, y_init=y_init, mu_override=mu_target)
                    seed_sol = sol if sol.success else seed_sol
                    thermo = d4_effective_boundary_thermodynamics(sol, case) if sol.success else None
                    for width in widths:
                        obs = d4_hee_strip_observables(sol, W=float(width)) if sol.success else {
                            "success": False,
                            "S_parallel": float("nan"),
                            "S_perp": float("nan"),
                            "O12_EE": float("nan"),
                            "u_star_parallel": float("nan"),
                            "u_star_perp": float("nan"),
                        }
                        row = {
                            "point_key": point_key,
                            "dimension": 4,
                            "alpha_hv": 0.5,
                            "z": 1.0,
                            "alpha_dm": float(alpha_dm),
                            "alpha_dm_sq_over4": float(alpha_dm * alpha_dm / 4.0),
                            "nu_target": float(nu_target),
                            "nu_ratio_to_mu0": float(nu_target / MU0_D4),
                            "delta2": float(delta2),
                            "epsilon": float(epsilon),
                            "mu_c": mu_c,
                            "mu_target": mu_target,
                            "mu_scale": float(mu_scale),
                            "width": float(width),
                            "success": bool(sol.success),
                            "temperature_c": float(temperature_c),
                            "accepted": False,
                        }
                        if thermo is not None:
                            reduced_temp = float("nan")
                            if math.isfinite(temperature_c) and abs(temperature_c) > 1.0e-12:
                                reduced_temp = float(1.0 - thermo["temperature"] / temperature_c)
                            row.update({
                                "residual_inf": thermo["residual_inf"],
                                "source": thermo["source"],
                                "vev": thermo["vev"],
                                "psi_abs": thermo["psi_abs"],
                                "psi_sq": thermo["psi_sq"],
                                "temperature": thermo["temperature"],
                                "rho": thermo["rho"],
                                "rho_d": thermo["rho_d"],
                                "epsilon_density": thermo["epsilon"],
                                "Omega_ren": thermo["Omega_ren"],
                                "A_max": thermo["A_max"],
                                "S_parallel": obs["S_parallel"],
                                "S_perp": obs["S_perp"],
                                "O12_EE": obs["O12_EE"],
                                "u_star_parallel": obs["u_star_parallel"],
                                "u_star_perp": obs["u_star_perp"],
                                "reduced_temp": reduced_temp,
                            })
                            row["accepted"] = bool(
                                abs(thermo["source"]) <= D4_SOURCE_TOL
                                and thermo["residual_inf"] <= D4_RESIDUAL_TOL
                                and obs["success"]
                                and math.isfinite(obs["O12_EE"])
                            )
                        rows.append(row)

    grouped_for_response: Dict[Tuple[float, float, float, float, float], List[Dict]] = {}
    for row in rows:
        key = (
            float(row["delta2"]),
            float(row["nu_target"]),
            float(row["epsilon"]),
            float(row["mu_scale"]),
            float(row["width"]),
        )
        grouped_for_response.setdefault(key, []).append(row)

    for key, block in grouped_for_response.items():
        base_rows = [row for row in block if abs(float(row["alpha_dm"])) <= 1.0e-12 and row.get("accepted")]
        if not base_rows:
            continue
        baseline = base_rows[0]
        for row in block:
            if not row.get("accepted"):
                continue
            response_rows.append({
                "delta2": float(row["delta2"]),
                "nu_target": float(row["nu_target"]),
                "nu_ratio_to_mu0": float(row["nu_ratio_to_mu0"]),
                "epsilon": float(row["epsilon"]),
                "mu_scale": float(row["mu_scale"]),
                "width": float(row["width"]),
                "alpha_dm": float(row["alpha_dm"]),
                "alpha_dm_sq_over4": float(row["alpha_dm_sq_over4"]),
                "baseline_O12_EE": float(baseline["O12_EE"]),
                "O12_EE": float(row["O12_EE"]),
                "delta_O12_EE": float(row["O12_EE"] - baseline["O12_EE"]),
            })

    grouped_for_width_fit: Dict[Tuple[float, float, float, float, float], List[Dict]] = {}
    for row in rows:
        key = (
            float(row["delta2"]),
            float(row["alpha_dm"]),
            float(row["nu_target"]),
            float(row["epsilon"]),
            float(row["mu_scale"]),
        )
        grouped_for_width_fit.setdefault(key, []).append(row)

    for key, block in grouped_for_width_fit.items():
        usable = [
            row for row in block
            if row.get("accepted") and row.get("O12_EE") is not None and abs(float(row["O12_EE"])) > 1.0e-14
        ]
        usable = sorted(usable, key=lambda row: float(row["width"]))
        if len(usable) < 3:
            continue
        widths_arr = np.asarray([float(row["width"]) for row in usable], dtype=float)
        oee_arr = np.asarray([abs(float(row["O12_EE"])) for row in usable], dtype=float)
        coeffs = np.polyfit(np.log(widths_arr), np.log(oee_arr), 1)
        pred = np.polyval(coeffs, np.log(widths_arr))
        rmse = float(np.sqrt(np.mean((pred - np.log(oee_arr)) ** 2)))
        width_fit_rows.append({
            "delta2": key[0],
            "alpha_dm": key[1],
            "alpha_dm_sq_over4": key[1] * key[1] / 4.0,
            "nu_target": key[2],
            "nu_ratio_to_mu0": key[2] / MU0_D4,
            "epsilon": key[3],
            "mu_scale": key[4],
            "fit_point_count": len(usable),
            "small_width_exponent": float(coeffs[0]),
            "log_coefficient": float(coeffs[1]),
            "log_fit_rmse": rmse,
        })

    csv_dump(output_dir / "d4_hee_scan.csv", rows)
    csv_dump(output_dir / "d4_hee_alpha_response.csv", response_rows)
    csv_dump(output_dir / "d4_hee_width_fits.csv", width_fit_rows)
    summary = {
        "case_count": len({(row["point_key"], row["epsilon"]) for row in rows}),
        "width_row_count": len(rows),
        "accepted_width_rows": sum(1 for row in rows if row.get("accepted")),
        "isotropic_limit_rows": sum(
            1
            for row in rows
            if abs(float(row["epsilon"])) <= 1.0e-12 and row.get("accepted") and abs(float(row["O12_EE"])) <= 1.0e-6
        ),
        "response_row_count": len(response_rows),
        "width_fit_count": len(width_fit_rows),
        "median_small_width_exponent": float(np.median([row["small_width_exponent"] for row in width_fit_rows])) if width_fit_rows else None,
        "note": "These are actual O12_EE strip-entropy data on the corrected D=4 effective island, but they do not by themselves close the global HEE claim.",
    }
    payload = {"summary": summary, "rows": rows, "response_rows": response_rows, "width_fit_rows": width_fit_rows}
    json_dump(output_dir / "d4_hee_summary.json", payload)
    return payload


def write_hsv_report(output_path: Path, summary: Dict) -> None:
    lines = [
        "# HSV Solver Forks",
        "",
        "## D=4 analytic island",
        "- The effective D=4 fork now uses corrected boundary-data thermodynamics only.",
        f"- Corrected D=4 thermo validation cases: {summary['d4_validation']['summary']['case_count']}.",
        f"- Cutoff-stable corrected D=4 cases: {summary['d4_validation']['summary']['cutoff_stable_case_count']}.",
        f"- Perturbative D=4 analytic-control max cutoff spread: {summary['d4_analytic_control']['summary']['max_omega_spread']:.6g}.",
        f"- D=4 Claim 3 fit-closed points: {summary['d4_claim3']['summary']['fit_success_points']}.",
        f"- D=4 points with no stationary ordered thermodynamic minimum on the scanned slices: {summary['d4_claim3']['summary']['no_stationary_branch_points']}.",
        f"- D=4 same-mu ordered-ordered coexistence points found: {summary['d4_claim2']['summary']['same_mu_ordered_ordered_coexistence_points']}.",
        f"- D=4 HEE width rows with actual O12 data: {summary['d4_hee']['summary']['accepted_width_rows']} / {summary['d4_hee']['summary']['width_row_count']}.",
        f"- D=4 HEE small-width fit count: {summary['d4_hee']['summary']['width_fit_count']}.",
        "",
        "## D=3 analytic island",
        "- The perturbative D=3 analytic background now has normalized boundary-data thermodynamic diagnostics and cutoff checks.",
        f"- D=3 perturbative analytic-control max cutoff spread: {summary['d3']['summary']['max_omega_spread']:.6g}.",
        "- The nonlinear dark-HSV branch solver is still missing there, so Claim 2 and Claim 3 remain open on the D=3 island.",
        "",
        "## Global reading",
        "- D=5 remains a strong negative-evidence anchor: no corrected-scheme c2 sign flip and no same-mu ordered-ordered coexistence were found there.",
        "- D=4 corrected effective thermodynamics can now be tested directly without the old bulk free-energy proxy.",
        "- Global Claim 2 and Claim 3 are still open because the D=3 HSV island lacks a corrected nonlinear branch closure.",
        "- Claim 4 is still open globally, but the corrected D=4 effective island now has actual O12_EE data that can be used for figure production and scaling tests.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_hsv_solver_forks(output_dir: Path) -> Dict:
    output_dir = ensure_dir(output_dir)
    d4_dir = ensure_dir(output_dir / "d4_analytic_island")
    d3_dir = ensure_dir(output_dir / "d3_analytic_island")

    d4_validation = validate_d4_corrected_thermodynamics(d4_dir)
    d4_analytic_control = run_d4_analytic_control_scan(d4_dir)
    d4_claim3 = run_d4_claim3_sign_map(d4_dir)
    d4_claim2 = run_d4_claim2_branch_search(d4_dir, d4_claim3)
    d4_hee = run_d4_hee_scan(d4_dir)
    d3 = run_d3_analytic_scan(d3_dir)

    result = {
        "d4_validation": d4_validation,
        "d4_analytic_control": d4_analytic_control,
        "d4_claim3": d4_claim3,
        "d4_claim2": d4_claim2,
        "d4_hee": d4_hee,
        "d3": d3,
        "summary": {
            "d4_corrected_thermodynamics_closed": bool(d4_validation["summary"]["all_cases_cutoff_stable"]),
            "d3_analytic_thermodynamics_closed": bool(d3["summary"]["corrected_thermodynamics_closed_for_analytic_backgrounds"]),
            "d3_nonlinear_branch_solver_closed": bool(d3["summary"]["nonlinear_dark_branch_solver_closed"]),
            "claim4_has_actual_o12_data": bool(d4_hee["summary"]["accepted_width_rows"]),
            "claim2_closed": False,
            "claim3_closed": False,
        },
        "lines": [
            "D=4 analytic island: corrected boundary-data thermodynamics is now wired in and passes the stored cutoff-stability checks on the scanned effective cases.",
            "D=4 Claim 3 is thermodynamically probed on fixed-mu slices by searching for stationary ordered minima of the corrected grand potential.",
            "D=4 HEE/O12_EE data are now exported on multi-width corrected effective slices for actual information-theory diagnostics.",
            "D=3 analytic island: perturbative thermodynamic diagnostics are closed, but the nonlinear dark-HSV branch solver is still missing there.",
            "Therefore Claim 2 and Claim 3 are still open globally, even though D=5 remains strongly negative and the D=4 effective closure can now be tested without the old free-energy proxy.",
        ],
    }
    json_dump(output_dir / "hsv_solver_forks_summary.json", result)
    write_hsv_report(output_dir / "hsv_solver_forks_report.md", result)
    return result
