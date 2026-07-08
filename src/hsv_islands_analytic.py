import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class HsvIslandParams:
    dimension: int
    alpha_hv: float
    z: float = 1.0
    delta2: float = 0.10
    epsilon_order: float = 1.0
    r0: float = 1.0
    alpha_dm: float = 0.0
    nu: float = 0.0


def d4_dark_lambda_eff(mu: float, alpha_dm: float, nu: float) -> float:
    mix = 0.5 * alpha_dm
    alpha_tilde = 1.0 - mix * mix
    if alpha_tilde <= 0.0:
        raise ValueError("alpha_tilde must stay positive")
    return (mu * mu - mix * mix * mu * nu) / alpha_tilde


def d4_exact_mu_c(alpha_dm: float = 0.0, nu: float = 0.0) -> float:
    """Exact reduced analytic-island onset law on the D=4 HSV island.

    From the reconstructed handoff:

        (mu_c^2 - (alpha_dm^2 / 4) mu_c nu) / (1 - alpha_dm^2 / 4) = 16 / 3
    """
    mix = 0.5 * alpha_dm
    alpha_tilde = 1.0 - mix * mix
    if alpha_tilde <= 0.0:
        raise ValueError("alpha_tilde must stay positive")
    disc = mix * mix * mix * mix * nu * nu + 64.0 * alpha_tilde / 3.0
    return 0.5 * (mix * mix * nu + math.sqrt(disc))


def d3_analytic_island_solution(u, p: HsvIslandParams) -> Dict[str, np.ndarray]:
    """Leading backreacted analytic island for D=3, (alpha,z)=(2,1).

    Uses eqs. (60)-(63) from 2504.13635 to O(epsilon^2 delta^2).
    This is a perturbative analytic background, not a full nonlinear dark-HSV closure.
    """
    if p.dimension != 3 or p.alpha_hv != 2.0 or p.z != 1.0:
        raise ValueError("D=3 analytic island requires dimension=3, alpha_hv=2, z=1")

    u = np.asarray(u, dtype=float)
    corr = (p.epsilon_order ** 2) * p.delta2
    base_n = 1.0 + 16.0 * p.delta2 / (3.0 * u ** 6) - 16.0 * p.delta2 / (3.0 * u ** 4)
    n2 = (-279.0 + 838.0 * u ** 2 + 1680.0 * u ** 4 - 282.0 * u ** 6 - 281.0 * u ** 8) / (
        2520.0 * u ** 6 * (1.0 + u ** 2) ** 3
    )
    sigma2 = (-2.0 + u ** 2) / (9.0 * (1.0 + u ** 2) ** 4)
    phi2 = (-1.0 + 2.0 * u ** 2) / (6.0 * math.sqrt(3.0) * (1.0 + u ** 2) ** 4)
    return {
        "N": base_n + corr * n2,
        "H": np.ones_like(u),
        "sigma": np.ones_like(u) + corr * sigma2,
        "phi": 2.0 * math.sqrt(3.0) * np.log(p.r0 * u) + corr * phi2,
    }


def d4_analytic_island_solution(u, p: HsvIslandParams) -> Dict[str, np.ndarray]:
    """Leading backreacted analytic island for D=4, (alpha,z)=(1/2,1).

    Uses eqs. (81)-(84) with H2 = -J2 and the common base blackening from eq. (44),
    again only to O(epsilon^2 delta^2).
    """
    if p.dimension != 4 or p.alpha_hv != 0.5 or p.z != 1.0:
        raise ValueError("D=4 analytic island requires dimension=4, alpha_hv=1/2, z=1")

    u = np.asarray(u, dtype=float)
    corr = (p.epsilon_order ** 2) * p.delta2
    base_n = 1.0 + 16.0 * p.delta2 / (3.0 * u ** 6) - 16.0 * p.delta2 / (3.0 * u ** 4)
    n2 = (-279.0 - 628.0 * u ** 2 + 1050.0 * u ** 4 + 138.0 * u ** 6 - 281.0 * u ** 8) / (
        2520.0 * u ** 6 * (1.0 + u ** 2) ** 3
    )
    h2 = (1.0 - 2.0 * u ** 2) / (24.0 * (1.0 + u ** 2) ** 4)
    j2 = -h2
    sigma2 = -(5.0 + 2.0 * u ** 2) / (36.0 * (1.0 + u ** 2) ** 4)
    phi2 = -(1.0 - 2.0 * u ** 2) / (12.0 * math.sqrt(3.0) * (1.0 + u ** 2) ** 4)
    return {
        "N": base_n + corr * n2,
        "H": np.ones_like(u) + corr * h2,
        "J": np.ones_like(u) + corr * j2,
        "sigma": np.ones_like(u) + corr * sigma2,
        "phi": 2.0 * math.sqrt(3.0) * np.log(p.r0 * u) + corr * phi2,
    }


def hsv_island_status_lines() -> Dict[str, str]:
    return {
        "scope": "The analytic-island module contains perturbative D=3/D=4 HSV backgrounds, the reduced exact D=4 onset law, and normalized boundary-data thermodynamic diagnostics for those analytic backgrounds.",
        "not_closed": "The perturbative analytic-island thermodynamics is usable as a cutoff-stability diagnostic, but it is not by itself a full nonlinear dark-HSV closure for Claim 2 or Claim 3.",
    }


def mass_coefficient_from_blackening(u: float, n_value: float, dn_du: float) -> float:
    """Extract the asymptotic black-brane mass coefficient from N(u).

    The analytic islands share the large-u structure

        N(u) = 1 - M / u^4 + O(u^-6),

    so a cutoff-local combination of N and N' can be used to read off M.
    """
    return float(-3.0 * u ** 4 * (n_value - 1.0) - 0.5 * u ** 5 * dn_du)


def horizon_temperature_from_profile(u: np.ndarray, n_values: np.ndarray, sigma_values: np.ndarray) -> float:
    fit_count = min(6, len(u))
    if fit_count < 2:
        return float("nan")
    x = np.asarray(u[:fit_count], dtype=float) - 1.0
    y = np.asarray(n_values[:fit_count], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    sigma_h = float(sigma_values[0])
    return float(sigma_h * slope / (4.0 * math.pi))


def analytic_entropy_density(p: HsvIslandParams) -> float:
    """Normalized horizon entropy density used throughout the HSV diagnostics.

    The current HSV island modules work in the same normalized units as the D=5
    reanalysis, so the entropy density is stored in the same conventionally
    normalized `2 pi` units. This is sufficient for branch differences and
    cutoff-stability checks, which are the quantities used downstream.
    """
    _ = p
    return float(2.0 * math.pi)


def _interpolate_at_cutoff(u: np.ndarray, values: np.ndarray, u_cut: float) -> float:
    return float(np.interp(float(u_cut), np.asarray(u, dtype=float), np.asarray(values, dtype=float)))


def hsv_renormalized_thermodynamics(
    u: np.ndarray,
    solution: Dict[str, np.ndarray],
    p: HsvIslandParams,
    u_cut: Optional[float] = None,
) -> Dict[str, float]:
    """Normalized boundary-data thermodynamics for the perturbative analytic islands.

    These analytic backgrounds do not yet carry the full nonlinear visible/dark
    gauge-sector branch data, so the returned grand potential contains only the
    energy and entropy contributions. The function is still useful because the
    mass coefficient, temperature, and cutoff stability can be checked directly
    from the boundary data without using the old divergent bulk free-energy
    proxy.
    """
    u = np.asarray(u, dtype=float)
    reduced_n_values = np.asarray(solution["N"], dtype=float)
    full_n_values = (1.0 - u ** -4) * reduced_n_values
    sigma_values = np.asarray(solution["sigma"], dtype=float)
    if u_cut is None:
        u_cut = float(u[-1])
    if not (float(u[0]) <= float(u_cut) <= float(u[-1])):
        raise ValueError("u_cut must lie inside the supplied radial grid")

    dn_du = np.gradient(full_n_values, u)
    n_value = _interpolate_at_cutoff(u, full_n_values, u_cut)
    dn_value = _interpolate_at_cutoff(u, dn_du, u_cut)
    mass = mass_coefficient_from_blackening(float(u_cut), n_value, dn_value)
    temperature = horizon_temperature_from_profile(u, full_n_values, sigma_values)
    entropy = analytic_entropy_density(p)
    omega = float(mass - temperature * entropy)
    return {
        "dimension": float(p.dimension),
        "alpha_hv": float(p.alpha_hv),
        "z": float(p.z),
        "delta2": float(p.delta2),
        "epsilon_order": float(p.epsilon_order),
        "u_cut": float(u_cut),
        "mass": float(mass),
        "epsilon": float(mass),
        "temperature": float(temperature),
        "entropy_density": float(entropy),
        "rho": 0.0,
        "rho_d": 0.0,
        "mu": 0.0,
        "nu": 0.0,
        "Omega_ren": omega,
    }
