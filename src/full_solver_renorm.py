import math
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq, root


@dataclass
class ModelParams:
    alpha_dm: float = 0.0
    delta2: float = 0.10  # backreaction parameter gamma = kappa^2/g^2
    nu_target: float = 0.0  # dark chemical potential
    r_max: float = 12.0
    eps: float = 1.0e-5
    compact_u_min: float = 1.0e-5
    compact_grid_size: int = 160
    root_residual_tol: float = 1.0e-6
    onset_bvp_tol: float = 1.0e-6

    @property
    def mix(self) -> float:
        return 0.5 * self.alpha_dm

    @property
    def alpha_tilde(self) -> float:
        return 1.0 - self.mix ** 2


def rhs(r: float, y: np.ndarray, p: ModelParams) -> np.ndarray:
    # y = [N, sigma, f, fp, phi, phip, eta, etap, w, wp]
    N, sigma, f, fp, phi, phip, eta, etap, w, wp = y
    # simple guards
    N = max(N, 1.0e-12)
    sigma = max(sigma, 1.0e-12)
    f = max(f, 1.0e-12)

    mix = p.mix
    delta2 = p.delta2
    atil = p.alpha_tilde

    e_elec = phip * phip + etap * etap + 2.0 * mix * phip * etap
    sigma_p = sigma * (2.0 * r * (fp / f) ** 2 + delta2 * f ** 4 / (3.0 * r) * (wp * wp + phi * phi * w * w / (N * N * sigma * sigma)))
    N_p = 4.0 * r - 2.0 * N / r - N * sigma_p / sigma - delta2 * r * e_elec / (3.0 * sigma * sigma)

    # second order equations
    c = 3.0 / r - sigma_p / sigma
    phipp = -c * phip + f ** 4 * phi * w * w / (r * r * N * atil)
    etapp = -c * etap + mix * f ** 4 * phi * w * w / (r * r * N * atil)
    dark_w_mass = mix * mix * eta * phi / (atil * N * N * sigma * sigma)
    wpp = (
        - (1.0 / r + sigma_p / sigma + N_p / N + 4.0 * fp / f) * wp
        - (phi * phi / (atil * N * N * sigma * sigma)) * w
        + dark_w_mass * w
    )

    # generalized f equation from 1109.4592 with electric sector extended by eta, mix
    stress_combo = f ** 4 * wp * wp / (3.0 * r * r) - f ** 4 * phi * phi * w * w / (3.0 * r * r * sigma * sigma * N * N) + e_elec / (6.0 * N * sigma * sigma)
    fpp = f * (
        1.0 / (r * r)
        - 2.0 / N
        + N_p / (2.0 * r * N)
        + sigma_p / (2.0 * r * sigma)
        + delta2 * stress_combo
        + (fp / f) ** 2
    ) - (sigma_p / sigma + 3.0 / r + N_p / N) * fp

    return np.array([N_p, sigma_p, fp, fpp, phip, phipp, etap, etapp, wp, wpp], dtype=float)


def horizon_n1(sigma0: float, phi1: float, eta1: float, p: ModelParams) -> float:
    mix = p.mix
    e_elec = phi1 * phi1 + eta1 * eta1 + 2.0 * mix * phi1 * eta1
    return 4.0 - p.delta2 * e_elec / (3.0 * sigma0 * sigma0)


def initial_state_iso(params_vec: np.ndarray, p: ModelParams) -> np.ndarray:
    logsigma0, phi1, eta1 = params_vec
    sigma0 = float(np.exp(logsigma0))
    n1 = horizon_n1(sigma0, phi1, eta1, p)
    eps = p.eps
    return np.array([
        n1 * eps,
        sigma0,
        1.0,
        0.0,
        phi1 * eps,
        phi1,
        eta1 * eps,
        eta1,
        0.0,
        0.0,
    ], dtype=float)


def initial_state_aniso(params_vec: np.ndarray, p: ModelParams) -> np.ndarray:
    logsigma0, logf0, phi1, eta1, w0 = params_vec
    sigma0 = float(np.exp(logsigma0))
    f0 = float(np.exp(logf0))
    n1 = horizon_n1(sigma0, phi1, eta1, p)
    eps = p.eps
    return np.array([
        n1 * eps,
        sigma0,
        f0,
        0.0,
        phi1 * eps,
        phi1,
        eta1 * eps,
        eta1,
        w0,
        0.0,
    ], dtype=float)


def exact_charge_combo(mu_target: float, p: ModelParams) -> float:
    return mu_target * mu_target + p.nu_target * p.nu_target + 2.0 * p.mix * mu_target * p.nu_target


def exact_compact_blackening(mu_target: float, p: ModelParams, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_mix = exact_charge_combo(mu_target, p)
    a = 1.0 + 2.0 * p.delta2 * q_mix / 3.0
    b = 2.0 * p.delta2 * q_mix / 3.0
    F = 1.0 - a * u * u + b * u * u * u
    F_p = -2.0 * a * u + 3.0 * b * u * u
    return F, F_p


def compact_boundary_source(w: float, w_u: float, u: float) -> float:
    # In u = r^{-2}, the asymptotic source is S = w - u w_u when w = S + V u + O(u^2).
    return float(w - u * w_u)


def exact_isotropic_solution(mu_target: float, p: ModelParams):
    q_vis = mu_target
    q_dark = p.nu_target
    q_mix = exact_charge_combo(mu_target, p)
    a = 1.0 + 2.0 * p.delta2 * q_mix / 3.0
    b = 2.0 * p.delta2 * q_mix / 3.0

    def sol(r_eval):
        r_arr = np.atleast_1d(np.asarray(r_eval, dtype=float))
        N = r_arr ** 2 - a / (r_arr ** 2) + b / (r_arr ** 4)
        sigma = np.ones_like(r_arr)
        f = np.ones_like(r_arr)
        fp = np.zeros_like(r_arr)
        phi = q_vis * (1.0 - 1.0 / (r_arr ** 2))
        phip = 2.0 * q_vis / (r_arr ** 3)
        eta = q_dark * (1.0 - 1.0 / (r_arr ** 2))
        etap = 2.0 * q_dark / (r_arr ** 3)
        w = np.zeros_like(r_arr)
        wp = np.zeros_like(r_arr)
        out = np.vstack([N, sigma, f, fp, phi, phip, eta, etap, w, wp])
        if np.isscalar(r_eval):
            return out[:, 0]
        return out

    r_grid = np.linspace(1.0 + p.eps, p.r_max, 400)
    return SimpleNamespace(success=True, sol=sol, t=r_grid, y=sol(r_grid))


def compact_onset_coefficients(u: np.ndarray, mu_target: float, p: ModelParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    F, F_p = exact_compact_blackening(mu_target, p, u)
    visible_dark_mass = mu_target * mu_target - p.mix * p.mix * mu_target * p.nu_target
    q_term = ((1.0 - u) ** 2) * visible_dark_mass / (4.0 * p.alpha_tilde * u * F * F)
    return F, F_p, q_term


def solve_linear_onset_compact(
    p: ModelParams,
    mu_guess: Optional[float] = None,
    guess_profile: Optional[Dict[str, List[float]]] = None,
) -> Dict:
    """Solve the compact-coordinate linear onset BVP on the exact isotropic background.

    We work with u = r^{-2} on u in [0, 1], where u = 0 is the AdS boundary and u = 1 the horizon.
    The source-free onset mode satisfies

        w_uu + (F_u / F) w_u + Q(u; mu) w = 0,

    with boundary conditions
        w - u w_u = 0   at u -> 0  (source-free)
        w_u = 0         at u -> 1  (regular horizon)
        w = 1           at u -> 1  (normalization)
    """
    if mu_guess is None:
        mu_guess = 4.0 / math.sqrt(3.0)

    u0 = p.compact_u_min
    u1 = 1.0 - p.compact_u_min
    u_grid = np.linspace(u0, u1, p.compact_grid_size)

    if guess_profile is not None:
        guess_u = np.asarray(guess_profile["u"], dtype=float)
        guess_w = np.asarray(guess_profile["w"], dtype=float)
        guess_w_u = np.asarray(guess_profile["w_u"], dtype=float)
        y_init = np.vstack([
            np.interp(u_grid, guess_u, guess_w),
            np.interp(u_grid, guess_u, guess_w_u),
        ])
    else:
        y_init = np.vstack([u_grid.copy(), np.ones_like(u_grid)])

    def ode(u, y, p_mu):
        mu_trial = float(p_mu[0])
        F, F_p, q_term = compact_onset_coefficients(u, mu_trial, p)
        return np.vstack([y[1], -(F_p / F) * y[1] - q_term * y[0]])

    def bc(ya, yb, p_mu):
        source = compact_boundary_source(ya[0], ya[1], u0)
        return np.array([source, yb[1], yb[0] - 1.0], dtype=float)

    last_failure: Dict[str, object] = {"success": False, "message": "compact onset BVP did not run"}
    for tol in [p.onset_bvp_tol, max(5.0e-6, 5.0 * p.onset_bvp_tol), max(1.0e-5, 10.0 * p.onset_bvp_tol)]:
        try:
            sol = solve_bvp(
                ode,
                bc,
                u_grid,
                y_init,
                p=np.array([mu_guess], dtype=float),
                tol=tol,
                max_nodes=100000,
                verbose=0,
            )
        except Exception as exc:
            last_failure = {
                "success": False,
                "message": f"compact onset BVP failed: {exc!r}",
                "mu_guess": float(mu_guess),
                "tol": float(tol),
            }
            continue

        if sol.success:
            profile = sol.sol(u_grid)
            source = compact_boundary_source(profile[0, 0], profile[1, 0], u_grid[0])
            return {
                "success": True,
                "mu_c": float(sol.p[0]),
                "mu_guess": float(mu_guess),
                "u": u_grid.tolist(),
                "w": profile[0].tolist(),
                "w_u": profile[1].tolist(),
                "source": float(source),
                "rms_residual_max": float(np.max(sol.rms_residuals)) if len(sol.rms_residuals) else 0.0,
                "bvp": sol,
                "tol_used": float(tol),
            }

        last_failure = {
            "success": False,
            "message": sol.message,
            "mu_guess": float(mu_guess),
            "tol": float(tol),
        }
    return last_failure


def shoot_linear_onset_compact(mu: float, p: ModelParams) -> Dict:
    u0 = p.compact_u_min
    u1 = 1.0 - p.compact_u_min
    u_desc = np.linspace(u1, u0, p.compact_grid_size)

    def lin_rhs(u, y):
        w, w_u = y
        F, F_p, q_term = compact_onset_coefficients(np.array([u], dtype=float), mu, p)
        return np.array([w_u, -(F_p[0] / F[0]) * w_u - q_term[0] * w], dtype=float)

    try:
        sol = solve_ivp(
            lin_rhs,
            (u1, u0),
            np.array([1.0, 0.0], dtype=float),
            method="Radau",
            t_eval=u_desc,
            atol=1.0e-9,
            rtol=1.0e-8,
            max_step=0.01,
        )
    except Exception as exc:
        return {"success": False, "message": f"compact onset shooting failed: {exc!r}", "mu_guess": float(mu)}
    if not sol.success or np.any(~np.isfinite(sol.y)):
        return {"success": False, "message": "compact onset shooting failed", "mu_guess": float(mu)}

    w_b, w_u_b = sol.y[:, -1]
    source = compact_boundary_source(w_b, w_u_b, u0)
    return {
        "success": True,
        "mu_c": float(mu),
        "mu_guess": float(mu),
        "u": sol.t[::-1].tolist(),
        "w": sol.y[0, ::-1].tolist(),
        "w_u": sol.y[1, ::-1].tolist(),
        "source": float(source),
        "rms_residual_max": 0.0,
        "method": "compact_shooting",
    }


def integrate_from_horizon(y0: np.ndarray, p: ModelParams, dense: bool = False):
    sol = solve_ivp(
        fun=lambda r, y: rhs(r, y, p),
        t_span=(1.0 + p.eps, p.r_max),
        y0=y0,
        method="Radau",
        atol=1e-8,
        rtol=1e-7,
        dense_output=dense,
        max_step=0.1,
    )
    return sol


def boundary_source_w(yb: np.ndarray, r: float) -> float:
    # if w = S + V/r^2, then S ~ w + r/2 w'
    w = yb[8]
    wp = yb[9]
    return float(w + 0.5 * r * wp)


def boundary_chemical_potentials(yb: np.ndarray, r: float) -> Dict[str, float]:
    # If phi = mu + q/r^2 + O(r^-4), then mu = phi + (r/2) phi'.
    phi, phip = yb[4], yb[5]
    eta, etap = yb[6], yb[7]
    return {
        "mu": float(phi + 0.5 * r * phip),
        "nu": float(eta + 0.5 * r * etap),
    }


def residual_iso(params_vec: np.ndarray, mu_target: float, p: ModelParams) -> np.ndarray:
    try:
        y0 = initial_state_iso(params_vec, p)
        sol = integrate_from_horizon(y0, p)
    except Exception:
        return np.array([1e3, 1e3, 1e3])
    if (not sol.success) or np.any(~np.isfinite(sol.y[:, -1])) or np.min(sol.y[0]) <= 0:
        return np.array([1e3, 1e3, 1e3])
    yb = sol.y[:, -1]
    chem = boundary_chemical_potentials(yb, p.r_max)
    return np.array([
        yb[1] - 1.0,         # sigma(infty)=1
        chem["mu"] - mu_target,
        chem["nu"] - p.nu_target,
    ])


def residual_aniso(params_vec: np.ndarray, mu_target: float, p: ModelParams) -> np.ndarray:
    try:
        y0 = initial_state_aniso(params_vec, p)
        sol = integrate_from_horizon(y0, p)
    except Exception:
        return np.ones(5) * 1e3
    if (not sol.success) or np.any(~np.isfinite(sol.y[:, -1])) or np.min(sol.y[0]) <= 0:
        return np.ones(5) * 1e3
    yb = sol.y[:, -1]
    chem = boundary_chemical_potentials(yb, p.r_max)
    return np.array([
        yb[1] - 1.0,
        yb[2] - 1.0,
        chem["mu"] - mu_target,
        chem["nu"] - p.nu_target,
        boundary_source_w(yb, p.r_max),
    ])


def solve_isotropic(mu_target: float, p: ModelParams, guess: Optional[np.ndarray] = None) -> Dict:
    params = np.array([0.0, 2.0 * mu_target, 2.0 * p.nu_target], dtype=float)
    y0 = initial_state_iso(params, p)
    ivp = exact_isotropic_solution(mu_target, p)
    n1 = horizon_n1(1.0, params[1], params[2], p)
    temp = n1 / (4.0 * math.pi)
    return {
        "success": True,
        "params": params.tolist(),
        "y0": y0.tolist(),
        "temperature": temp,
        "rho": float(2.0 * p.delta2 * (mu_target + p.mix * p.nu_target)),
        "rho_d": float(2.0 * p.delta2 * (p.nu_target + p.mix * mu_target)),
        "residual_inf": 0.0,
        "solution": ivp,
    }


def solve_anisotropic(mu_target: float, p: ModelParams, guess: Optional[np.ndarray] = None) -> Dict:
    if guess is None:
        iso = solve_isotropic(mu_target, p)
        if not iso["success"]:
            return {"success": False, "message": "failed isotropic seed"}
        g = iso["params"]
        guess = np.array([g[0], 0.0, g[1], g[2], 0.5])
    sol = root(lambda z: residual_aniso(z, mu_target, p), guess, method='hybr', tol=1e-10)
    if not sol.success:
        sol = root(lambda z: residual_aniso(z, mu_target, p), guess, method='lm', tol=1e-10)
    if not sol.success:
        return {"success": False, "message": sol.message, "guess": guess.tolist()}
    resid = residual_aniso(sol.x, mu_target, p)
    resid_inf = float(np.max(np.abs(resid)))
    if (not np.all(np.isfinite(resid))) or resid_inf > p.root_residual_tol:
        return {
            "success": False,
            "message": "anisotropic root failed residual check",
            "guess": guess.tolist(),
            "residual_inf": resid_inf,
        }
    y0 = initial_state_aniso(sol.x, p)
    ivp = integrate_from_horizon(y0, p, dense=True)
    yb = ivp.y[:, -1]
    n1 = horizon_n1(float(np.exp(sol.x[0])), sol.x[2], sol.x[3], p)
    temp = n1 * float(np.exp(sol.x[0])) / (4.0 * math.pi)
    r = p.r_max
    chem = boundary_chemical_potentials(yb, r)
    rho = p.delta2 * r ** 3 * (yb[5] + p.mix * yb[7])
    rho_d = p.delta2 * r ** 3 * (yb[7] + p.mix * yb[5])
    source = boundary_source_w(yb, r)
    vev = (yb[8] - source) * r * r
    return {
        "success": True,
        "params": sol.x.tolist(),
        "y0": y0.tolist(),
        "temperature": temp,
        "mu": chem["mu"],
        "nu": chem["nu"],
        "rho": float(rho),
        "rho_d": float(rho_d),
        "source": float(source),
        "vev": float(vev),
        "residual_inf": resid_inf,
        "solution": ivp,
    }


def w_source_for_mu(mu: float, p: ModelParams, iso_seed: Optional[np.ndarray] = None) -> float:
    """Compact-coordinate source diagnostic for the linear onset ODE at fixed mu."""
    shot = shoot_linear_onset_compact(mu, p)
    if not shot.get("success"):
        return np.nan
    return float(shot["source"])


def estimate_mu_c(bracket: Tuple[float, float], p: ModelParams, guess_profile: Optional[Dict[str, List[float]]] = None) -> Dict:
    """Robust compact-coordinate onset extractor on the exact isotropic background.

    The old helper used a large-r shooting proxy and was visibly inconsistent with the small-branch
    intercept fit. We now work entirely in the compact coordinate on the exact isotropic background,
    using robust compact shooting first and the collocation BVP only as a secondary fallback.
    """
    mu_guesses: List[float]
    if isinstance(bracket, (tuple, list, np.ndarray)) and len(bracket) == 2:
        a = float(bracket[0])
        b = float(bracket[1])
        mid = 0.5 * (a + b)
        mu_guesses = [mid, a, b, 0.5 * (a + mid), 0.5 * (mid + b), 4.0 / math.sqrt(3.0)]
    else:
        mu_guesses = [float(bracket)]

    if isinstance(bracket, (tuple, list, np.ndarray)) and len(bracket) == 2:
        left = max(1.0e-6, float(min(bracket)))
        right = max(left + 1.0e-3, float(max(bracket)))
    else:
        center = max(1.0e-6, float(mu_guesses[0]))
        left = max(1.0e-6, center - 0.5)
        right = center + 0.5

    last_failure: Dict = {"success": False, "message": "no compact shooting sign change found"}

    def finite_source(mu_trial: float) -> float:
        src = w_source_for_mu(mu_trial, p)
        if not math.isfinite(src):
            raise ValueError("non-finite compact shooting source")
        return float(src)

    scan_windows = [(left, right), (max(1.0e-3, left - 0.5), right + 0.5), (max(1.0e-3, left - 1.0), right + 1.0)]
    for scan_left, scan_right in scan_windows:
        grid = np.linspace(scan_left, scan_right, 9)
        values: List[Tuple[float, float]] = []
        for mu_trial in grid:
            try:
                values.append((float(mu_trial), finite_source(float(mu_trial))))
            except Exception:
                continue
        for (mu_a, src_a), (mu_b, src_b) in zip(values[:-1], values[1:]):
            if src_a == 0.0:
                shot = shoot_linear_onset_compact(mu_a, p)
                if shot.get("success"):
                    return shot
            if src_a * src_b > 0.0:
                continue
            try:
                mu_root = brentq(lambda x: finite_source(float(x)), mu_a, mu_b, xtol=1.0e-10, rtol=1.0e-9, maxiter=200)
            except Exception:
                continue
            shot = shoot_linear_onset_compact(float(mu_root), p)
            if shot.get("success") and abs(float(shot["source"])) <= 5.0e-7:
                return shot
    return last_failure


def ricci_scalar_numeric(r: np.ndarray, y: np.ndarray, p: ModelParams) -> np.ndarray:
    # y on grid shape (10, n)
    N, sigma, f, fp, phi, phip, eta, etap, w, wp = y
    # compute first/second via rhs for consistency
    out = np.zeros_like(r)
    for i, rv in enumerate(r):
        vec = y[:, i]
        dydr = rhs(float(rv), vec, p)
        Np, sigp, _, fpp, _, phipp, _, etapp, _, wpp = dydr
        # direct Ricci scalar via Einstein trace: R = -20 + (gamma/3) T^M_M ???
        mix = p.mix
        e_elec = phip[i] ** 2 + etap[i] ** 2 + 2.0 * mix * phip[i] * etap[i]
        F2 = 2.0 * N[i] * f[i] ** 4 * wp[i] ** 2 / (rv ** 2) - 2.0 * f[i] ** 4 * phi[i] ** 2 * w[i] ** 2 / (rv ** 2 * max(N[i], 1e-12) * sigma[i] ** 2) - 2.0 * phip[i] ** 2 / sigma[i] ** 2
        B2 = -2.0 * etap[i] ** 2 / sigma[i] ** 2
        FB = -2.0 * phip[i] * etap[i] / sigma[i] ** 2
        traceT = -0.5 * F2 - 0.5 * B2 - p.alpha_dm * FB
        out[i] = -20.0 - p.delta2 * traceT / 3.0
    return out


def renormalized_grand_potential(branch: Dict, p: ModelParams) -> float:
    raise RuntimeError("Naive bulk Omega_ren has been removed. Use grand_potential_from_boundary_data().")


def residual_wh(params_vec: np.ndarray, w0: float, p: ModelParams) -> np.ndarray:
    logsigma0, logf0, phi1, eta1 = params_vec
    full = np.array([logsigma0, logf0, phi1, eta1, w0], dtype=float)
    try:
        y0 = initial_state_aniso(full, p)
        sol = integrate_from_horizon(y0, p)
    except Exception:
        return np.ones(4) * 1e3
    if (not sol.success) or np.any(~np.isfinite(sol.y[:, -1])) or np.min(sol.y[0]) <= 0:
        return np.ones(4) * 1e3
    yb = sol.y[:, -1]
    chem = boundary_chemical_potentials(yb, p.r_max)
    return np.array([yb[1] - 1.0, yb[2] - 1.0, chem["nu"] - p.nu_target, boundary_source_w(yb, p.r_max)])


def solve_branch_by_wh(w0: float, p: ModelParams, guess: Optional[np.ndarray] = None) -> Dict:
    if guess is None:
        guess = np.array([0.0, 0.0, 8.0 / math.sqrt(3.0), 2.0 * p.nu_target])
    sol = root(lambda z: residual_wh(z, w0, p), guess, method='hybr', tol=1e-10)
    if not sol.success:
        sol = root(lambda z: residual_wh(z, w0, p), guess, method='lm', tol=1e-10)
    if not sol.success:
        return {"success": False, "message": sol.message, "guess": guess.tolist(), "w0": w0}
    resid = residual_wh(sol.x, w0, p)
    resid_inf = float(np.max(np.abs(resid)))
    if (not np.all(np.isfinite(resid))) or resid_inf > p.root_residual_tol:
        return {
            "success": False,
            "message": "small-branch root failed residual check",
            "guess": guess.tolist(),
            "w0": w0,
            "residual_inf": resid_inf,
        }
    full = np.array([sol.x[0], sol.x[1], sol.x[2], sol.x[3], w0], dtype=float)
    y0 = initial_state_aniso(full, p)
    ivp = integrate_from_horizon(y0, p, dense=True)
    yb = ivp.y[:, -1]
    sigma0 = float(np.exp(sol.x[0]))
    n1 = horizon_n1(sigma0, sol.x[2], sol.x[3], p)
    temp = n1 * sigma0 / (4.0 * math.pi)
    chem = boundary_chemical_potentials(yb, p.r_max)
    mu = chem["mu"]
    source = boundary_source_w(yb, p.r_max)
    vev = (yb[8] - source) * p.r_max * p.r_max
    return {
        "success": True,
        "params": full.tolist(),
        "y0": y0.tolist(),
        "temperature": temp,
        "mu": mu,
        "nu": chem["nu"],
        "rho": float(p.delta2 * p.r_max ** 3 * (yb[5] + p.mix * yb[7])),
        "rho_d": float(p.delta2 * p.r_max ** 3 * (yb[7] + p.mix * yb[5])),
        "source": float(source),
        "vev": float(vev),
        "w0": float(w0),
        "residual_inf": resid_inf,
        "solution": ivp,
    }


def bundle_branch(mu: float, alpha_dm: float, delta2: float = 0.10, nu_target: float = 0.0, branch_type: str = "aniso") -> Dict:
    p = ModelParams(alpha_dm=alpha_dm, delta2=delta2, nu_target=nu_target)
    if branch_type == "iso":
        out = solve_isotropic(mu, p)
    else:
        out = solve_anisotropic(mu, p)
    if out.get("success"):
        thermo = grand_potential_from_boundary_data(out, p, mu=mu)
        out["Omega_ren"] = thermo["Omega_ren"]
        out["epsilon"] = thermo["epsilon"]
        out["p_bar"] = thermo["p_bar"]
        out["rho"] = thermo["rho"]
        out["rho_d"] = thermo["rho_d"]
        out["mu"] = mu
        out["alpha_dm"] = alpha_dm
        out["delta2"] = delta2
        out["nu_target"] = nu_target
        out["branch_type"] = branch_type
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=4.0)
    parser.add_argument("--alpha_dm", type=float, default=0.0)
    parser.add_argument("--delta2", type=float, default=0.10)
    parser.add_argument("--nu", type=float, default=0.0)
    parser.add_argument("--branch", choices=["iso", "aniso"], default="iso")
    args = parser.parse_args()
    out = bundle_branch(args.mu, args.alpha_dm, args.delta2, args.nu, args.branch)
    print(json.dumps({k: v for k, v in out.items() if k != 'solution'}, indent=2, default=str))



def holographic_stress_tensor(branch: Dict, p: ModelParams, rb: Optional[float] = None) -> Dict[str,float]:
    """Cutoff-stable renormalized boundary stress tensor for flat AdS5 boundary.
    Uses T_ab = - lim_{rb->∞} rb^2 (K_ab - K gamma_ab + 3 gamma_ab).
    Returns lower-index components (epsilon = T_tt, pressures = T_xx,T_yy,T_zz).
    """
    if rb is None:
        rb = p.r_max
    y = np.asarray(branch["solution"].sol(rb), dtype=float)
    N, sigma, f, fp, phi, phip, eta, etap, w, wp = y
    Np, sigp = rhs(float(rb), y, p)[0], rhs(float(rb), y, p)[1]
    sqrtN = math.sqrt(max(N, 1.0e-30))
    gtt = -N * sigma * sigma
    gxx = rb * rb / (f ** 4)
    gyy = rb * rb * (f ** 2)
    dgtt = -(Np * sigma * sigma + 2.0 * N * sigma * sigp)
    dgxx = 2.0 * rb / (f ** 4) - 4.0 * rb * rb * fp / (f ** 5)
    dgyy = 2.0 * rb * (f ** 2) + 2.0 * rb * rb * f * fp
    Ktt = 0.5 * sqrtN * dgtt
    Kxx = 0.5 * sqrtN * dgxx
    Kyy = 0.5 * sqrtN * dgyy
    K = Ktt / gtt + Kxx / gxx + 2.0 * Kyy / gyy
    Ttt = -rb * rb * (Ktt - K * gtt + 3.0 * gtt)
    Txx = -rb * rb * (Kxx - K * gxx + 3.0 * gxx)
    Tyy = -rb * rb * (Kyy - K * gyy + 3.0 * gyy)
    return {
        "epsilon": float(Ttt),
        "p_x": float(Txx),
        "p_y": float(Tyy),
        "p_z": float(Tyy),
        "p_bar": float((Txx + 2.0 * Tyy) / 3.0),
    }


def physical_charge_densities(branch: Dict, p: ModelParams, rb: Optional[float] = None) -> Dict[str,float]:
    """Renormalized visible/dark charge densities from mixed canonical momenta.
    In the conventions of this solver, the physical grand-canonical densities are
    rho = delta2 * r^3 (phi' + mix * eta')
    rho_d = delta2 * r^3 (eta' + mix * phi').
    """
    if rb is None:
        rb = p.r_max
    y = np.asarray(branch["solution"].sol(rb), dtype=float)
    phip, etap = y[5], y[7]
    return {
        "rho": float(p.delta2 * rb ** 3 * (phip + p.mix * etap)),
        "rho_d": float(p.delta2 * rb ** 3 * (etap + p.mix * phip)),
    }


def entropy_density(branch: Dict) -> float:
    """Bekenstein-Hawking entropy density in the normalization used by this solver.
    For the present action conventions, s = 2 pi * area_density, and area_density = 1
    because the horizon is fixed at r_h = 1 and the f-anisotropy cancels in det(g_ij).
    """
    return float(2.0 * math.pi)


def grand_potential_from_boundary_data(branch: Dict, p: ModelParams, mu: Optional[float] = None, rb: Optional[float] = None) -> Dict[str,float]:
    """Thermodynamic scalar reconstructed from finite boundary data.
    Omega = epsilon - T s - mu rho - nu rho_d
    With the corrected stress tensor and mixed charge densities this is cutoff-stable.
    """
    if mu is None:
        mu = branch.get("mu", None)
        if mu is None:
            raise ValueError("mu must be supplied for branches not storing it explicitly")
    st = holographic_stress_tensor(branch, p, rb=rb)
    qs = physical_charge_densities(branch, p, rb=rb)
    s = entropy_density(branch)
    omega = st["epsilon"] - branch["temperature"] * s - mu * qs["rho"] - p.nu_target * qs["rho_d"]
    out = dict(st)
    out.update(qs)
    out["s"] = s
    out["Omega_ren"] = float(omega)
    out["minus_p_bar"] = float(-st["p_bar"])
    return out


def validate_renormalization() -> Dict[str, Dict[str, float]]:
    """Small built-in validation summary used in the final report."""
    summary = {}
    # 1) isotropic AdS5-RN check: Omega -> -p_bar as rb increases
    p0 = ModelParams(alpha_dm=0.0, delta2=0.10, nu_target=0.0, r_max=80.0)
    iso = solve_isotropic(4.0, p0)
    vals = []
    for rb in [20.0, 40.0, 80.0]:
        d = grand_potential_from_boundary_data(iso, p0, mu=4.0, rb=rb)
        vals.append((rb, d["Omega_ren"], d["minus_p_bar"]))
    summary["iso_delta01_mu4"] = {
        "rb20_Omega": vals[0][1], "rb40_Omega": vals[1][1], "rb80_Omega": vals[2][1],
        "rb20_minus_pbar": vals[0][2], "rb40_minus_pbar": vals[1][2], "rb80_minus_pbar": vals[2][2],
    }
    # 2) branch-difference cutoff stability at mu_X=0
    p1 = ModelParams(alpha_dm=0.4, delta2=0.10, nu_target=0.0, r_max=60.0)
    iso1 = solve_isotropic(3.72, p1)
    an1 = solve_anisotropic(3.72, p1)
    vals = []
    for rb in [20.0, 30.0, 40.0, 50.0, 60.0]:
        oi = grand_potential_from_boundary_data(iso1, p1, mu=3.72, rb=rb)["Omega_ren"]
        oa = grand_potential_from_boundary_data(an1, p1, mu=3.72, rb=rb)["Omega_ren"]
        vals.append((rb, oa - oi))
    summary["diff_alpha04_nu0_mu372"] = {f"rb{int(rb)}_dOmega": dO for rb, dO in vals}
    # 3) branch-difference cutoff stability at mu_X=2
    p2 = ModelParams(alpha_dm=0.4, delta2=0.10, nu_target=2.0, r_max=60.0)
    iso2 = solve_isotropic(3.58, p2)
    an2 = solve_anisotropic(3.58, p2)
    vals = []
    for rb in [20.0, 30.0, 40.0, 50.0, 60.0]:
        oi = grand_potential_from_boundary_data(iso2, p2, mu=3.58, rb=rb)["Omega_ren"]
        oa = grand_potential_from_boundary_data(an2, p2, mu=3.58, rb=rb)["Omega_ren"]
        vals.append((rb, oa - oi))
    summary["diff_alpha04_nu2_mu358"] = {f"rb{int(rb)}_dOmega": dO for rb, dO in vals}
    return summary
