#!/usr/bin/env python3
"""Generate v166 dense scalar-BVP CSVs.

This is data-generation code only. It produces scalar BVP and perturbative
Case-III-b response CSVs; it does not plot figures.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp

PHI_SOURCE = 0.4
DELTA_REF = 1.10
W_REF = 0.10

def solve_scalar_bvp(D:int, theta:float, z:float, Delta:float=DELTA_REF, U:float=80.0, eps:float=1e-5, source_amp:float=PHI_SOURCE, tol:float=1e-5):
    d = D - 2
    nh = d*theta + z + d
    A = d*theta + z + d + 1
    m2 = Delta*(Delta - (A - 1))
    def f(u): return 1 - u**(-nh)
    def fp(u): return nh*u**(-nh-1)
    def P(u): return u**A*f(u)
    def Pp(u): return A*u**(A-1)*f(u) + u**A*fp(u)
    def ode(u, y):
        return np.vstack((y[1], (m2*u**(A-2)*y[0] - Pp(u)*y[1]) / P(u)))
    def bc(ya, yb):
        # Regular horizon condition and fixed UV source.
        return np.array([nh*ya[1] - m2*ya[0], yb[0] - source_amp*U**(-Delta)])
    u = np.geomspace(1+eps, U, 360)
    y0 = np.vstack((source_amp*u**(-Delta), -Delta*source_amp*u**(-Delta-1)))
    sol = solve_bvp(ode, bc, u, y0, tol=tol, max_nodes=18000, verbose=0)
    if sol.status != 0:
        raise RuntimeError(f"solve_bvp failed for D={D}, theta={theta}, z={z}, Delta={Delta}: {sol.message}")
    return sol, dict(D=D, d=d, theta=theta, z=z, Delta=Delta, U=U, nh=nh, A=A, m2=m2, status=sol.status, message=sol.message)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('data'), help='output directory')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for D in [3,4,5]:
        for theta in np.linspace(0,4,81):
            sol, meta = solve_scalar_bvp(D, float(theta), 2.0)
            ustar = min(1.0/W_REF, meta['U'])
            phi = float(sol.sol(ustar)[0])
            rows.append({**meta, 'scan':'theta', 'x':float(theta), 'W':W_REF, 'u_star_proxy':ustar, 'Phi_at_ustar':phi, 'relative_portal_shift':phi*phi})
        for z in np.linspace(1,5,81):
            sol, meta = solve_scalar_bvp(D, 1.0, float(z))
            ustar = min(1.0/W_REF, meta['U'])
            phi = float(sol.sol(ustar)[0])
            rows.append({**meta, 'scan':'z', 'x':float(z), 'W':W_REF, 'u_star_proxy':ustar, 'Phi_at_ustar':phi, 'relative_portal_shift':phi*phi})
    pd.DataFrame(rows).to_csv(args.out/'O12_perturbative_scalar_BVP_theta_z_dense_v166.csv', index=False)

    ref_loci = {3:(2.0,1.0), 4:(0.5,1.0), 5:(0.0,1.0)}
    Ws = np.linspace(0.05,0.30,121)
    width_rows, profile_rows = [], []
    for D,(theta,z) in ref_loci.items():
        for Delta in [1.10,1.25,1.50,1.75,2.00,2.25,2.50,3.00,3.50,4.00]:
            sol, meta = solve_scalar_bvp(D, theta, z, Delta=Delta)
            if abs(Delta-DELTA_REF) < 1e-12:
                for u in np.geomspace(1.001,80,300):
                    phi = float(sol.sol(u)[0])
                    profile_rows.append({**meta, 'u':u, 'Phi':phi, 'Phi2':phi*phi})
            for W in Ws:
                ustar = min(1.0/W, meta['U'])
                phi = float(sol.sol(ustar)[0])
                width_rows.append({**meta, 'W':W, 'u_star_proxy':ustar, 'Phi_at_ustar':phi, 'relative_portal_shift':phi*phi})
    pd.DataFrame(width_rows).to_csv(args.out/'O12_perturbative_scalar_BVP_width_dense_v166.csv', index=False)
    pd.DataFrame(profile_rows).to_csv(args.out/'scalar_BVP_profiles_dense_v166.csv', index=False)

if __name__ == '__main__':
    main()
