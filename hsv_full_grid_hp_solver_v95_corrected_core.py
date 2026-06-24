#!/usr/bin/env python3
"""
Corrected core for HSV off-locus HEE solver.
Key patch relative to uploaded hsv_full_grid_hp_solver.py:
  Aexp(d,a,z) = d*a + d + 3*z - 3, not d*a + 3*z.
  Delta = Aexp - 1.
  source_aniso magnetic exponent uses Aexp.
This file intentionally refuses to emit D=3 off-locus rows until derive_D3_system_general()
returns a locus-residual-passing coupled {N2,sigma2,phi2,b2} direct-integration system.
"""
import math, numpy as np
from scipy.integrate import solve_bvp

def nh(d,a,z): return d*a+z+d
def mexp(d,a,z): return d*a+z+d-2
def Aexp(d,a,z): return d*a+d+3*z-3
def Delta(d,a,z): return Aexp(d,a,z)-1
def N0(u,d,a,z): return 1-np.asarray(u,float)**(-nh(d,a,z))
def source_aniso(u,d,a,z,mu,w,wp):
    u=np.asarray(u,float); N=N0(u,d,a,z); b=mu*(1-u**(-mexp(d,a,z)))
    return (u**Aexp(d,a,z)*N*wp(u)**2 - u**(d*a+z+d-5)*b*b*w(u)**2/np.maximum(N,1e-300))/3.0

def solve_zero_mode_corrected(d,a,z,U=80.0,eps=1e-5,tol=1e-6):
    A=Aexp(d,a,z); m=mexp(d,a,z); h=nh(d,a,z); Del=Delta(d,a,z)
    if m<=0 or Del<=0: raise ValueError('invalid grid point: m<=0 or Delta<=0')
    x=np.geomspace(1+eps,U,500)
    y=np.vstack([x**(-Del), -Del*x**(-Del-1)]); y/=y[0,0]
    def fun(u,y,p):
        mu=p[0]; N=1-u**(-h); K=u**A*N
        Kp=A*u**(A-1)*N + u**A*(h*u**(-h-1))
        Q=u**(d*a+z+d-5)*(1-u**(-m))**2/N
        return np.vstack([y[1], (-mu**2*Q*y[0]-Kp*y[1])/K])
    def bc(ya,yb,p): return np.array([ya[0]-1, ya[1], U*yb[1]+Del*yb[0]])
    return solve_bvp(fun,bc,x,y,p=np.array([4.0]),tol=tol,bc_tol=tol*0.1,max_nodes=30000)

if __name__ == '__main__':
    for D,d,a,z,ref in [(3,1,2,1,4),(4,2,0.5,1,4),(5,3,0,1,4),(5,3,1,1,5.679551)]:
        sol=solve_zero_mode_corrected(d,a,z)
        print(f'D={D} a={a} z={z}: mu={sol.p[0]:.9f}, ref={ref}, success={sol.success}')
