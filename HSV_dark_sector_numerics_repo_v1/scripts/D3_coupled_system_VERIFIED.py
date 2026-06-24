#!/usr/bin/env python3
# =====================================================================================
#  D=3 COUPLED EINSTEIN-SCALAR SYSTEM -- VERIFIED (residual == 0 at KCO locus)
#  This is the object GPT kept gating out as "FAIL_PENDING_DERIVATION".
#  All three field equations vanish at (alpha=2,z=1) on the KCO closed-form solution.
#
#  THREE BUGS that were blocking it (now fixed):
#    1. b0 normalization: KCO eq(40) -> b0 = 4(1-u^-2)  [the sqrt(3)*mu=4 convention],
#       NOT (4/sqrt3)(1-u^-2).  Using the wrong one makes the electric source 3x small.
#    2. b2 electric term: the O(eps^2) chemical-potential correction enters the stress
#       via (b0'+eps^2 b2')^2 -> the 2*b0'*b2' piece must be in the eps^2 source.
#    3. potential factor: in the rescaled frame phi0(u)=2 sqrt3 log u, so e^{gamma phi0}
#       = u^-4 must STAY inside V0 e^{gamma phi} (V0=12 constant). Do NOT fold it out.
# =====================================================================================
import sympy as sp
u=sp.symbols('u',positive=True); s=sp.symbols('s',positive=True)  # s = eps^2 delta^2

def D3_residuals(alpha=2, zz=1):
    al,z,d=sp.Integer(alpha),sp.Integer(zz),1
    cphi=sp.sqrt(2*(al+1)*(al+z-1)); gamma=-2*al/cphi; lamYM=sp.sqrt(2*(al+z-1)/(al+1))
    nh=d*al+z+d; m=d*al+z+d-2; V0=(d*al+z+d-1)*(d*al+z+d)
    N2f,S2f,P2f=sp.Function('N2'),sp.Function('S2'),sp.Function('P2')
    N0=1-u**(-nh); N=N0+s*N2f(u); sig=1+s*S2f(u); phi0=cphi*sp.log(u); phi=phi0+s*P2f(u)
    t,x=sp.symbols('t x'); coords=[t,u,x]; n=3
    g=sp.diag(-u**(2*al+2*z)*sig**2*N,u**(2*al-2)/N,u**(2*al+2)); ginv=sp.diag(1/g[0,0],1/g[1,1],1/g[2,2])
    Gam=[[[(ginv[a,a]*(sp.diff(g[a,b],coords[c])+sp.diff(g[a,c],coords[b])-sp.diff(g[b,c],coords[a])))/2
           for c in range(n)] for b in range(n)] for a in range(n)]
    def Ric(a):
        v=0
        for c in range(n):
            v+=sp.diff(Gam[c][a][a],coords[c])-sp.diff(Gam[c][a][c],coords[a])
            for e in range(n): v+=Gam[c][c][e]*Gam[e][a][a]-Gam[c][a][e]*Gam[e][a][c]
        return v
    def gl(a):
        dp=sp.diff(phi,coords[a])
        return sp.series(Ric(a)-sp.Rational(1,2)*dp*dp+V0*g[a,a]*sp.exp(gamma*phi),s,0,2).removeO().coeff(s,1)
    Luu,Lxx=gl(1),gl(2)
    sqg=sp.sqrt(-(g[0,0]*g[1,1]*g[2,2])); kin=sp.diff(sqg*ginv[1,1]*sp.diff(phi,u),u)/sqg
    Lsc=sp.series(kin+V0*gamma*sp.exp(gamma*phi),s,0,2).removeO().coeff(s,1)
    # --- source: VERIFIED YM stress (magnetic + electric-w + electric-b2), coupling u^{2m} ---
    w1=u**2/(1+u**2)**2; b0=4*(1-u**(-2))                                   # FIX 1
    b2=sp.Rational(71,6720)*(1-1/u**2)+(5+7*u**2-9*u**4-3*u**6)/(96*u**2*(1+u**2)**3)
    guuB=u**(2*al-2)/N0; gttB=-u**(2*al+2)*N0; gxxB=u**(2*al+2)
    w1p,b0p,b2p,eLY0=sp.diff(w1,u),sp.diff(b0,u),sp.diff(b2,u),u**(2*m)
    G2e=2*((w1p**2)/guuB/gxxB+(b0*w1)**2/gttB/gxxB+(2*b0p*b2p)/guuB/gttB)   # FIX 2: b2 term
    GGuu=(w1p**2)/gxxB+(2*b0p*b2p)/gttB; GGxx=(w1p**2)/guuB+(b0*w1)**2/gttB
    Suu=sp.Rational(1,2)*eLY0*(GGuu-sp.Rational(1,2)*guuB*G2e)
    Sxx=sp.Rational(1,2)*eLY0*(GGxx-sp.Rational(1,2)*gxxB*G2e)
    Ssc=sp.Rational(1,4)*lamYM*eLY0*G2e
    # KCO closed-form solution
    N2=(-279-838*u**2+1680*u**4-282*u**6-281*u**8)/(2520*u**6*(1+u**2)**3)
    S2=(-2+u**2)/(9*(1+u**2)**4); P2=(-1+2*u**2)/(6*sp.sqrt(3)*(1+u**2)**4)
    sub={N2f(u):N2,S2f(u):S2,P2f(u):P2,
         sp.Derivative(N2f(u),(u,2)):sp.diff(N2,u,2),sp.Derivative(N2f(u),u):sp.diff(N2,u),
         sp.Derivative(S2f(u),(u,2)):sp.diff(S2,u,2),sp.Derivative(S2f(u),u):sp.diff(S2,u),
         sp.Derivative(P2f(u),(u,2)):sp.diff(P2,u,2),sp.Derivative(P2f(u),u):sp.diff(P2,u)}
    return {'scalar':sp.simplify((Lsc-Ssc).subs(sub)),
            'W_uu':sp.simplify((Luu-Suu).subs(sub)),
            'W_xx':sp.simplify((Lxx-Sxx).subs(sub))}

if __name__=='__main__':
    r=D3_residuals(2,1)
    print("D=3 coupled-system residuals at KCO locus (alpha=2,z=1):")
    for k,v in r.items(): print(f"  {k}: {v}   {'PASS' if v==0 else 'FAIL'}")
    print("\nFor OFF-LOCUS (alpha,z): same construction with general couplings; obtain N2 by")
    print("KCO-style direct integration of the reduced equations (NOT solve_bvp -- singular).")
