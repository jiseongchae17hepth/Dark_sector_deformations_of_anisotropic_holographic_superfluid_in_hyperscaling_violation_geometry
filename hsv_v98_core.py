from __future__ import annotations
import math, numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import mpmath as mp

mp.mp.dps = 45

# ---------- exponents ----------
def nh(d,a,z): return d*a+z+d
def mexp(d,a,z): return d*a+z+d-2
def Aexp(d,a,z): return d*a+d+3*z-3
def Delta(d,a,z): return Aexp(d,a,z)-1
def Qexp(d,a,z): return d*a+d+5*z-5
def N0(u,d,a,z): return 1-np.asarray(u,float)**(-nh(d,a,z))

# ---------- accurate zero mode in xi=1/u ----------
def solve_zero_mode_xi(d,a,z,eps=2e-5,tol=3e-7,guess_lam=16.0,max_nodes=100000):
    h=nh(d,a,z); m=mexp(d,a,z); A=Aexp(d,a,z); De=Delta(d,a,z)
    if m<=0 or De<=0: return None
    x=np.unique(np.r_[np.geomspace(eps,.02,60),np.linspace(.02,.98,260),1-np.geomspace(eps,.02,60)[::-1]])
    F0=np.ones_like(x); G0=np.zeros_like(x)
    def K(x): return x**A*(1-x**h)
    def Kp(x): return A*x**(A-1)*(1-x**h)-h*x**(A+h-1)
    def P(x): return (A-1)*h*x**(A+h-2)
    def Q(x): return x**Qexp(d,a,z)*(1-x**m)**2/(1-x**h)
    def ode(x,y,p):
        lam=p[0]
        return np.vstack([y[1],(P(x)*y[0]-lam*Q(x)*y[0]-Kp(x)*y[1])/K(x)])
    def bc(ya,yb,p): return np.array([ya[0]-1,ya[1],yb[1]+De*yb[0]])
    sol=solve_bvp(ode,bc,x,np.vstack([F0,G0]),p=np.array([guess_lam]),tol=tol,bc_tol=tol*.05,max_nodes=max_nodes)
    return sol

def make_zero_profile_xi(sol,d,a,z):
    De=Delta(d,a,z); m=mexp(d,a,z); mu=math.sqrt(max(float(sol.p[0]),0.0))
    xmin=float(sol.x[0]); xmax=float(sol.x[-1])
    def w(u):
        aa=np.asarray(u,float); xi=1/aa; out=np.empty_like(aa)
        mid=(xi>=xmin)&(xi<=xmax)
        if np.any(mid):
            F=sol.sol(xi[mid])[0]; out[mid]=xi[mid]**De*F
        low=xi<xmin
        if np.any(low): out[low]=xi[low]**De # F(0)=1
        high=xi>xmax
        if np.any(high):
            Fh,Gh=sol.sol(np.array([xmax])); wh=xmax**De*Fh[0]
            out[high]=wh
        return out
    def wp(u):
        aa=np.asarray(u,float); xi=1/aa; out=np.empty_like(aa)
        mid=(xi>=xmin)&(xi<=xmax)
        if np.any(mid):
            F,G=sol.sol(xi[mid]); dwdxi=De*xi[mid]**(De-1)*F+xi[mid]**De*G; out[mid]=-xi[mid]**2*dwdxi
        low=xi<xmin
        if np.any(low): out[low]=-De*aa[low]**(-De-1)
        high=xi>xmax
        if np.any(high): out[high]=0.0
        return out
    def b0(u):
        aa=np.asarray(u,float); return mu*(1-aa**(-m))
    return mu,w,wp,b0

# ---------- D3 b2,w3 Fredholm ----------
def solve_b2_w3_D3(a,z,mu,w,wp,U=100.0,eps=1e-5,tol=1e-7,mu2_guess=0.01):
    d=1; h=nh(d,a,z); m=mexp(d,a,z); A=Aexp(d,a,z); De=Delta(d,a,z); r=a+z-4
    x=np.geomspace(1+eps,U,900)
    bg=mu2_guess*(1-x**(-m)); bgp=mu2_guess*m*x**(-m-1)
    y=np.vstack([bg,bgp,np.zeros_like(x),np.zeros_like(x)])
    def ode(u,y,p):
        b2,b2p,w3,w3p=y
        N=1-u**(-h); b=mu*(1-u**(-m)); W=w(u)
        Pb=u**(m+1); Pbp=(m+1)*u**m
        rhsb=u**r*b*W*W/N
        b2pp=(rhsb-Pbp*b2p)/Pb
        Pw=u**A*N; Pwp=A*u**(A-1)*N+u**A*h*u**(-h-1)
        Q=u**r*b*b/N; R=2*u**r*b*b2*W/N
        w3pp=(-Q*w3-R-Pwp*w3p)/Pw
        return np.vstack([b2p,b2pp,w3p,w3pp])
    def bc(ya,yb,p):
        mu2=p[0]
        return np.array([ya[0], U*yb[1]+m*(yb[0]-mu2), ya[3], yb[2], U*yb[3]+De*yb[2]])
    sol=solve_bvp(ode,bc,x,y,p=np.array([mu2_guess]),tol=tol,bc_tol=tol*.05,max_nodes=200000)
    return sol

def make_b2_profile(sol,m,U=100.0):
    mu2=float(sol.p[0]); rho=(float(sol.sol(U)[0])-mu2)*U**m
    def b2(u):
        aa=np.asarray(u,float); out=np.empty_like(aa); mask=aa<=U
        out[mask]=sol.sol(aa[mask])[0]; out[~mask]=mu2+rho*aa[~mask]**(-m)
        return out
    def b2p(u):
        aa=np.asarray(u,float); out=np.empty_like(aa); mask=aa<=U
        out[mask]=sol.sol(aa[mask])[1]; out[~mask]=-m*rho*aa[~mask]**(-m-1)
        return out
    return mu2,b2,b2p


# ---------- D3 b2 by Fredholm solvability (preferred direct integration) ----------
def solve_b2_fredholm_D3(a,z,mu,w,U=250.0,n=24000):
    from scipy.integrate import cumulative_trapezoid, trapezoid
    m=a+z-1; h=a+z+1; r=a+z-4
    x=np.linspace(0,1,n); u=1+1e-9+(U-1-1e-9)*x*x
    N=1-u**(-h); b=mu*(1-u**(-m)); W=w(u)
    R=u**r*b*W*W/N; P=u**(m+1)
    I=cumulative_trapezoid(R,u,initial=0.0)
    B1=cumulative_trapezoid(1/P,u,initial=0.0)
    B0=cumulative_trapezoid(I/P,u,initial=0.0)
    wt=2*u**r*b*W*W/N
    J1=trapezoid(wt*B1,u); J0=trapezoid(wt*B0,u); C=-J0/J1
    b2=C*B1+B0; b2p=(C+I)/P
    Iinf=I[-1]; mu2=b2[-1]+(C+Iinf)/(m*U**m); rho=(b2[-1]-mu2)*U**m
    bi=PchipInterpolator(u,b2); bpi=PchipInterpolator(u,b2p)
    def bf(q):
        q=np.asarray(q,float); out=np.empty_like(q); mask=q<=U
        out[mask]=bi(q[mask]); out[~mask]=mu2+rho*q[~mask]**(-m); return out
    def bpf(q):
        q=np.asarray(q,float); out=np.empty_like(q); mask=q<=U
        out[mask]=bpi(q[mask]); out[~mask]=-m*rho*q[~mask]**(-m-1); return out
    return mu2,bf,bpf,{'u':u,'b2':b2,'b2p':b2p,'C_horizon_flux':C,'J0':J0,'J1':J1,'U':U}

# ---------- D3 exact linearized sources ----------
def d3_source_arrays(u,a,z,mu,w,wp,b2p,weight=None):
    u=np.asarray(u,float); h=a+z+1; m=a+z-1; N=1-u**(-h); b=mu*(1-u**(-m)); bp=mu*m*u**(-m-1)
    guu=u**(2*a-2)/N; gtt=-u**(2*a+2*z)*N; gxx=u**(2*a+2); ey=u**(2*m)
    W=w(u); Wp=wp(u); B2p=b2p(u); wt=np.ones_like(u) if weight is None else weight(u)
    G2=2*wt*(Wp**2/guu/gxx+(b*W)**2/gtt/gxx+2*bp*B2p/guu/gtt)
    GGtt=wt*(2*bp*B2p/guu+(b*W)**2/gxx)
    GGxx=wt*(Wp**2/guu+(b*W)**2/gtt)
    Stt=.5*ey*(GGtt-.5*gtt*G2)
    Sxx=.5*ey*(GGxx-.5*gxx*G2)
    lam=math.sqrt(2*m/(a+1)); Ssc=.25*lam*ey*G2
    return Stt,Sxx,Ssc

def d3_rhs_point(u,y,a,z,Stt,Sxx,Sxxp,Ssc):
    N,V,S,P,Q=y; h=a+z+1; m=a+z-1; c=math.sqrt(2*(a+1)*m); B=1-u**(-h); Bp=h*u**(-h-1); A=a+1; k=c*h/A; M=a+z
    G=V+h*N/u+k*P/u+Sxx/(A*u**3); Sp=-G/B
    rest=h*(V/u-N/u**2)+k*(Q/u-P/u**2)+Sxxp/(A*u**3)-3*Sxx/(A*u**4)
    cNpp=.5*u**(2*z+2)*B; cSpp=u**(2*z+2)*B**2
    cNp=.5*(3*a+3*z+2)*u**(2*z+1)*B
    cSp=.5*h*u**(-2*a-1)*(u**h-1)*(4*u**h-1)
    cN=M*h*u**(2*z)*B; cP=2*a*M*h/c*u**(2*z)*B
    L0=cNp*V+cSp*Sp+cN*N+cP*P; coeff=cNpp-cSpp/B
    Npp=(Stt-cSpp*(Bp/B**2)*G+cSpp/B*rest-L0)/coeff
    cPpp=u**(-2*a+2)*B; cNpS=c*u**(1-2*a); cSpS=c*u**(-2*a+1)*B
    cPpS=u**(-3*a-z)*(u**h*(a+z+2)-1); cNS=c*h*u**(-2*a)
    R=a*a+a*z-a-z+1; cPS=2*h*R/m*u**(-2*a)
    Ppp=(Ssc-cNpS*V-cSpS*Sp-cPpS*Q-cNS*N-cPS*P)/cPpp
    return np.array([V,Npp,Sp,Q,Ppp])

def solve_D3_metric(a,z,mu,w,wp,b2p,weight=None,U=80.0,eps=1e-4,tol=2e-6,guess=None):
    xx=np.linspace(0,1,6500); ug=1+1e-7+(U-1-1e-7)*xx**2
    st,sx,ss=d3_source_arrays(ug,a,z,mu,w,wp,b2p,weight)
    sxI=PchipInterpolator(ug,sx,extrapolate=True); stI=PchipInterpolator(ug,st,extrapolate=True); ssI=PchipInterpolator(ug,ss,extrapolate=True); sxIp=sxI.derivative()
    uh=1+eps; x=np.geomspace(uh,U,800)
    if guess is None: y=np.zeros((5,x.size))
    else: y=guess.sol(x)
    def ode(u,y):
        out=np.empty_like(y); ST=stI(u); SX=sxI(u); SXP=sxIp(u); SS=ssI(u)
        for j,q in enumerate(u): out[:,j]=d3_rhs_point(q,y[:,j],a,z,float(ST[j]),float(SX[j]),float(SXP[j]),float(SS[j]))
        return out
    qh=1+1e-7; sxh=float(sxI(qh)); ssh=float(ssI(qh))
    h=a+z+1; m=a+z-1; c=math.sqrt(2*(a+1)*m); A=a+1; k=c*h/A; R=a*a+a*z-a-z+1
    def bc(ya,yb):
        return np.array([ya[0]-eps*ya[1], ya[1]+k*ya[3]+sxh/A, ssh-c*ya[1]-h*ya[4]-2*h*R/m*ya[3], yb[2], yb[3]])
    sol=solve_bvp(ode,bc,x,y,tol=tol,bc_tol=tol*.05,max_nodes=180000)
    return sol,(stI,sxI,ssI)

def make_D3_N2(sol,a,z,U=80.0):
    h=a+z+1; C=float(sol.sol(U)[0])*U**h
    def N2(u):
        aa=np.asarray(u,float); out=np.empty_like(aa); mask=aa<=U
        out[mask]=sol.sol(aa[mask])[0]; out[~mask]=C*aa[~mask]**(-h)
        return out
    return N2,C

# ---------- D4/D5 exact traceless response ----------
def source_aniso(u,d,a,z,mu,w,wp,weight=None):
    u=np.asarray(u,float); N=1-u**(-nh(d,a,z)); b=mu*(1-u**(-mexp(d,a,z))); wt=np.ones_like(u) if weight is None else weight(u)
    raw=wt*(u**Aexp(d,a,z)*N*wp(u)**2-u**(d*a+z+d-5)*b*b*w(u)**2/N)
    k=-0.25 if d==2 else 1/3
    return k*raw

def solve_A_response(d,a,z,mu,w,wp,weight=None,U=350.0,n=9000):
    x=np.linspace(0,1,n); u=1+1e-8+(U-1-1e-8)*x**2
    S=source_aniso(u,d,a,z,mu,w,wp,weight); h=nh(d,a,z); P=u**(h+1)*(1-u**(-h))
    flux=cumulative_trapezoid(S,u,initial=0.0); Ap=flux/P
    Finf=float(flux[-1]); tailU=-Finf/(h*U**h)
    rev=cumulative_trapezoid(Ap[::-1],u[::-1],initial=0.0)
    Aresp=tailU+rev[::-1]
    interp=PchipInterpolator(u,Aresp,extrapolate=False)
    def Afun(arr):
        aa=np.asarray(arr,float); out=np.empty_like(aa); mask=aa<=U
        out[mask]=interp(aa[mask]); out[~mask]=-Finf/(h*aa[~mask]**h)
        return out
    return Afun, {'u':u,'source':S,'flux':flux,'Aprime':Ap,'A':Aresp,'Finf':Finf,'tailU':tailU}

# ---------- endpoint-safe widths and HEE ----------
def _mp(v): return mp.mpf(str(float(v)))

def width_from_ut(ut,d,a,z):
    ut=mp.mpf(ut); p=mp.mpf(2*d*(a+1)); h=mp.mpf(nh(d,a,z))
    def ft(t):
        if abs(t) < mp.mpf('1e-30'):
            return 2/(mp.sqrt(p)*mp.sqrt(1-ut**(-h)))
        x=1-t*t
        if x==1: return 2/(mp.sqrt(p)*mp.sqrt(1-ut**(-h)))
        if x<=0: return mp.mpf('0')
        om1=-mp.expm1(h*(mp.log(x)-mp.log(ut))); om2=-mp.expm1(p*mp.log(x))
        if om2==0: return 2/(mp.sqrt(p)*mp.sqrt(1-ut**(-h)))
        return 2*t*x**(d*(a+1))/(mp.sqrt(om1)*mp.sqrt(om2))
    return 2/ut*mp.quad(ft,[0,1])

def ut_for_width(W,d,a,z):
    W=mp.mpf(str(W)); lo=mp.mpf('1.0000000001'); hi=mp.mpf(2)
    while width_from_ut(hi,d,a,z)>W: hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if width_from_ut(mid,d,a,z)>W: lo=mid
        else: hi=mid
    return (lo+hi)/2

def pure_width_constant(d,a):
    q=mp.mpf(d)*(mp.mpf(str(a))+1); p=2*q
    return 2/p*mp.beta((q+1)/p, mp.mpf('0.5'))

def pure_ut_for_width(W,d,a):
    return pure_width_constant(d,a)/mp.mpf(str(W))

def hee_D3(W,a,z,N2fun):
    ut=pure_ut_for_width(W,1,a); p=mp.mpf(2*(a+1)); h=mp.mpf(a+z+1)
    def ft(t):
        if abs(t) < mp.mpf('1e-30'):
            uf=float(ut); Nv=mp.mpf(str(float(N2fun(np.array([uf]))[0]))); Nbg=1-ut**(-h)
            return -ut**a*Nv/(Nbg**mp.mpf('1.5')*mp.sqrt(p))
        x=1-t*t
        if x==1:
            uf=float(ut); Nv=mp.mpf(str(float(N2fun(np.array([uf]))[0]))); Nbg=1-ut**(-h)
            return -ut**a*Nv/(Nbg**mp.mpf('1.5')*mp.sqrt(p))
        if x<=0: return mp.mpf('0')
        u=ut/x
        uf=float(u) if u < mp.mpf('1e300') else float('inf')
        Nv=mp.mpf(str(float(N2fun(np.array([uf]))[0])))
        Nbg=-mp.expm1(-h*mp.log(u))
        return 2*t*(-mp.mpf('0.5')*u**(a-1)*Nv/(Nbg**mp.mpf('1.5')*mp.sqrt(-mp.expm1(p*mp.log(x))))*ut/x**2)
    return mp.quad(ft,[0,1]),ut

def hee_O12(W,d,a,z,Afun):
    ut=pure_ut_for_width(W,d,a); p=mp.mpf(2*d*(a+1)); h=mp.mpf(nh(d,a,z))
    def ft(t):
        if abs(t) < mp.mpf('1e-30'):
            if d==3: return mp.mpf('0')
            uf=float(ut); Av=mp.mpf(str(float(Afun(np.array([uf]))[0]))); Nfac=mp.sqrt(1-ut**(-h))
            return 4*Av*ut**(d*(a+1)-1)/(Nfac*mp.sqrt(p))
        x=1-t*t
        if x==1:
            if d==3: return mp.mpf('0')
            uf=float(ut); Av=mp.mpf(str(float(Afun(np.array([uf]))[0]))); Nfac=mp.sqrt(1-ut**(-h))
            return 4*Av*ut**(d*(a+1)-1)/(Nfac*mp.sqrt(p))
        if x<=0: return mp.mpf('0')
        u=ut/x; uf=float(u) if u < mp.mpf('1e300') else float('inf')
        Av=mp.mpf(str(float(Afun(np.array([uf]))[0]))); Nfac=mp.sqrt(-mp.expm1(h*(mp.log(x)-mp.log(ut))))
        if d==2: val=2*Av*u**(d*(a+1)-2)/(Nfac*mp.sqrt(-mp.expm1(p*mp.log(x))))*ut/x**2
        else: val=6*Av*u**(3*a+1)*mp.sqrt(-mp.expm1(p*mp.log(x)))/Nfac*ut/x**2
        return 2*t*val
    return mp.quad(ft,[0,1]),ut

# analytic anchors
def D3_N2_an(u):
    u=np.asarray(u,float); return (-279-838*u**2+1680*u**4-282*u**6-281*u**8)/(2520*u**6*(1+u**2)**3)
def D4_H2_an(u):
    u=np.asarray(u,float); return (1-2*u**2)/(24*(1+u**2)**4)
def D5_f2_an(u):
    u=np.asarray(u,float); return -(1-2*u**2)/(18*(1+u**2)**4)
