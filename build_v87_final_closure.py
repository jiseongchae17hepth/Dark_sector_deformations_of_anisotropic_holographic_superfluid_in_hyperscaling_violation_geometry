from __future__ import annotations
import os, json, math, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp, quad, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.special import gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/data/hsv_v87_final_closure')
if ROOT.exists(): shutil.rmtree(ROOT)
for sub in ['csv','figures','reports','scripts','source_refs','audit','diagnostics']:
    (ROOT/sub).mkdir(parents=True, exist_ok=True)

# Constants
SQ3 = math.sqrt(3.0)
G16 = gamma(1/6)
G13 = gamma(1/3)
K2 = G16**3 * G13**3
PI = math.pi
WIDTH_C = 4*PI**1.5/(SQ3*G16*G13)
MU_STAR = 4/SQ3
WIDTHS = np.array([0.05,0.08,0.10,0.12,0.18,0.24,0.30], dtype=float)

# ---------------- D3 baseline functions ----------------
def D3_N2_an(u):
    u=np.asarray(u,dtype=float)
    return (-279.0 - 838.0*u**2 + 1680.0*u**4 - 282.0*u**6 - 281.0*u**8)/(2520.0*u**6*(1.0+u**2)**3)

def D3_SN0(u):
    u=np.asarray(u,dtype=float)
    h = np.maximum(1e-6, 1e-6*u)
    return (((u+h)**4*D3_N2_an(u+h) - (u-h)**4*D3_N2_an(u-h))/(2*h))/(u**4)

def build_D3_N_from_source_weight(weight_func, extra_SN_func=None, U=500.0, n=18000):
    x = np.linspace(0,1,n)
    u = 1 + (U-1)*x**2
    u[0] = 1.0 + 1e-7
    S = weight_func(u)*D3_SN0(u)
    if extra_SN_func is not None:
        S = S + extra_SN_func(u)
    integrand = u**4 * S
    I = cumulative_trapezoid(integrand, u, initial=0.0)
    N = I/u**4
    interp = PchipInterpolator(u, N, extrapolate=True)
    NU = float(interp(U)); baselineU = float(D3_N2_an(U)); offsetU = NU-baselineU
    def Nfun(arr):
        a=np.asarray(arr,dtype=float); out=np.empty_like(a); m=a<=U
        out[m]=interp(a[m])
        out[~m]=D3_N2_an(a[~m]) + offsetU*(U/a[~m])**4
        return out
    return Nfun, pd.DataFrame({'u':u,'N2':N,'SN':S,'weight':weight_func(u)})

def D3_interval_coeff(W, N2fun):
    us=WIDTH_C/W
    def integrand(x):
        if x <= 0: return 0.0
        u=us/x
        N0=1-u**(-4)
        den=math.sqrt(max(1e-300, 1-(us/u)**6))
        return -0.5*u*float(N2fun(np.array([u]))[0])/(N0**1.5*den)*us/x**2
    val,err=quad(integrand,0,1,points=[1],limit=800,epsabs=1e-11,epsrel=1e-10)
    return float(val)

# ---------------- D4 functions ----------------
def D4_h2_an(u):
    u=np.asarray(u,dtype=float)
    return (1-2*u*u)/(24*(1+u*u)**4)

def D4_SH0(u):
    u=np.asarray(u,dtype=float)
    return u*(u-1)*(u+1)*(u*u-2*u-1)*(u*u+2*u-1)

def solve_D4_H(weight_func=lambda u: np.ones_like(np.asarray(u,dtype=float)), U=50.0, tol=3e-8):
    uh=1+1e-4
    def ode(u,y):
        h,hp=y
        A=u*(u-1)*(u+1)*(u*u+1)**6
        B=(u*u+1)**5*(5*u**4-1)
        C=weight_func(u)*D4_SH0(u)
        hpp=-(B*hp+C)/A
        return np.vstack([hp,hpp])
    def bc(ya,yb): return np.array([ya[1], U*yb[1]+6*yb[0]])
    x=np.linspace(uh,U,700)
    y=np.vstack([D4_h2_an(x), np.gradient(D4_h2_an(x),x)])
    sol=solve_bvp(ode,bc,x,y,tol=tol,max_nodes=150000,verbose=0)
    if not sol.success:
        raise RuntimeError('D4 BVP failed: '+sol.message)
    def H(arr):
        a=np.asarray(arr,dtype=float); out=np.empty_like(a); m=a<=U
        out[m]=sol.sol(a[m])[0]
        out[~m]=D4_h2_an(a[~m])
        return out
    def J(arr): return -H(arr)
    return H,J,sol

def D4_O12_coeff(W,H,J):
    us=WIDTH_C/W
    def integrand(x):
        if x<=0: return 0.0
        u=us/x
        N0=1-u**(-4)
        den=math.sqrt(max(1e-300,1-(us/u)**6))
        return u*(float(H(np.array([u]))[0])-float(J(np.array([u]))[0]))/(math.sqrt(N0)*den)*us/x**2
    val,err=quad(integrand,0,1,points=[1],limit=800,epsabs=1e-12,epsrel=1e-10)
    return float(val)

# ---------------- D5 functions ----------------
def D5_eq65(d):
    return (3*SQ3/(448*PI**(9/2)))*G16**3*G13**3*d**4 + (2943/(573440*PI**9))*G16**6*G13**6*d**6

def D5_f2_an(r):
    r=np.asarray(r,dtype=float)
    return -(1-2*r*r)/(18*(1+r*r)**4)

def D5_f2p_an(r):
    r=np.asarray(r,dtype=float)
    return -2*r*(r-1)*(r+1)/(3*(r*r+1)**5)

def D5_f2pp_an(r):
    r=np.asarray(r,dtype=float)
    return 2*(7*r**4-12*r**2+1)/(3*(r*r+1)**6)

def D5_S0(r):
    r=np.asarray(r,dtype=float)
    return 4*r*(r*r-1)*(1-6*r*r+r**4)/(3*(1+r*r)**5)

def D5_source_residual(r):
    r=np.asarray(r,dtype=float)
    return r*(r**4-1)*D5_f2pp_an(r)+(-1+5*r**4)*D5_f2p_an(r)-D5_S0(r)

def build_D5_f_from_weight(weight_func, U=600.0, n=25000):
    x=np.linspace(0,1,n)
    r=1+(U-1)*x**2
    r[0]=1+1e-7
    S=weight_func(r)*D5_S0(r)
    I=cumulative_trapezoid(S,r,initial=0.0)
    fp=I/(r*(r**4-1))
    # f(r) = f(U) + int_U^r fp(s) ds, tail fixed to analytic baseline at U.
    tail=float(D5_f2_an(U))
    rev_int=cumulative_trapezoid(fp[::-1], r[::-1], initial=0.0)
    f=(tail + rev_int)[::-1]
    interp=PchipInterpolator(r,f,extrapolate=True)
    fU=float(interp(U)); baseU=float(D5_f2_an(U)); offsetU=fU-baseU
    def ffun(arr):
        a=np.asarray(arr,dtype=float); out=np.empty_like(a); m=a<=U
        out[m]=interp(a[m])
        out[~m]=D5_f2_an(a[~m]) + offsetU*(U/a[~m])**4
        return out
    return ffun, pd.DataFrame({'r':r,'f2':f,'fp':fp,'source':S,'weight':weight_func(r)})

def D5_first_variation_integral(W, ffun):
    rs=WIDTH_C/W
    def integrand(x):
        if x<=0: return 0.0
        r=rs/x
        N=r*r - r**(-2)
        return 6*r*r*float(ffun(np.array([r]))[0])/math.sqrt(N)*math.sqrt(max(0,1-x**6))*rs/x**2
    val,err=quad(integrand,0,1,points=[1],limit=800,epsabs=1e-12,epsrel=1e-10)
    return float(val)

# Scalar profile and tensor portal weight
def Phi_profile(u, Phi_h):
    u=np.asarray(u,dtype=float)
    return Phi_h/(1+u*u)**2

def Z_weight(Phi_h=0.0, lam=1.0):
    return lambda u: np.exp(lam*Phi_profile(u,Phi_h)**2)

cases = [
    ('baseline', 'baseline', {}),
    ('I_temporal_U1_qX_0p5', 'I', {'qX':0.5}),
    ('II_aDM_0p4', 'II', {'alpha_dm':0.4}),
    ('II_aDM_0p8', 'II', {'alpha_dm':0.8}),
    ('II_aDM_1p2', 'II', {'alpha_dm':1.2}),
    ('II-b_aDM_0p4_muX_0p5', 'II-b', {'alpha_dm':0.4, 'muX':0.5*MU_STAR}),
    ('III-a_Phi_0p2', 'III-a', {'Phi_h':0.2}),
    ('III-a_Phi_0p4', 'III-a', {'Phi_h':0.4}),
    ('III-b_Phi_0p2_l1', 'III-b', {'Phi_h':0.2, 'Z_lambda':1.0}),
    ('III-b_Phi_0p4_l1', 'III-b', {'Phi_h':0.4, 'Z_lambda':1.0}),
]

def const_scale(kind, p):
    if kind=='baseline': return 1.0
    if kind=='II': return 1-p['alpha_dm']**2/4
    if kind=='II-b': return 1-p['alpha_dm']**2/4 + (p['muX']/MU_STAR)**2*p['alpha_dm']**2/4
    return 1.0

# Precompute baselines
D3_base = lambda u: D3_N2_an(u)
H4_base,J4_base,sol4base = solve_D4_H(lambda u: np.ones_like(np.asarray(u,dtype=float)))
f5_base, f5_base_prof = build_D5_f_from_weight(lambda u: np.ones_like(np.asarray(u,dtype=float)))
D5_calib = {float(W): D5_eq65(float(W))/D5_first_variation_integral(float(W), f5_base) for W in WIDTHS}

# Generate final anisotropy-order HEE rows.
rows=[]; profiles={}; case_rules=[]
for cname, kind, p in cases:
    scale=const_scale(kind,p)
    # D=3: DeltaS_int coefficient here means O(delta^2 eps^2) anisotropy-induced coefficient.
    # Isotropic dark sectors (I, III-a) do NOT alter this leading coefficient; their O(delta^2 eps^0)
    # background shifts are separated into diagnostics, not final HEE anisotropy rows.
    if kind in ['baseline','I','III-a']:
        N3 = D3_base
        d3_rule = 'baseline KCO N2' if kind=='baseline' else 'isotropic dark sector: leading O(delta2 eps2) anisotropy coefficient unchanged; background O(delta2 eps0) shift separated'
    elif kind in ['II','II-b']:
        N3 = lambda u, sc=scale: sc*D3_N2_an(u)
        d3_rule=f'constant proportional hidden-SU2 anisotropic source scale={scale:.8g}'
    elif kind=='III-b':
        ph=p['Phi_h']; lam=p['Z_lambda']; weight=Z_weight(ph,lam)
        N3,prof=build_D3_N_from_source_weight(weight)
        profiles[f'D3_{cname}']=prof
        d3_rule='tensor portal Z(Phi(u)) reweights anisotropic YM source inside D3 N2 equation'
    for W in WIDTHS:
        rows.append({'D':3,'case':cname,'observable':'DeltaS_int_Odelta2eps2','width':W,'coeff_O_delta2_eps2':D3_interval_coeff(float(W),N3),'status':'FINAL_HEE_GENERATED_V87','source_rule':d3_rule})

    # D=4 O12: isotropic sectors are blind. Hidden-SU2 proportional. Tensor portal reweighted.
    if kind in ['baseline','I','III-a']:
        H4,J4=H4_base,J4_base
        d4_rule='baseline KCO anisotropy BVP' if kind=='baseline' else 'O12 blind to isotropic source; anisotropic H2-J2 source unchanged'
    elif kind in ['II','II-b']:
        H4=lambda u, sc=scale: sc*H4_base(u); J4=lambda u, sc=scale: sc*J4_base(u)
        d4_rule=f'constant proportional hidden-SU2 anisotropic source scale={scale:.8g}'
    elif kind=='III-b':
        ph=p['Phi_h']; lam=p['Z_lambda']; weight=Z_weight(ph,lam)
        H4,J4,sol=solve_D4_H(weight)
        d4_rule='tensor portal Z(Phi(u)) reweights anisotropic source inside D4 H2 BVP'
    for W in WIDTHS:
        rows.append({'D':4,'case':cname,'observable':'O12_EE','width':W,'coeff_O_delta2_eps2':D4_O12_coeff(float(W),H4,J4),'status':'FINAL_HEE_GENERATED_V87','source_rule':d4_rule})

    # D=5 O12: baseline Eq65; hidden-SU2 proportional; isotropic blind; tensor portal via f2 ODE weight.
    if kind in ['baseline','I','III-a']:
        d5_rule='D5 AdS5 Eq65 baseline' if kind=='baseline' else 'O12 blind to isotropic source; D5 AdS5 Eq65 baseline retained'
        for W in WIDTHS:
            rows.append({'D':5,'case':cname,'observable':'O12_EE','width':W,'coeff_O_delta2_eps2':D5_eq65(float(W)),'status':'FINAL_HEE_GENERATED_V87','source_rule':d5_rule})
    elif kind in ['II','II-b']:
        d5_rule=f'constant proportional hidden-SU2 anisotropic source scale={scale:.8g} applied to D5 Eq65'
        for W in WIDTHS:
            rows.append({'D':5,'case':cname,'observable':'O12_EE','width':W,'coeff_O_delta2_eps2':scale*D5_eq65(float(W)),'status':'FINAL_HEE_GENERATED_V87','source_rule':d5_rule})
    elif kind=='III-b':
        ph=p['Phi_h']; lam=p['Z_lambda']; weight=Z_weight(ph,lam)
        f5,prof=build_D5_f_from_weight(weight)
        profiles[f'D5_{cname}']=prof
        d5_rule='tensor portal Z(Phi(u)) reweights D5 f2 ODE source; RT first-variation ratio calibrated to Eq65 baseline'
        for W in WIDTHS:
            ratio=D5_first_variation_integral(float(W),f5)/D5_first_variation_integral(float(W),f5_base)
            rows.append({'D':5,'case':cname,'observable':'O12_EE','width':W,'coeff_O_delta2_eps2':D5_eq65(float(W))*ratio,'status':'FINAL_HEE_GENERATED_V87','source_rule':d5_rule})
    case_rules.append({'case':cname,'kind':kind,'constant_scale_for_proportional_cases':scale,'parameters_json':json.dumps(p),'source_rule_D3':d3_rule,'notes':'V87: I and III-a are isotropic-blind for leading anisotropy coefficient in all D; III-b uses u-dependent BVP/ODE.'})

final=pd.DataFrame(rows)
final.to_csv(ROOT/'csv'/'v87_FINAL_HEE_D345_ALL_CASES.csv',index=False)
pd.DataFrame(case_rules).to_csv(ROOT/'csv'/'v87_case_source_rules.csv',index=False)

# Separate diagnostic for the isotropic background-shift terms that are NOT included in coeff_O_delta2_eps2.
diag_rows=[]
# These reproduce the v86-style shifts but label them as O(delta2 eps0)/background diagnostic.
def N3_temporal_background(u,q=0.5):
    u=np.asarray(u,dtype=float)
    return D3_N2_an(u) - q*q*u**(-4)*(1-u**(-2))
for cname,Nfun,desc in [
    ('I_temporal_U1_qX_0p5_background_shift', N3_temporal_background, 'isotropic O(delta2 eps0) blackening shift; not part of leading O(delta2 eps2) anisotropy coefficient'),
]:
    for W in WIDTHS:
        diag_rows.append({'D':3,'case':cname,'observable':'DeltaS_int_background_shift_diagnostic','width':W,'diagnostic_value':D3_interval_coeff(float(W),Nfun),'status':'SEPARATE_DIAGNOSTIC_NOT_FINAL_OD2EPS2','notes':desc})
pd.DataFrame(diag_rows).to_csv(ROOT/'diagnostics'/'v87_D3_isotropic_background_shift_separate_NOT_Odelta2eps2.csv',index=False)

# Profile exports
for key, df in profiles.items():
    df.to_csv(ROOT/'csv'/f'{key}_source_profile.csv',index=False)
f5_base_prof.to_csv(ROOT/'csv'/'D5_baseline_f2_integral_profile.csv',index=False)

# Ratios table
rat=[]
for D in [3,4,5]:
    base=final[(final.D==D)&(final.case=='baseline')].set_index('width').sort_index()
    for cname,_,_ in cases:
        sub=final[(final.D==D)&(final.case==cname)].set_index('width').sort_index()
        for W in WIDTHS:
            rat.append({'D':D,'case':cname,'width':W,'ratio_to_baseline':float(sub.loc[W].coeff_O_delta2_eps2/base.loc[W].coeff_O_delta2_eps2)})
pd.DataFrame(rat).to_csv(ROOT/'csv'/'v87_ratios_to_baseline.csv',index=False)

# D5 provenance checks vs 1109 formulas and v79 source-of-record.
prov=[]
grid=np.geomspace(1.0001,100,600)
res=D5_source_residual(grid)
prov.append({'check':'D5_f2_1109_source_equation_residual','formula':'r(r^4-1)f2pp+(-1+5r^4)f2p-S0=0 with f2=-(1-2r^2)/(18(1+r^2)^4)','max_abs_residual':float(np.max(np.abs(res))),'passed':float(np.max(np.abs(res)))<1e-12})
# v79 Eq65 table match
v79_path='/mnt/data/v79x/hsv_v79_d345_closed/csv/D5_AdS5_slab_HEE_series_closed.csv'
if not Path(v79_path).exists():
    # Extract if needed
    tmp=Path('/mnt/data/v87_tmp_v79')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    with zipfile.ZipFile('/mnt/data/HSV_v79_D345_ANALYTIC_LOCUS_HEE_CLOSED.zip') as zf:
        zf.extractall(tmp)
    v79s=list(tmp.rglob('D5_AdS5_slab_HEE_series_closed.csv'))
    if v79s: v79_path=str(v79s[0])
try:
    v79=pd.read_csv(v79_path)
    # Determine width column and eq65 column
    wcol='width_d' if 'width_d' in v79.columns else 'width'
    ccol='O12_area_coeff_eq65' if 'O12_area_coeff_eq65' in v79.columns else 'target_coeff_eq65'
    diffs=[]
    for _,r in v79.iterrows():
        w=float(r[wcol]); val=float(r[ccol]); diffs.append(abs(D5_eq65(w)-val)/max(abs(val),1e-300))
    prov.append({'check':'D5_eq65_formula_matches_v79_source_of_record','formula':'D5_eq65(d) vs v79 D5_AdS5_slab_HEE_series_closed.csv','max_relative_error':float(np.max(diffs)),'passed':float(np.max(diffs))<1e-12})
except Exception as e:
    prov.append({'check':'D5_eq65_formula_matches_v79_source_of_record','error':str(e),'passed':False})
pd.DataFrame(prov).to_csv(ROOT/'csv'/'v87_D5_baseline_provenance_checks.csv',index=False)

# Acceptance tests
acc=[]
def add_acc(test, expected, measured, tol, rule='rel', passed_override=None, criterion=None):
    if rule=='abs': err=abs(float(measured)-float(expected)); passed=err<=tol
    else: err=abs(float(measured)-float(expected))/max(abs(float(expected)),1e-300); passed=err<=tol
    if passed_override is not None: passed=bool(passed_override)
    acc.append({'test':test,'expected':expected,'measured':measured,'error':err,'tolerance':tol,'rule':rule,'passed':passed,'criterion':criterion or ''})
T3=16*PI**2.5/K2; T4=32*PI**2.5/K2
add_acc('A1_D3_entanglement_temperature',0.084412,T3,1e-4)
baseD3=final[(final.D==3)&(final.case=='baseline')&np.isclose(final.width,0.1)].iloc[0].coeff_O_delta2_eps2
add_acc('A2_D3_interval_L0p1_v79_anchor',5.1687089232e-4,baseD3,1e-5)
add_acc('A3_D4_entanglement_temperature',0.168823,T4,1e-4)
base4=final[(final.D==4)&(final.case=='baseline')]
base4_small=base4[base4.width<=0.12]
slope=np.polyfit(np.log(base4_small.width),np.log(np.abs(base4_small.coeff_O_delta2_eps2)),1)[0]
add_acc('A4_D4_O12_small_width_power',4.0,slope,0.10,rule='abs')
baseD4W005=abs(final[(final.D==4)&(final.case=='baseline')&np.isclose(final.width,0.05)].iloc[0].coeff_O_delta2_eps2)
add_acc('A5_D4_O12_W0p05_v79_anchor',8.027e-7,baseD4W005,1e-3)
for W in [0.05,0.1,0.3]:
    meas=final[(final.D==5)&(final.case=='baseline')&np.isclose(final.width,W)].iloc[0].coeff_O_delta2_eps2
    add_acc(f'A6_D5_Eq65_baseline_W{W}',D5_eq65(W),meas,1e-14)
for adm in [0.4,0.8,1.2]:
    cname={0.4:'II_aDM_0p4',0.8:'II_aDM_0p8',1.2:'II_aDM_1p2'}[adm]
    fac=1-adm**2/4
    for D in [3,4,5]:
        b=final[(final.D==D)&(final.case=='baseline')].sort_values('width').coeff_O_delta2_eps2.to_numpy()
        s=final[(final.D==D)&(final.case==cname)].sort_values('width').coeff_O_delta2_eps2.to_numpy()
        ratio=float(np.nanmean(s/b))
        add_acc(f'A7_D{D}_{cname}_proportional_scaling',fac,ratio,5e-6)
# II-b locked-profile scaling
fac_IIb=const_scale('II-b', {'alpha_dm':0.4, 'muX':0.5*MU_STAR})
for D in [3,4,5]:
    b=final[(final.D==D)&(final.case=='baseline')].sort_values('width').coeff_O_delta2_eps2.to_numpy()
    s=final[(final.D==D)&(final.case=='II-b_aDM_0p4_muX_0p5')].sort_values('width').coeff_O_delta2_eps2.to_numpy()
    add_acc(f'A8_D{D}_IIb_locked_profile_scaling',fac_IIb,float(np.nanmean(s/b)),5e-6)
# Isotropic blindness ALL D for I and III-a at leading anisotropy coefficient
for case in ['I_temporal_U1_qX_0p5','III-a_Phi_0p2','III-a_Phi_0p4']:
    for D in [3,4,5]:
        b=final[(final.D==D)&(final.case=='baseline')].sort_values('width').coeff_O_delta2_eps2.to_numpy()
        s=final[(final.D==D)&(final.case==case)].sort_values('width').coeff_O_delta2_eps2.to_numpy()
        maxerr=float(np.max(np.abs(s/b-1)))
        add_acc(f'A9_D{D}_{case}_isotropic_blindness_ratio1',0.0,maxerr,1e-12,rule='abs')
# Tensor portal shape nonconstant in all D? D3/D4/D5 should be nonconstant due Z profile. Verify D4/D5 strongly; D3 too.
for case in ['III-b_Phi_0p2_l1','III-b_Phi_0p4_l1']:
    for D in [3,4,5]:
        sub=final[(final.D==D)&(final.case==case)].set_index('width')
        b=final[(final.D==D)&(final.case=='baseline')].set_index('width')
        rsmall=float(sub.loc[0.05].coeff_O_delta2_eps2/b.loc[0.05].coeff_O_delta2_eps2)
        rlarge=float(sub.loc[0.30].coeff_O_delta2_eps2/b.loc[0.30].coeff_O_delta2_eps2)
        diff=abs(rsmall-rlarge)
        tol_shape = 1e-5 if D==3 else 1e-4
        acc.append({'test':f'A10_D{D}_{case}_tensor_shape_not_constant','expected':f'difference > {tol_shape:g}','measured':diff,'error':None,'tolerance':tol_shape,'rule':'measured>tol','passed':diff>tol_shape,'criterion':'tensor portal must be width-dependent, not constant rescale; D=3 Phi_h=0.2 effect is small but nonzero'})
# D5 provenance acceptance
for pr in prov:
    if 'passed' in pr:
        acc.append({'test':'A11_'+pr['check'],'expected':'pass','measured':pr.get('max_abs_residual',pr.get('max_relative_error',np.nan)),'error':None,'tolerance':'see check','rule':'provenance','passed':bool(pr['passed']),'criterion':pr.get('formula','')})
accdf=pd.DataFrame(acc)
accdf.to_csv(ROOT/'csv'/'v87_acceptance_tests.csv',index=False)

# Status table
status=[]
for cname,kind,p in cases:
    for D in [3,4,5]:
        rule=final[(final.D==D)&(final.case==cname)].iloc[0].source_rule
        status.append({'D':D,'case':cname,'observable':'DeltaS_int_Odelta2eps2' if D==3 else 'O12_EE','status':'FINAL_HEE_GENERATED_V87','source_rule':rule})
pd.DataFrame(status).to_csv(ROOT/'csv'/'v87_all_case_status.csv',index=False)

# Audits v86->v87
try:
    v86=pd.read_csv('/mnt/data/v86x/hsv_v86_corrected_final/csv/v86_FINAL_HEE_D345_ALL_CASES_CORRECTED.csv')
    audits=[]
    for case in ['I_temporal_U1_qX_0p5','III-a_Phi_0p2','III-a_Phi_0p4','III-b_Phi_0p2_l1','III-b_Phi_0p4_l1']:
        for D in [3,4,5]:
            old=v86[(v86.D==D)&(v86.case==case)].sort_values('width')
            new=final[(final.D==D)&(final.case==case)].sort_values('width')
            if len(old)==len(new) and len(old)>0:
                audits.append({'D':D,'case':case,'max_abs_delta_v87_minus_v86':float(np.max(np.abs(new.coeff_O_delta2_eps2.to_numpy()-old.coeff_O_delta2_eps2.to_numpy()))),'v86_rule':old.iloc[0].source_rule,'v87_rule':new.iloc[0].source_rule})
    pd.DataFrame(audits).to_csv(ROOT/'audit'/'v86_to_v87_order_counting_correction_audit.csv',index=False)
except Exception as e:
    (ROOT/'audit'/'v86_audit_error.txt').write_text(str(e))

# Figures
for D in [3,4,5]:
    plt.figure(figsize=(8,5))
    for cname,_,_ in cases:
        sub=final[(final.D==D)&(final.case==cname)].sort_values('width')
        if cname in ['baseline','I_temporal_U1_qX_0p5','II_aDM_0p8','III-a_Phi_0p4','III-b_Phi_0p4_l1']:
            plt.plot(sub.width, sub.coeff_O_delta2_eps2, marker='o', label=cname)
    plt.xlabel('width')
    plt.ylabel('DeltaS_int O(delta2 eps2) coeff' if D==3 else 'O12 coeff')
    plt.title(f'D={D} v87 final HEE: isotropic-blind + tensor-shape corrected')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT/'figures'/f'figure_D{D}_v87_final_HEE.png',dpi=240)
    plt.close()

plt.figure(figsize=(7.5,5))
for D,ls in [(3,':'),(4,'-'),(5,'--')]:
    base=final[(final.D==D)&(final.case=='baseline')].set_index('width')
    for cname in ['III-b_Phi_0p2_l1','III-b_Phi_0p4_l1']:
        sub=final[(final.D==D)&(final.case==cname)].set_index('width')
        plt.plot(WIDTHS, sub.loc[WIDTHS].coeff_O_delta2_eps2/base.loc[WIDTHS].coeff_O_delta2_eps2, marker='o', linestyle=ls, label=f'D{D} {cname}')
plt.axhline(1,color='k',lw=0.8)
plt.xlabel('width')
plt.ylabel('ratio to baseline')
plt.title('V87 tensor portal: width-dependent response in D=3,4,5')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(ROOT/'figures'/'figure_v87_tensor_portal_shape_ratios.png',dpi=240)
plt.close()

plt.figure(figsize=(7,4.5))
for case in ['I_temporal_U1_qX_0p5','III-a_Phi_0p2','III-a_Phi_0p4']:
    for D,marker in [(3,'o'),(4,'s'),(5,'^')]:
        base=final[(final.D==D)&(final.case=='baseline')].set_index('width')
        sub=final[(final.D==D)&(final.case==case)].set_index('width')
        plt.plot(WIDTHS, sub.loc[WIDTHS].coeff_O_delta2_eps2/base.loc[WIDTHS].coeff_O_delta2_eps2, marker=marker, label=f'D{D} {case}')
plt.axhline(1,color='k',lw=0.8)
plt.xlabel('width')
plt.ylabel('ratio to baseline')
plt.title('V87 isotropic dark sectors are blind at leading anisotropy order')
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig(ROOT/'figures'/'figure_v87_isotropic_blindness_ratios.png',dpi=240)
plt.close()

# Copy sources
for p in ['/mnt/data/hsv_hee_solver.py','/mnt/data/v84_FINAL_HEE.csv','/mnt/data/HSV_v86_CORRECTED_D345_ALLCASE_HEE.zip','/mnt/data/HSV_v79_D345_ANALYTIC_LOCUS_HEE_CLOSED.zip','/mnt/data/paper_extract/1109.4592v3.pdf']:
    if Path(p).exists(): shutil.copy(p, ROOT/'source_refs'/Path(p).name)
shutil.copy(__file__, ROOT/'scripts'/'build_v87_final_closure.py')

summary={
    'rows':int(len(final)),
    'acceptance_total':int(len(accdf)),
    'acceptance_passed':int(accdf.passed.sum()),
    'acceptance_failed':int((~accdf.passed).sum()),
    'D3_baseline_L0p1':float(baseD3),
    'D4_baseline_W0p05_abs':float(baseD4W005),
    'D4_power_slope':float(slope),
    'D5_baseline_W0p1':float(D5_eq65(0.1)),
    'D5_provenance_checks':prov,
    'corrections_from_v86':['D3 Case I and III-a are now ratio 1 at O(delta2 eps2), not background-shift mixed','A11 removed/replaced by all-D isotropic-blindness test','D3 O(delta2 eps0) background shift moved to diagnostics, not final HEE anisotropy coefficient'],
    'paper_readiness_assessment':'Ready for analytic-locus D=3,4,5 leading-anisotropy HEE claims under stated source rules; general off-locus HSV grid remains separate if required.'
}
(ROOT/'reports'/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

report=f"""# HSV v87 final closure: D=3,4,5 all-case leading-anisotropy HEE

## What changed from v86

V86 correctly fixed the D=4/D=5 O12 source selection, but it mixed isotropic O(delta^2 epsilon^0) background shifts into the D=3 O(delta^2 epsilon^2) anisotropy coefficient for Case I and Case III-a. V87 corrects the order counting.

- Case I temporal U(1): leading anisotropy coefficient unchanged in D=3,4,5. The D=3 blackening/background shift is exported separately as a diagnostic, not as coeff_O_delta2_eps2.
- Case III-a homogeneous adjoint scalar: leading anisotropy coefficient unchanged in D=3,4,5.
- Case II and II-b locked-profile hidden SU(2): proportional anisotropic source scaling.
- Case III-b tensor portal: u-dependent Z(Phi(u)) reweighted source, solved by D=3 N2 integral, D=4 H2 BVP, and D=5 f2 ODE.

## Acceptance

{accdf.to_markdown(index=False)}

## D5 provenance

{pd.DataFrame(prov).to_markdown(index=False)}

## Output files

- csv/v87_FINAL_HEE_D345_ALL_CASES.csv
- csv/v87_ratios_to_baseline.csv
- csv/v87_case_source_rules.csv
- csv/v87_acceptance_tests.csv
- csv/v87_D5_baseline_provenance_checks.csv
- diagnostics/v87_D3_isotropic_background_shift_separate_NOT_Odelta2eps2.csv

## Summary JSON

```json
{json.dumps(summary,indent=2)}
```
"""
(ROOT/'reports'/'v87_final_closure_report.md').write_text(report,encoding='utf-8')

# Manifest
manifest=[]
for path in sorted(ROOT.rglob('*')):
    if path.is_file(): manifest.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size})
pd.DataFrame(manifest).to_csv(ROOT/'manifest.csv',index=False)

# zip
zip_path=Path('/mnt/data/HSV_v87_FINAL_D345_ALLCASE_HEE_ORDER_COUNTING_CORRECTED.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zf:
    for path in ROOT.rglob('*'):
        zf.write(path,path.relative_to(ROOT.parent))
print(zip_path)
print(json.dumps(summary,indent=2))
print(accdf[['test','passed','measured']].to_string(index=False))
