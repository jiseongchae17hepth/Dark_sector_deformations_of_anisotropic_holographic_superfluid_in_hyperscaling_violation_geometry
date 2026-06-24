#!/usr/bin/env python3
from __future__ import annotations
import sys, math, json, time, hashlib, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import mpmath as mp
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0,'/mnt/data')
import hsv_v97_core as core

mp.mp.dps=45
ROOT=Path('/mnt/data/hsv_v97_full_d345')
CSV=ROOT/'csv'/'v97_FULL_D345_off_locus_HEE_all_cases.csv'
CASES={
 'baseline':('STRICT_VISIBLE_EOM',1.0,None),
 'I_temporal_U1_qX_0p5':('STRICT_LEADING_ORDER_ISOTROPIC_NULL',1.0,None),
 'II_aDM_0p4':('STRICT_MINIMAL_HIDDEN_SU2_MUX0',1-0.4**2/4,None),
 'II_aDM_0p8':('STRICT_MINIMAL_HIDDEN_SU2_MUX0',1-0.8**2/4,None),
 'II_aDM_1p2':('STRICT_MINIMAL_HIDDEN_SU2_MUX0',1-1.2**2/4,None),
 'II-b_aDM_0p4_muX_0p5':('LOCKED_PROFILE_MUX_PRESCRIPTION',1-0.4**2/4+(0.5**2)*0.4**2/4,None),
 'III-a_Phi_0p2':('STRICT_LEADING_ORDER_ISOTROPIC_NULL',1.0,None),
 'III-a_Phi_0p4':('STRICT_LEADING_ORDER_ISOTROPIC_NULL',1.0,None),
 'III-b_Phi_0p2_l1':('Z_WEIGHTED_RESPONSE_PRESCRIPTION',None,0.2),
 'III-b_Phi_0p4_l1':('Z_WEIGHTED_RESPONSE_PRESCRIPTION',None,0.4),
}

def zweight(phi_h):
    def wt(u):
        q=np.asarray(u,float); phi=phi_h/(1+q*q)**2
        return np.exp(phi*phi)
    return wt

# Exact unperturbed finite-temperature width, smooth after x=1-t^2.
def width_exact_fast(ut,d,a,z):
    q=d*(a+1.0); p=2*q; h=core.nh(d,a,z)
    def f(t):
        if t==0.0:
            return math.sqrt(2.0/q)/math.sqrt(max(1.0-ut**(-h),1e-300))
        x=1.0-t*t
        if x<=0.0: return 0.0
        om1=-math.expm1(h*(math.log(x)-math.log(ut)))
        om2=-math.expm1(p*math.log(x))
        return 2*t*x**q/math.sqrt(max(om1*om2,1e-300))
    val,_=quad(f,0,1,epsabs=2e-12,epsrel=2e-12,limit=300)
    return 2*val/ut

def ut_exact_fast(W,d,a,z):
    lo=1+1e-10; hi=2.0
    while width_exact_fast(hi,d,a,z)>W: hi*=2
    return brentq(lambda u:width_exact_fast(u,d,a,z)-W,lo,hi,xtol=2e-13,rtol=2e-13,maxiter=200)

# Fixed-boundary first variation of RT/geodesic functional.
def hee_O12_fixed(W,d,a,z,Afun):
    ut=mp.mpf(str(ut_exact_fast(W,d,a,z)))
    p=mp.mpf(str(2*d*(a+1.0))); h=mp.mpf(str(core.nh(d,a,z)))
    power=mp.mpf(str(2*a if d==2 else 3*a+1))
    factor=mp.mpf(4 if d==2 else 6)
    def ft(t):
        if t<=mp.mpf('1e-35') or t>=1-mp.mpf('1e-35'): return mp.mpf('0')
        x=1-t*t; u=ut/x
        uf=float(u) if u<mp.mpf('1e300') else float('inf')
        av=float(Afun(np.array([uf]))[0])
        A=mp.mpf(str(av)) if math.isfinite(av) else mp.mpf('0')
        N=-mp.expm1(h*(mp.log(x)-mp.log(ut)))
        shape=mp.sqrt(-mp.expm1(p*mp.log(x)))
        val=factor*A*u**power*shape/mp.sqrt(N)*ut/x**2
        return 2*t*val
    return mp.quad(ft,[0,1]),ut

def solve_point(D,a,z):
    d=D-2
    zs=core.solve_zero_mode_xi(d,a,z,tol=1e-6)
    if zs is None or not zs.success: raise RuntimeError('zero mode failed')
    mu,w,wp,b0=core.make_zero_profile_xi(zs,d,a,z)
    A,aqc=core.solve_A_response(d,a,z,mu,w,wp,U=400,n=12000)
    portals={}
    for ph in [0.2,0.4]:
        portals[ph]=core.solve_A_response(d,a,z,mu,w,wp,weight=zweight(ph),U=400,n=12000)[0]
    return mu,A,portals

def main():
    df=pd.read_csv(CSV)
    if 'coeff_paper_small_width' not in df.columns:
        df['coeff_paper_small_width']=df['coeff_O_delta2_epsilon2']
    if 'coeff_finite_width_first_variation' not in df.columns:
        df['coeff_finite_width_first_variation']=np.nan
    if 'coefficient_method' not in df.columns:
        df['coefficient_method']=np.where(df['coeff_O_delta2_epsilon2'].notna(),'PAPER_SMALL_WIDTH_FIRST_VARIATION','UNAVAILABLE')
    bad=df[df['coeff_O_delta2_epsilon2'].isna() & ~((df.D==3)&(df.alpha==0)&(df.z==1))]
    groups=bad[['D','alpha','z']].drop_duplicates().sort_values(['D','alpha','z'])
    audit=[]; t0=time.time()
    for _,g in groups.iterrows():
        D=int(g.D);a=float(g.alpha);z=float(g.z);d=D-2
        print(f'FILL D={D} a={a} z={z}',flush=True)
        mu,A,portals=solve_point(D,a,z)
        widths=sorted(df[(df.D==D)&(df.alpha==a)&(df.z==z)&df.coeff_O_delta2_epsilon2.isna()].width.unique())
        for W in widths:
            vb,ut=hee_O12_fixed(float(W),d,a,z,A); base=float(mp.re(vb)); ut=float(ut)
            pv={}
            for ph in [0.2,0.4]:
                vv,_=hee_O12_fixed(float(W),d,a,z,portals[ph]);pv[ph]=float(mp.re(vv))
            mask=(df.D==D)&(df.alpha==a)&(df.z==z)&(df.width==W)
            for case,(rule,scale,ph) in CASES.items():
                m=mask&(df.case==case)
                coeff=base*scale if ph is None else pv[ph]
                df.loc[m,'coeff_finite_width_first_variation']=coeff
                df.loc[m,'coeff_O_delta2_epsilon2']=coeff
                df.loc[m,'visible_baseline_coeff']=base
                df.loc[m,'ratio_to_visible_baseline']=coeff/base if base!=0 else np.nan
                df.loc[m,'turning_point_u_star']=ut
                df.loc[m,'width_regime']='FINITE_WIDTH_EXACT_EXTENSION'
                df.loc[m,'coefficient_method']='FINITE_WIDTH_FIXED_BOUNDARY_FIRST_VARIATION'
                old=df.loc[m,'notes'].fillna('').astype(str)
                df.loc[m,'notes']=old+' Exact unperturbed turning point and fixed-boundary RT first variation used because the paper small-width turning point lies at or inside the horizon.'
            audit.append(dict(D=D,alpha=a,z=z,width=W,mu_c=mu,turning_point_exact=ut,
                              baseline_finite_width=base,tensor_0p2=pv[0.2],tensor_0p4=pv[0.4],
                              paper_small_width_available=False))
        df.to_csv(CSV,index=False)
        pd.DataFrame(audit).to_csv(ROOT/'audit'/'v97_finite_width_extension_audit.csv',index=False)
    # enrich reliability fields after fill
    for c in ['abs_coeff','log10_abs_coeff','baseline_abs','log10_abs_baseline']:
        if c not in df.columns: df[c]=np.nan
    df['abs_coeff']=df['coeff_O_delta2_epsilon2'].abs()
    df['log10_abs_coeff']=np.log10(df['abs_coeff'].where(df['abs_coeff']>0))
    df['baseline_abs']=df['visible_baseline_coeff'].abs()
    df['log10_abs_baseline']=np.log10(df['baseline_abs'].where(df['baseline_abs']>0))
    df['ratio_reliability']=np.select([
        df['baseline_abs'].isna() | (df['baseline_abs']<1e-25),
        df['baseline_abs']<1e-14,
        df['baseline_abs']<1e-10],
        ['NEAR_ZERO_BASELINE_RATIO_SENSITIVE','SMALL_BASELINE_SHOW_ABSOLUTE_TOO','MODERATE_BASELINE_CHECK_ABSOLUTE'],
        default='DIRECT_RATIO_STABLE')
    df.to_csv(CSV,index=False)
    # method-specific complete tables
    paper=df[df['coeff_paper_small_width'].notna()].copy()
    paper.to_csv(ROOT/'csv'/'v97_paper_small_width_HEE_grid.csv',index=False)
    full=df[df['coeff_O_delta2_epsilon2'].notna()].copy()
    full.to_csv(ROOT/'csv'/'v97_complete_canonical_HEE_grid.csv',index=False)
    # update summary/report
    sp=ROOT/'reports'/'summary.json'; summary=json.loads(sp.read_text())
    summary.update(valid_rows=int(df.coeff_O_delta2_epsilon2.notna().sum()),
                   degenerate_rows=int(df.coeff_O_delta2_epsilon2.isna().sum()),
                   paper_small_width_rows=int(df.coeff_paper_small_width.notna().sum()),
                   finite_width_extension_rows=int((df.coefficient_method=='FINITE_WIDTH_FIXED_BOUNDARY_FIRST_VARIATION').sum()),
                   coefficient_methods=df.coefficient_method.value_counts(dropna=False).to_dict(),
                   finite_width_patch_runtime_seconds=time.time()-t0)
    sp.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=ROOT/'reports'/'v97_full_report.md'
    txt=report.read_text(encoding='utf-8')
    txt+='''\n\n## Finite-width completion\nThe original-paper small-width first-variation column is retained unchanged wherever its pure-background turning point satisfies u_* > 1. For 28 large-alpha/large-width parameter-width combinations where the asymptotic estimate placed u_* at or inside the horizon, the canonical full-grid column is completed using the exact unperturbed black-brane turning point and the fixed-boundary first variation of the RT functional. The two methods are stored in separate columns and `coefficient_method` records which value populates the canonical column. No locus-derived multiplicative correction factor is used.\n'''
    report.write_text(txt,encoding='utf-8')
    # refresh manifest + zip
    manifests=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file():
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            manifests.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=h))
    pd.DataFrame(manifests).to_csv(ROOT/'manifest.csv',index=False)
    zpath=Path('/mnt/data/HSV_v97_FULL_D345_OFF_LOCUS_EOM_RT_CLOSED.zip')
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in ROOT.rglob('*'):
            if p.is_file(): zf.write(p,p.relative_to(ROOT.parent))
    print('DONE valid',df.coeff_O_delta2_epsilon2.notna().sum(),'nan',df.coeff_O_delta2_epsilon2.isna().sum(),'zip',zpath,flush=True)

if __name__=='__main__': main()
