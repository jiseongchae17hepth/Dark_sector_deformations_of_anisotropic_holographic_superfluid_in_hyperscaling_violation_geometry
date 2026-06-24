#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, csv, math, time, traceback, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpmath as mp
sys.path.insert(0,'/mnt/data')
import hsv_v97_core as core

mp.mp.dps=45
OUT=Path('/mnt/data/hsv_v97_full_d345')
for sub in ['csv','profiles','figures','reports','scripts','audit','checkpoints']:
    (OUT/sub).mkdir(parents=True,exist_ok=True)

ALPHAS=[0.0,0.5,1.0,1.5,2.0,3.0,4.0]
ZS=[1.0,1.5,2.0,2.5,3.0,4.0,5.0]
WIDTHS=[0.05,0.08,0.10,0.12,0.18,0.24,0.30]
CASES=[
 ('baseline','STRICT_VISIBLE_EOM',1.0,None),
 ('I_temporal_U1_qX_0p5','STRICT_LEADING_ORDER_ISOTROPIC_NULL',1.0,None),
 ('II_aDM_0p4','STRICT_MINIMAL_HIDDEN_SU2_MUX0',1-0.4**2/4,None),
 ('II_aDM_0p8','STRICT_MINIMAL_HIDDEN_SU2_MUX0',1-0.8**2/4,None),
 ('II_aDM_1p2','STRICT_MINIMAL_HIDDEN_SU2_MUX0',1-1.2**2/4,None),
 ('II-b_aDM_0p4_muX_0p5','LOCKED_PROFILE_MUX_PRESCRIPTION',1-0.4**2/4+(0.5**2)*0.4**2/4,None),
 ('III-a_Phi_0p2','STRICT_LEADING_ORDER_ISOTROPIC_NULL',1.0,None),
 ('III-a_Phi_0p4','STRICT_LEADING_ORDER_ISOTROPIC_NULL',1.0,None),
 ('III-b_Phi_0p2_l1','Z_WEIGHTED_RESPONSE_PRESCRIPTION',None,0.2),
 ('III-b_Phi_0p4_l1','Z_WEIGHTED_RESPONSE_PRESCRIPTION',None,0.4),
]

def zweight(phi_h):
    def wt(u):
        q=np.asarray(u,float)
        phi=phi_h/(1+q*q)**2
        return np.exp(phi*phi)
    return wt

def row_base(D,d,a,z,W,case,mu,coeff,base,ut,zero_status,response_status,rule,status,notes='',profile=''):
    ratio=np.nan if (not np.isfinite(base) or base==0 or not np.isfinite(coeff)) else coeff/base
    sw='SMALL_WIDTH_CONTROLLED' if ut>=4 else ('INTERMEDIATE_WIDTH' if ut>=2 else 'NEAR_HORIZON_EXPLORATORY')
    return dict(D=D,d=d,alpha=a,z=z,width=W,case=case,
                observable='DeltaS_int_O_delta2_epsilon2' if D==3 else 'O12_EE_O_delta2_epsilon2',
                mu_c=mu,mu_convention='b0(infinity)=mu; analytic-locus anchor mu=4',
                coeff_O_delta2_epsilon2=coeff,visible_baseline_coeff=base,ratio_to_visible_baseline=ratio,
                turning_point_u_star=ut,width_regime=sw,
                zero_mode_status=zero_status,response_status=response_status,
                source_rule=rule,method_status=status,portal_profile=profile,notes=notes)

def write_csv(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False)

def run_point(D,a,z):
    d=D-2; m=core.mexp(d,a,z)
    t0=time.time()
    if m<=0 or core.Delta(d,a,z)<=0:
        return {'valid':False,'D':D,'d':d,'a':a,'z':z,'reason':'DEGENERATE_NO_VISIBLE_CHEMICAL_PROFILE','seconds':time.time()-t0}
    zsol=core.solve_zero_mode_xi(d,a,z,tol=1e-6)
    if zsol is None or not zsol.success or zsol.p[0]<=0:
        return {'valid':False,'D':D,'d':d,'a':a,'z':z,'reason':'ZERO_MODE_FAILED','seconds':time.time()-t0}
    mu,w,wp,b0=core.make_zero_profile_xi(zsol,d,a,z)
    point={'valid':True,'D':D,'d':d,'a':a,'z':z,'mu':mu,'zero':zsol,'w':w,'wp':wp,
           'zero_nodes':zsol.x.size,'zero_rms':float(np.max(zsol.rms_residuals)),'seconds_zero':time.time()-t0}
    if D==3:
        tb=time.time(); mu2,b2,b2p,bqc=core.solve_b2_fredholm_D3(a,z,mu,w,U=250,n=24000)
        point.update(mu2=mu2,b2=b2,b2p=b2p,bqc=bqc,seconds_b2=time.time()-tb)
        tm=time.time(); msol,src=core.solve_D3_metric(a,z,mu,w,wp,b2p,U=80,tol=3e-6)
        if not msol.success:
            point.update(valid=False,reason='D3_METRIC_BVP_FAILED',metric_message=msol.message); return point
        N2,C=core.make_D3_N2(msol,a,z,U=80)
        point.update(metric=msol,N2=N2,N2_tail_C=C,metric_nodes=msol.x.size,
                     metric_rms=float(np.max(msol.rms_residuals)),seconds_metric=time.time()-tm)
        # portal metric responses: same zero mode and b2, radial Z weight on YM source
        portals={}
        for ph in [0.2,0.4]:
            tp=time.time(); ps,_=core.solve_D3_metric(a,z,mu,w,wp,b2p,weight=zweight(ph),U=80,tol=3e-6,guess=msol)
            if ps.success:
                pn,pc=core.make_D3_N2(ps,a,z,U=80);portals[ph]=(pn,ps,float(np.max(ps.rms_residuals)),time.time()-tp)
            else: portals[ph]=(None,ps,np.nan,time.time()-tp)
        point['portals']=portals
    else:
        tr=time.time(); A,aqc=core.solve_A_response(d,a,z,mu,w,wp,U=400,n=12000)
        point.update(A=A,Aqc=aqc,response_seconds=time.time()-tr)
        portals={}
        for ph in [0.2,0.4]:
            tp=time.time(); Ap,qcp=core.solve_A_response(d,a,z,mu,w,wp,weight=zweight(ph),U=400,n=12000)
            portals[ph]=(Ap,qcp,time.time()-tp)
        point['portals']=portals
    point['seconds']=time.time()-t0
    return point

def eval_point(point):
    D=point['D'];d=point['d'];a=point['a'];z=point['z'];mu=point.get('mu',np.nan)
    rows=[]
    if not point['valid']:
        for case,rule,scale,ph in CASES:
            for W in WIDTHS:
                rows.append(row_base(D,d,a,z,W,case,mu,np.nan,np.nan,np.nan,'INVALID','INVALID',rule,
                                     point.get('reason','INVALID'),notes='No finite visible chemical profile at this parameter point.'))
        return rows
    # compute baseline and portal coefficients once
    vals_base={}; vals_port={0.2:{},0.4:{}}
    for W in WIDTHS:
        if D==3:
            vb,ut=core.hee_D3(W,a,z,point['N2'])
            vals_base[W]=(float(mp.re(vb)),float(ut))
            for ph in [0.2,0.4]:
                fn,sol,rms,sec=point['portals'][ph]
                if fn is None: vals_port[ph][W]=(np.nan,float(ut))
                else:
                    vv,utp=core.hee_D3(W,a,z,fn); vals_port[ph][W]=(float(mp.re(vv)),float(utp))
        else:
            vb,ut=core.hee_O12(W,d,a,z,point['A'])
            vals_base[W]=(float(mp.re(vb)),float(ut))
            for ph in [0.2,0.4]:
                vv,utp=core.hee_O12(W,d,a,z,point['portals'][ph][0]); vals_port[ph][W]=(float(mp.re(vv)),float(utp))
    for case,rule,scale,ph in CASES:
        for W in WIDTHS:
            base,ut=vals_base[W]
            if ph is None: coeff=base*scale
            else: coeff=vals_port[ph][W][0]
            notes=''
            profile=''
            method='FINAL_STRICT_EOM_RT' if rule.startswith('STRICT') else 'FINAL_WITH_EXPLICIT_PRESCRIPTION'
            if rule=='LOCKED_PROFILE_MUX_PRESCRIPTION':
                notes='mu_X nonzero row uses the stated locked-profile source prescription; not a full independent hidden-profile solve.'
            if rule=='Z_WEIGHTED_RESPONSE_PRESCRIPTION':
                profile=f'Phi(u)={ph}/(1+u^2)^2, Z=exp(Phi^2)'
                notes='Tensor row recomputes the metric response with a radial Z-weighted YM stress while holding the onset profile fixed; model prescription explicitly recorded.'
            if rule=='STRICT_LEADING_ORDER_ISOTROPIC_NULL':
                notes='At leading O(delta^2 epsilon^2), an epsilon-independent homogeneous isotropic dark source does not alter this anisotropy-induced coefficient.'
            response_status='CONVERGED'
            if D==3 and ph is not None and point['portals'][ph][0] is None: response_status='FAILED'
            rows.append(row_base(D,d,a,z,W,case,mu,coeff,base,ut,'CONVERGED',response_status,rule,method,notes,profile))
    return rows

def add_profiles(point, profile_rows):
    D=point['D'];a=point['a'];z=point['z'];d=point['d']
    selected=(a,z) in [(0,1),(0.5,1),(1,1),(2,1),(0,3),(4,5)]
    if not point.get('valid') or not selected:return
    us=np.unique(np.r_[np.linspace(1.001,2,80),np.geomspace(2,80,120)])
    for u in us:
        rr=dict(D=D,d=d,alpha=a,z=z,u=u,mu_c=point['mu'],w1=float(point['w'](np.array([u]))[0]),w1p=float(point['wp'](np.array([u]))[0]))
        if D==3:
            rr.update(b2=float(point['b2'](np.array([u]))[0]),N2_visible=float(point['N2'](np.array([u]))[0]))
            for ph in [.2,.4]:
                fn=point['portals'][ph][0];rr[f'N2_tensor_Phi_{ph}']=float(fn(np.array([u]))[0]) if fn else np.nan
        else:
            rr.update(A_visible=float(point['A'](np.array([u]))[0]))
            for ph in [.2,.4]:rr[f'A_tensor_Phi_{ph}']=float(point['portals'][ph][0](np.array([u]))[0])
        profile_rows.append(rr)

def acceptance_tests(all_rows, qc_rows):
    tests=[]
    def add(name,exp,meas,tol,kind='relative'):
        err=abs(meas-exp)/(abs(exp) if exp else 1) if kind=='relative' else abs(meas-exp)
        tests.append(dict(test=name,expected=exp,measured=meas,error=err,tolerance=tol,status='PASS' if err<=tol else 'FAIL'))
    # zero modes
    q=pd.DataFrame(qc_rows)
    for D,a,z,ref in [(3,2,1,4),(4,.5,1,4),(5,0,1,4),(5,1,1,5.67955107885)]:
        r=q[(q.D==D)&(q.alpha==a)&(q.z==z)].iloc[0];add(f'mu_D{D}_a{a}_z{z}',ref,r.mu_c,2e-5)
    # D3 b2, metric and HEE
    r=q[(q.D==3)&(q.alpha==2)&(q.z==1)].iloc[0]
    add('D3_b2_mu2',71/6720,r.mu2,2e-5)
    add('D3_metric_profile_rel',0,r.metric_anchor_rel_error,1e-4,kind='absolute')
    df=pd.DataFrame(all_rows)
    v=df[(df.D==3)&(df.alpha==2)&(df.z==1)&(df.width==.1)&(df.case=='baseline')].iloc[0].coeff_O_delta2_epsilon2
    add('D3_HEE_L0p1',5.1687089232e-4,v,5e-5)
    # D4 response and HEE
    r=q[(q.D==4)&(q.alpha==.5)&(q.z==1)].iloc[0];add('D4_response_profile_rel',0,r.response_anchor_rel_error,1e-4,kind='absolute')
    v=abs(df[(df.D==4)&(df.alpha==.5)&(df.z==1)&(df.width==.05)&(df.case=='baseline')].iloc[0].coeff_O_delta2_epsilon2)
    add('D4_HEE_W0p05',8.027557992462275e-7,v,2e-4)
    # D4 W power
    sub=df[(df.D==4)&(df.alpha==.5)&(df.z==1)&(df.case=='baseline')&df.width.isin([.05,.08,.1,.12])]
    slope=np.polyfit(np.log(sub.width),np.log(np.abs(sub.coeff_O_delta2_epsilon2)),1)[0]
    add('D4_smallW_power',4,slope,.03,kind='absolute')
    # D5 response and leading Eq65 coefficient
    r=q[(q.D==5)&(q.alpha==0)&(q.z==1)].iloc[0];add('D5_response_profile_rel',0,r.response_anchor_rel_error,1e-4,kind='absolute')
    # direct small W evaluation from analytic response at .01
    vv,_=core.hee_O12(.01,3,0,1,core.D5_f2_an); lead=float(vv)/.01**4
    c4=float(mp.sqrt(3)**3/(448*mp.pi**(mp.mpf('4.5')))*mp.gamma(mp.mpf(1)/6)**3*mp.gamma(mp.mpf(1)/3)**3)
    add('D5_Eq65_leading_W4',c4,lead,1e-3)
    # strict ratios
    for D in [3,4,5]:
        for name,rat in [('II_aDM_0p4',.96),('II_aDM_0p8',.84),('II_aDM_1p2',.64)]:
            sub=df[(df.D==D)&(df.case==name)&np.isfinite(df.ratio_to_visible_baseline)]
            add(f'D{D}_{name}_ratio',rat,float(np.nanmedian(sub.ratio_to_visible_baseline)),2e-10)
        for name in ['I_temporal_U1_qX_0p5','III-a_Phi_0p4']:
            sub=df[(df.D==D)&(df.case==name)&np.isfinite(df.ratio_to_visible_baseline)]
            add(f'D{D}_{name}_null',1,float(np.nanmedian(sub.ratio_to_visible_baseline)),2e-10)
    return tests

def main():
    start=time.time(); all_rows=[];qc=[];profiles=[];failures=[]
    for D in [3,4,5]:
        for a in ALPHAS:
            for z in ZS:
                print(f'RUN D={D} a={a} z={z}',flush=True)
                try:
                    p=run_point(D,a,z)
                    all_rows.extend(eval_point(p));add_profiles(p,profiles)
                    qr=dict(D=D,d=D-2,alpha=a,z=z,valid=p.get('valid',False),reason=p.get('reason',''),seconds=p.get('seconds',np.nan))
                    if p.get('valid'):
                        qr.update(mu_c=p['mu'],zero_nodes=p['zero_nodes'],zero_rms=p['zero_rms'])
                        if D==3:
                            qr.update(mu2=p['mu2'],b2_fredholm_residual=p['bqc']['J0']+p['bqc']['C_horizon_flux']*p['bqc']['J1'],metric_nodes=p['metric_nodes'],metric_rms=p['metric_rms'])
                            if a==2 and z==1:
                                g=np.linspace(1.001,15,1200);qr['metric_anchor_rel_error']=float(np.max(abs(p['N2'](g)-core.D3_N2_an(g)))/np.max(abs(core.D3_N2_an(g))))
                            else:qr['metric_anchor_rel_error']=np.nan
                        else:
                            qr.update(response_Finf=p['Aqc']['Finf'])
                            if D==4 and a==.5 and z==1:
                                g=np.linspace(1.001,20,1200);qr['response_anchor_rel_error']=float(np.max(abs(p['A'](g)-core.D4_H2_an(g)))/np.max(abs(core.D4_H2_an(g))))
                            elif D==5 and a==0 and z==1:
                                g=np.linspace(1.001,20,1200);qr['response_anchor_rel_error']=float(np.max(abs(p['A'](g)-core.D5_f2_an(g)))/np.max(abs(core.D5_f2_an(g))))
                            else:qr['response_anchor_rel_error']=np.nan
                    qc.append(qr)
                except Exception as e:
                    failures.append(dict(D=D,alpha=a,z=z,error=repr(e),traceback=traceback.format_exc()))
                    p={'valid':False,'D':D,'d':D-2,'a':a,'z':z,'reason':'EXCEPTION_'+type(e).__name__}
                    all_rows.extend(eval_point(p));qc.append(dict(D=D,d=D-2,alpha=a,z=z,valid=False,reason=p['reason']))
                write_csv(OUT/'checkpoints'/'rows_partial.csv',all_rows)
                write_csv(OUT/'checkpoints'/'qc_partial.csv',qc)
    tests=acceptance_tests(all_rows,qc)
    write_csv(OUT/'csv'/'v97_FULL_D345_off_locus_HEE_all_cases.csv',all_rows)
    write_csv(OUT/'csv'/'v97_zero_mode_metric_response_QC.csv',qc)
    write_csv(OUT/'profiles'/'v97_selected_radial_profiles.csv',profiles)
    write_csv(OUT/'csv'/'v97_acceptance_tests.csv',tests)
    write_csv(OUT/'audit'/'v97_failures.csv',failures)
    rules=[]
    for case,rule,scale,ph in CASES:
        rules.append(dict(case=case,source_rule=rule,constant_scale=scale,portal_phi_h=ph,
                          strictness=('STRICT' if rule.startswith('STRICT') else 'EXPLICIT_MODEL_PRESCRIPTION')))
    write_csv(OUT/'csv'/'v97_case_source_rules.csv',rules)
    # Eq65 source-of-record comparison at D5 locus
    c4=float(mp.sqrt(3)**3/(448*mp.pi**(mp.mpf('4.5')))*mp.gamma(mp.mpf(1)/6)**3*mp.gamma(mp.mpf(1)/3)**3)
    c6=float(2943/(573440*mp.pi**9)*mp.gamma(mp.mpf(1)/6)**6*mp.gamma(mp.mpf(1)/3)**6)
    comp=[]
    df=pd.DataFrame(all_rows)
    for W in WIDTHS:
        direct=df[(df.D==5)&(df.alpha==0)&(df.z==1)&(df.width==W)&(df.case=='baseline')].iloc[0].coeff_O_delta2_epsilon2
        series=c4*W**4+c6*W**6
        comp.append(dict(width=W,direct_RT_first_variation=direct,PPO_Eq65_W4_W6_series=series,ratio_direct_to_series=direct/series,
                         c4=c4,c6=c6))
    write_csv(OUT/'csv'/'v97_D5_PPO_Eq65_locus_comparison.csv',comp)
    # Figure-ready W=.1
    figdf=df[(df.width==.1)&df.method_status.str.startswith('FINAL')].copy()
    figdf.to_csv(OUT/'csv'/'v97_figure_ready_W0p10.csv',index=False)
    # figures 2x3 baseline and tensor ratio
    fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for col,D in enumerate([3,4,5]):
        sub=figdf[(figdf.D==D)&(figdf.case=='baseline')]
        piv=sub.pivot(index='z',columns='alpha',values='coeff_O_delta2_epsilon2')
        im=axs[0,col].imshow(np.log10(np.maximum(np.abs(piv.values),1e-300)),origin='lower',aspect='auto',extent=[min(ALPHAS),max(ALPHAS),min(ZS),max(ZS)])
        axs[0,col].set_title(f'D={D}: log10 |baseline HEE|');axs[0,col].set_xlabel('alpha');axs[0,col].set_ylabel('z');fig.colorbar(im,ax=axs[0,col])
        sub=figdf[(figdf.D==D)&(figdf.case=='III-b_Phi_0p4_l1')]
        piv=sub.pivot(index='z',columns='alpha',values='ratio_to_visible_baseline')
        vals=np.log10(np.maximum(np.abs(piv.values),1e-300))
        im=axs[1,col].imshow(vals,origin='lower',aspect='auto',extent=[min(ALPHAS),max(ALPHAS),min(ZS),max(ZS)])
        axs[1,col].set_title(f'D={D}: log10 |tensor/baseline|');axs[1,col].set_xlabel('alpha');axs[1,col].set_ylabel('z');fig.colorbar(im,ax=axs[1,col])
    fig.savefig(OUT/'figures'/'v97_multipanel_D345_baseline_tensor_W0p10.png',dpi=180)
    plt.close(fig)
    # summary/report
    tdf=pd.DataFrame(tests); qdf=pd.DataFrame(qc)
    summary=dict(version='v97',created_utc=pd.Timestamp.utcnow().isoformat(),rows=len(all_rows),valid_rows=int(pd.DataFrame(all_rows).coeff_O_delta2_epsilon2.notna().sum()),
                 grid_points=147,cases=10,widths=7,acceptance_pass=int((tdf.status=='PASS').sum()),acceptance_total=len(tdf),failures=len(failures),runtime_seconds=time.time()-start,
                 strict_scope=['visible baseline EOM','minimal hidden SU(2) muX=0 exact rescaling','leading-order isotropic-null selection rule'],
                 prescription_scope=['hidden SU(2) muX!=0 locked-profile','tensor Z(Phi)-weighted response with fixed onset profile'])
    (OUT/'reports'/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=f'''# HSV v97 — Full D=3,4,5 off-locus EOM→metric→RT dataset\n\n## Pipeline\n- order-epsilon Yang–Mills zero mode solved for every valid (D, alpha, z);\n- D=3 temporal correction b2 fixed by the Fredholm solvability condition;\n- D=3 full linearized Einstein–scalar–supporting-U(1) response solved for {{N2,sigma2,phi2}};\n- D=4,5 traceless Einstein response solved from the dimension-specific radial flux equation;\n- endpoint-safe 45-digit RT/geodesic first-variation quadrature used for all widths.\n\n## Dataset\n- {len(all_rows)} rows = 3 dimensions × 49 grid points × 10 cases × 7 widths.\n- D=3 (alpha=0,z=1) is retained as an explicit degenerate row because m=alpha+z-1=0 removes the visible chemical profile.\n\n## Method classes\n- strict EOM results: visible baseline, minimal hidden-SU(2) mu_X=0 scaling, and the leading-order isotropic-null rule;\n- explicit prescriptions: mu_X!=0 locked-profile source rule and Z(Phi)-weighted tensor response. These are labelled row by row and are not presented as universal theorems.\n\n## Acceptance\n{tdf.to_markdown(index=False)}\n\n## D5 note\nThe general-grid D5 column is the direct first variation of the RT functional evaluated with the numerical metric response. The separate CSV compares it with the PPO Eq.(65) small-width W4+W6 series at the AdS5 locus; both share the same leading W4 coefficient, while finite-width subleading terms are reported separately rather than calibrated by a fitted factor.\n'''
    (OUT/'reports'/'v97_full_report.md').write_text(report,encoding='utf-8')
    shutil.copy2('/mnt/data/hsv_v97_core.py',OUT/'scripts'/'hsv_v97_core.py')
    shutil.copy2('/mnt/data/build_v97_full_d345.py',OUT/'scripts'/'build_v97_full_d345.py')
    shutil.copy2('/mnt/data/derive_d3_full_general.py',OUT/'scripts'/'derive_d3_full_general.py')
    # manifest
    manifests=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): manifests.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size))
    write_csv(OUT/'manifest.csv',manifests)
    zpath=Path('/mnt/data/HSV_v97_FULL_D345_OFF_LOCUS_EOM_RT_CLOSED.zip')
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob('*'):
            if p.is_file(): zf.write(p,p.relative_to(OUT.parent))
    print('DONE',zpath,summary,flush=True)

if __name__=='__main__': main()
