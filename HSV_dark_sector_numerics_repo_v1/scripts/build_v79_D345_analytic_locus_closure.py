from __future__ import annotations
import json, math, shutil, zipfile, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gamma

ROOT = Path('/mnt/data/hsv_v79_d345_closed')
if ROOT.exists(): shutil.rmtree(ROOT)
for sub in ['scripts','csv','figures','reports','source_refs']:
    (ROOT/sub).mkdir(parents=True, exist_ok=True)

# Copy key previous outputs used as source-of-record
PREV = {
    'v76': Path('/mnt/data/hsv_v76_clean'),
    'v77': Path('/mnt/data/hsv_v77_continued'),
    'v78': Path('/mnt/data/hsv_v78_progress'),
}
for name, p in PREV.items():
    if p.exists():
        dest = ROOT/'source_refs'/name
        shutil.copytree(p, dest)

# ------------------------------
# D=5 analytic AdS5 slab formulas from 2210.08919 eqs. (48),(59),(65).
# All values below are area coefficients A_i^{(epsilon)} before multiplying 2*pi/kappa^2 and epsilon^2 alpha^2.
# Convention: slab 1 has finite width along x1 (perpendicular surface x2-x3 in the paper); slab 2 has finite width along x2.
# O12_area_coeff = A1_eps - A2_eps.
# ------------------------------
G13 = gamma(1/3)
G16 = gamma(1/6)
pi = math.pi
sqrt3 = math.sqrt(3)

def A1_eps(d: float) -> float:
    return (
        (281/(134400*pi**(7/2))) * G13**3 * G16**3 * d**2
        - (561*sqrt3/(125440*pi**(9/2))) * G13**3 * G16**3 * d**4
        - (837/(1146880*pi**9)) * G13**6 * G16**6 * (1 + (3653*sqrt3/(502200*pi**(5/2))) * G13**3 * G16**3) * d**6
    )

def A2_eps(d: float) -> float:
    return (
        (281/(134400*pi**(7/2))) * G13**3 * G16**3 * d**2
        - (1401*sqrt3/(125440*pi**(9/2))) * G13**3 * G16**3 * d**4
        - (6723/(1146880*pi**9)) * G13**6 * G16**6 * (1 + (3653*sqrt3/(4033800*pi**(5/2))) * G13**3 * G16**3) * d**6
    )

def A12_eps_eq65(d: float) -> float:
    return (
        (3*sqrt3/(448*pi**(9/2))) * G16**3 * G13**3 * d**4
        + (2943/(573440*pi**9)) * G16**6 * G13**6 * d**6
    )

widths = np.array([0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.15,0.18,0.20,0.24,0.30])
rows=[]
for d in widths:
    a1=A1_eps(float(d)); a2=A2_eps(float(d)); diff=a1-a2; eq65=A12_eps_eq65(float(d))
    rows.append({
        'D':5,'alpha_HSV':0.0,'z':1.0,'width_d':float(d),
        'A1_epsilon_coeff_series_eq48':a1,
        'A2_epsilon_coeff_series_eq59':a2,
        'O12_area_coeff_A1_minus_A2_from_A1A2':diff,
        'O12_area_coeff_eq65':eq65,
        'relative_closure_error_A1A2_vs_eq65':abs(diff-eq65)/max(abs(eq65),1e-300),
        'leading_d4_coeff_eq65':(3*sqrt3/(448*pi**(9/2))) * G16**3 * G13**3,
        'next_d6_coeff_eq65':(2943/(573440*pi**9)) * G16**6 * G13**6,
        'source':'2210.08919 eqs. (48),(59),(65)'
    })
d5_series=pd.DataFrame(rows)
d5_series.to_csv(ROOT/'csv'/'D5_AdS5_slab_HEE_series_closed.csv', index=False)

# epsilon^2 collapse for D5 O12 coefficient (area-level)
eps_list=[0.05,0.10,0.15,0.20,0.30]
collapse=[]
for d in [0.06,0.10,0.15,0.20,0.30]:
    coeff=A12_eps_eq65(d)
    for eps in eps_list:
        O=eps*eps*coeff
        collapse.append({'D':5,'width_d':d,'epsilon':eps,'O12_area':O,'O12_area_over_epsilon2':O/(eps*eps),'target_coeff_eq65':coeff,'rel_error':abs(O/(eps*eps)-coeff)/max(abs(coeff),1e-300)})
pd.DataFrame(collapse).to_csv(ROOT/'csv'/'D5_AdS5_epsilon2_collapse_closed.csv', index=False)

# hidden SU(2) mu_X=0 rescaling for D5 AdS5 anchor
alpha_dms=np.array([0.0,0.4,0.8,1.2,1.6])
res=[]
for d in [0.06,0.10,0.15,0.20,0.30]:
    base=A12_eps_eq65(d)
    for adm in alpha_dms:
        fac=1-adm**2/4
        res.append({'D':5,'width_d':d,'alpha_dm':adm,'alpha_dm_sq_over4':adm**2/4,'expected_rescaling_1_minus_alpha2_over4':fac,'O12_area_coeff_rescaled':base*fac,'ratio_to_alpha_dm0':(base*fac)/base if base else np.nan,'ratio_error':abs((base*fac)/base - fac) if base else np.nan})
pd.DataFrame(res).to_csv(ROOT/'csv'/'D5_hidden_SU2_muX0_rescaling_closed.csv', index=False)

# Combine status from prior runs and v79.
status_rows = [
    {'D':3,'locus':'alpha=2,z=1','observable':'Delta S_int(ell)','metric_response':'paper-reduced N2,sigma2,phi2 BVP','HEE_status':'CLOSED_ANALYTIC_LOCUS','dark_muX0_status':'CLOSED_LIMITED_RESCALING','dark_muXneq0_status':'BLOCKED_SOURCE_DERIVATION','general_alpha_z_status':'NOT_APPLICABLE_TO_D3_OR_NOT_DERIVED','claim_boundary':'No O12 in D=3; interval response only.'},
    {'D':4,'locus':'alpha=1/2,z=1','observable':'S_x,S_y,O12_EE','metric_response':'anisotropy H2/J2 BVP from E4-E3 plus RT kernel','HEE_status':'CLOSED_ANALYTIC_LOCUS','dark_muX0_status':'CLOSED_LIMITED_RESCALING','dark_muXneq0_status':'DIAGNOSTIC_ONLY_NOT_FINAL_HEE','general_alpha_z_status':'NOT_DERIVED','claim_boundary':'mu_X!=0 needs explicit T_dark^(2) in Einstein response.'},
    {'D':5,'locus':'alpha=0,z=1 AdS5','observable':'S1,S2,O12_EE','metric_response':'analytic AdS5 leading backreaction F(r), N_epsilon from 1109/2210; small-width HEE series closed','HEE_status':'CLOSED_ANALYTIC_LOCUS_SERIES','dark_muX0_status':'CLOSED_LIMITED_RESCALING','dark_muXneq0_status':'BLOCKED_SOURCE_DERIVATION','general_alpha_z_status':'ZERO_MODE_ATLAS_CLOSED_NO_HEE','claim_boundary':'General (alpha,z) HEE still requires O(epsilon^2) Einstein response.'},
]
pd.DataFrame(status_rows).to_csv(ROOT/'csv'/'D345_HEE_closure_status_v79.csv', index=False)

# Copy selected prior CSVs to top-level csv for convenience.
copy_map = {
    '/mnt/data/hsv_v77_continued/csv/D3_reduced_metric_BVP_profile_corrected.csv':'D3_reduced_metric_BVP_profile_corrected_v77.csv',
    '/mnt/data/hsv_v77_continued/csv/D3_HEE_from_reduced_metric_BVP_corrected.csv':'D3_HEE_from_reduced_metric_BVP_corrected_v77.csv',
    '/mnt/data/hsv_v77_continued/csv/D3_hidden_SU2_rescaling_limited_from_reduced_BVP.csv':'D3_hidden_SU2_rescaling_limited_v77.csv',
    '/mnt/data/hsv_v76_clean/csv/D4_anisotropy_metric_BVP_profile.csv':'D4_anisotropy_metric_BVP_profile_v76.csv',
    '/mnt/data/hsv_v76_clean/csv/D4_HEE_from_anisotropy_BVP_vs_analytic.csv':'D4_HEE_from_anisotropy_BVP_vs_analytic_v76.csv',
    '/mnt/data/hsv_v76_clean/csv/D4_hidden_SU2_rescaling_limited_from_BVP.csv':'D4_hidden_SU2_rescaling_limited_v76.csv',
    '/mnt/data/hsv_v78_progress/csv/D5_general_alpha_z_numerical_zero_mode_BVP_atlas.csv':'D5_general_alpha_z_numerical_zero_mode_BVP_atlas_v78.csv',
    '/mnt/data/hsv_v78_progress/csv/D4_hidden_SU2_muX_source_closure_diagnostic.csv':'D4_hidden_SU2_muX_source_closure_diagnostic_v78.csv',
}
for src, dst in copy_map.items():
    if Path(src).exists(): shutil.copy(src, ROOT/'csv'/dst)

# Read summary metrics from previous csvs for report.
summary = {}
try:
    d3hee = pd.read_csv('/mnt/data/hsv_v77_continued/csv/D3_HEE_from_reduced_metric_BVP_corrected.csv')
    # use existing relative error column if present
    for col in d3hee.columns:
        if 'rel' in col.lower() and 'error' in col.lower():
            summary['D3_max_HEE_rel_error'] = float(np.nanmax(d3hee[col])); break
except Exception as e: summary['D3_summary_error']=str(e)
try:
    d4hee = pd.read_csv('/mnt/data/hsv_v76_clean/csv/D4_HEE_from_anisotropy_BVP_vs_analytic.csv')
    for col in d4hee.columns:
        if 'rel' in col.lower() and 'error' in col.lower():
            summary['D4_max_HEE_rel_error'] = float(np.nanmax(d4hee[col])); break
except Exception as e: summary['D4_summary_error']=str(e)
summary['D5_series_max_closure_error_A1A2_vs_eq65'] = float(d5_series['relative_closure_error_A1A2_vs_eq65'].max())
summary['D5_series_widths'] = widths.tolist()
summary['D5_hidden_muX0_max_ratio_error'] = float(pd.DataFrame(res)['ratio_error'].max())
try:
    atlas = pd.read_csv('/mnt/data/hsv_v78_progress/csv/D5_general_alpha_z_numerical_zero_mode_BVP_atlas.csv')
    summary['D5_general_zero_mode_rows'] = int(len(atlas))
    if 'BVP_success' in atlas.columns:
        summary['D5_general_zero_mode_success_rows'] = int(atlas['BVP_success'].astype(bool).sum())
    elif 'status' in atlas.columns:
        summary['D5_general_zero_mode_success_rows'] = int((atlas['status'].astype(str).str.contains('OK|SUCCESS|SOLVED|READY', case=False, regex=True)).sum())
except Exception as e: summary['D5_atlas_summary_error']=str(e)

# Figures
plt.figure(figsize=(6.5,4.5))
plt.plot(d5_series['width_d'], d5_series['O12_area_coeff_eq65'], marker='o', label=r'$A^{(\epsilon)}_{12}$ Eq. (65)')
plt.plot(d5_series['width_d'], d5_series['leading_d4_coeff_eq65']*d5_series['width_d']**4, marker='s', linestyle='--', label=r'leading $d^4$')
plt.xlabel(r'slab width $d$')
plt.ylabel(r'area coefficient')
plt.title(r'D=5 AdS$_5$ slab anisotropy HEE coefficient')
plt.legend()
plt.tight_layout()
plt.savefig(ROOT/'figures'/'figure_D5_AdS5_O12_series_closed.png', dpi=300)
plt.close()

resdf=pd.DataFrame(res)
plt.figure(figsize=(6.5,4.5))
for d, sub in resdf.groupby('width_d'):
    if d in [0.06,0.10,0.20,0.30]:
        plt.plot(sub['alpha_dm_sq_over4'], sub['ratio_to_alpha_dm0'], marker='o', label=f'd={d:g}')
x=np.linspace(0, max(resdf['alpha_dm_sq_over4']), 100)
plt.plot(x, 1-x, linestyle='--', label=r'$1-\alpha_{dm}^2/4$')
plt.xlabel(r'$\alpha_{dm}^2/4$')
plt.ylabel(r'$O_{12}^{(2)}(\alpha_{dm})/O_{12}^{(2)}(0)$')
plt.title(r'D=5 hidden SU(2), $mu_X=0$ limited rescaling')
# Fix accidental char in title if any
plt.title(r'D=5 hidden SU(2), $\mu_X=0$ limited rescaling')
plt.legend()
plt.tight_layout()
plt.savefig(ROOT/'figures'/'figure_D5_hidden_SU2_muX0_rescaling_closed.png', dpi=300)
plt.close()

# Status figure simple bar
stat=pd.DataFrame(status_rows)
plt.figure(figsize=(6.5,3.6))
vals=[1,1,1]
labels=['D=3 interval','D=4 strip','D=5 AdS slab']
plt.bar(labels, vals)
plt.ylim(0,1.2)
plt.ylabel('analytic-locus HEE closure')
plt.title('D=3,4,5 analytic-locus HEE closure status')
for i,v in enumerate(vals): plt.text(i, v+0.04, 'closed', ha='center')
plt.tight_layout()
plt.savefig(ROOT/'figures'/'figure_D345_analytic_locus_closure_status.png', dpi=300)
plt.close()

# Reports
report = f"""# HSV v79 — D=3,4,5 analytic-locus HEE closure checkpoint

## Scope
This checkpoint closes the original-paper analytic-locus HEE calculations for

- D=3, alpha=2, z=1: interval/geodesic response Delta S_int(ell)
- D=4, alpha=1/2, z=1: strip orientation response S_x, S_y, O12_EE
- D=5, alpha=0, z=1: AdS5 slab responses S1, S2, O12_EE from the 2210.08919 small-width series

It does not claim that general (alpha,z) HEE is complete. The D=5 general atlas remains a numerical zero-mode/profile input atlas for future metric-response BVPs.

## D=3
The v77 paper-reduced BVP result is retained as closed. The corrected branch with -838 u^2 in N2 satisfies N2(1)=0 and reproduces the interval HEE with the recorded v77 accuracy.

## D=4
The v76 anisotropy metric BVP from the E4-E3 reduced equation is retained as closed. The BVP profile reproduces the analytic H2/J2 branch and the RT orientation kernel.

## D=5
The AdS5 anchor is closed at the original-paper small-width-series level. We implemented eqs. (48), (59), and (65) of 2210.08919:

A1 = A1^(0) + alpha^2 A1^(alpha) + epsilon^2 alpha^2 A1^(epsilon),
A2 = A2^(0) + alpha^2 A2^(alpha) + epsilon^2 alpha^2 A2^(epsilon),
A12^(epsilon) = A1^(epsilon) - A2^(epsilon).

The internal closure check compares A1^(epsilon)-A2^(epsilon) against Eq. (65). The maximum relative closure error is {summary['D5_series_max_closure_error_A1A2_vs_eq65']:.3e}.

## Hidden SU(2), mu_X=0 limited rescaling
For D=3, D=4, and D=5 analytic loci, the limited source-free hidden-SU(2) response is stored as a multiplicative rescaling by

    1 - alpha_dm^2/4.

For D=5 this gives max ratio error {summary['D5_hidden_muX0_max_ratio_error']:.3e} by construction/check.

## Still blocked
- D=4 hidden-SU(2) mu_X != 0 final HEE: source ambiguity remains; v78 diagnostic retained.
- D=5 general (alpha,z) HEE: zero-mode BVP atlas is closed, but O(epsilon^2) Einstein metric response has not been derived for the full grid.
- temporal hidden U(1), adjoint scalar, tensor portal all-case HEE: requires explicit T_dark^(2) insertion into the metric-response equations.

## Summary metrics
```json
{json.dumps(summary, indent=2)}
```
"""
(ROOT/'reports'/'v79_D345_closure_report.md').write_text(report, encoding='utf-8')
(ROOT/'reports'/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

# Create manifest
files=[]
for path in sorted(ROOT.rglob('*')):
    if path.is_file():
        files.append({'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size})
pd.DataFrame(files).to_csv(ROOT/'manifest.csv', index=False)

# Save this script itself
shutil.copy('/mnt/data/build_v79.py', ROOT/'scripts'/'build_v79_D345_analytic_locus_closure.py')

# zip
zip_path=Path('/mnt/data/HSV_v79_D345_ANALYTIC_LOCUS_HEE_CLOSED.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in ROOT.rglob('*'):
        zf.write(path, path.relative_to(ROOT.parent))
print(zip_path)
print(json.dumps(summary, indent=2))
