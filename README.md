# HSV dark-sector p-wave superfluid numerics

This repository package contains the Python solver files and CSV output tables used for the HSV dark-sector holographic-entanglement calculations.  It is intentionally limited to solver code, validation code, and numerical result tables.  Figure-generation notebooks and presentation-only graphics are not included.

## What is analytic, and what is numerical?

There is an analytic island inherited from the HSV p-wave literature.  In the notation used here, with `d = D - 2`, analytic Yang--Mills zero-mode solutions exist when

```text
d(theta + 1) = 3,
z = 1,
sqrt(3) * mu = 4.
```

Equivalently, the analytic anchor points are

```text
D = 3: theta = 2,   z = 1
D = 4: theta = 1/2, z = 1
D = 5: theta = 0,   z = 1
```

At this locus the visible p-wave solution contains

```text
b0(u) = 4(1 - u^-2)
w1(u) = u^2 / (1 + u^2)^2
```

and the subleading temporal correction is

```text
b2(u) = 71/6720 * (1 - u^-2)
        + (5 + 7u^2 - 9u^4 - 3u^6)/(96 u^2 (1 + u^2)^3).
```

The D=3 and D=4 leading metric backreactions are also known analytically on this island.  The D=3 coupled Einstein-scalar residual gate used in this project is implemented in `scripts/D3_coupled_system_VERIFIED.py`.

Away from this island, the Sturm--Liouville zero mode, metric response, and RT/interval integrals are not available in closed form in the workflow used here.  The off-locus grid is therefore obtained numerically:

```text
Yang--Mills zero mode -> quadratic stress-tensor source
-> O(delta^2 epsilon^2) metric response
-> RT/geodesic first variation.
```

## Important scope note

The CSV file `data/profiles/v98_selected_radial_profiles.csv` and the figure-table file `data/figure_tables/hsv_selected_radial_profile_plot_data.csv` contain selected representative radial profiles.  These profiles are useful for baseline diagnostics, but they should **not** be interpreted as a full case-specific dark-sector profile solve for every deformation.  In particular, the tensor-portal rows used in the current HEE tables are based on a `Z(Phi)`-weighted source/metric-response prescription.  If one wants to claim that the tensor portal changes the actual `omega(u)` and `b(u)` profiles, the Yang--Mills equations must be re-solved with the deformed gauge-kinetic operator.

See `docs/HANDOFF_RADIAL_PROFILE_RECALC.md` and `data/audit/radial_profile_vs_visible_audit.csv`.

## Directory layout

```text
scripts/
  Core Python solvers and reproduction scripts.

data/source_of_record/
  Source-of-record v98 CSV tables for the D=3,4,5 off-locus HEE grid and validation gates.

data/figure_tables/
  CSV tables used downstream for paper figures: mu, HEE, free-energy diagnostic, selected radial profiles, and Case-II scaling.

data/free_energy/
  Probe Yang--Mills free-energy diagnostic tables and restricted renormalized-potential candidate tables.

data/profiles/
  Selected radial profiles.

data/audit/
  Method-status, completeness, convergence, and radial-profile audit tables.

reports/
  Summary reports from the source-of-record package.

docs/
  Handoff notes and method caveats.
```

## Main source-of-record tables

```text
data/source_of_record/v98_FULL_D345_off_locus_HEE_all_cases.csv
```

Full D=3,4,5 HEE-response table.  Dimensions, theta/z grid, widths, cases, HEE coefficient, visible baseline coefficient, ratio-to-visible, and method flags are included.

```text
data/source_of_record/v98_acceptance_tests.csv
```

Hard acceptance gates.  All rows should have `status = PASS`.

```text
data/source_of_record/v98_case_source_rules.csv
```

Defines how each deformation is handled: strict visible EOM, hidden-SU(2) scaling, isotropic-null rule, or portal prescription.

```text
data/source_of_record/v98_zero_mode_metric_response_QC.csv
```

QC data for zero modes and metric response.

## Main scripts

```text
scripts/build_v98_full_d345_reference.py
scripts/hsv_v98_core.py
scripts/fill_finite_width_extension.py
scripts/derive_d3_full_general.py
scripts/D3_coupled_system_VERIFIED.py
```

These are the main source-of-record scripts from the v98 build.

Additional scripts are included for provenance and comparison:

```text
scripts/hsv_full_grid_hp_solver.py
scripts/hsv_full_grid_hp_solver_v95_corrected_core.py
scripts/hsv_hee_solver.py
scripts/build_v79_D345_analytic_locus_closure.py
scripts/build_v87_final_closure.py
```

## Quick validation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the lightweight acceptance check:

```bash
python scripts/run_acceptance_checks.py
```

Expected output includes:

```text
All recorded acceptance gates PASS.
Full HEE grid rows: 10290
Finite coeff rows: 10220
```

The 70 unavailable rows are the physically degenerate corner `D=3, theta=0, z=1`, where the visible chemical profile degenerates.

## Reproducing the full grid

The full grid-generation script is:

```bash
python scripts/build_v98_full_d345_reference.py
```

The exact runtime depends on the machine and numerical tolerances.  The script uses the core routines in `scripts/hsv_v98_core.py` and writes the source-of-record CSV tables.  For quick checks, use `scripts/run_acceptance_checks.py` first.

## Dark-sector interpretation used in the tables

The tables distinguish the following cases:

```text
baseline
I_temporal_U1_qX_0p5
II_aDM_0p4
II_aDM_0p8
II_aDM_1p2
II-b_aDM_0p4_muX_0p5
III-a_Phi_0p2
III-a_Phi_0p4
III-b_Phi_0p2_l1
III-b_Phi_0p4_l1
```

- Case I and Case II are hidden gauge-sector deformations.
- Case III-a and Case III-b are scalar-sector or portal deformations.
- Case II with `mu_X=0` has the analytic scaling rule `1 - alpha_dm^2/4` for the leading HEE coefficient.
- Case I and Case III-a are leading-order isotropic-null controls for anisotropic HEE.
- Case III-b is a scalar gauge-kinetic portal prescription in these tables.

## Citation/provenance notes

The analytic anchor follows the known p-wave analytic island and its HSV D=3/D=4 extensions.  The off-locus tables in this repository are numerical products of the shipped solver scripts.  The repository is prepared for reproducibility and audit, not as a polished plotting package.
