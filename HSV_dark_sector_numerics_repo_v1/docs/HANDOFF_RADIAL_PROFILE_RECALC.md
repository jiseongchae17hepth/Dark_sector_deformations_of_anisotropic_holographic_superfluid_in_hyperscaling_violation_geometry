
# HSV dark-sector radial-profile / solver handoff (v111)

## Immediate problem
The current radial-profile figures are not reliable evidence for dark-sector profile deformation.
The CSV used for figures, `hsv_selected_radial_profile_plot_data.csv`, explicitly labels all rows as
`selected_v98_profile_visible_shape` and says the profiles are representative selected profiles.
The audit file `audit/radial_profile_vs_visible_audit.csv` confirms:

- Case I: omega and b_profile are identical to visible baseline in the figure dataset.
- Case III-a: omega and b_profile are identical to visible baseline in the figure dataset.
- Case III-b: omega and b_profile are also identical to visible baseline in the figure dataset.
- Case II: omega differs only by the constant hidden-SU(2) scaling; b_profile is still identical.

Therefore, the present radial-profile figures are not generated from fully case-specific Yang--Mills branch solutions.
They should not be used to claim that omega(u) or b(u) changed, except for the explicit Case-II scaling already encoded.

## What actually produced the current HEE differences
The v98 build script says, for tensor rows:

`Tensor row recomputes the metric response with a radial Z-weighted YM stress while holding the onset profile fixed; model prescription explicitly recorded.`

In other words, Case III-b HEE deformation in the current dataset comes from reweighting the stress tensor source inside the metric-response solve, not from recomputing the Yang--Mills profiles w(u), b(u) under the modified equation of motion.

This is a model prescription, not a full profile solve.

## Consequence
If the paper wants to say that the dark sector deforms the order-parameter profile, then all radial profiles must be recomputed from the deformed Yang--Mills equations before metric backreaction and RT.
Current profile plots must be replaced or clearly labelled as representative visible-profile prescriptions.

## Required next solver task
For each case and each selected (D,theta,z), solve the correct branch in this order:

1. Build the case-specific Yang--Mills action/equations.
2. Solve zero mode / onset profile w1(u) with correct case-specific coefficients.
3. Solve temporal correction b2(u) if needed, especially D=3.
4. Compute the quadratic stress tensor source T^{(2)}_{MN}[w1,b0,b2,...].
5. Solve metric response g^{(2)}_{MN}.
6. Evaluate RT/geodesic first variation.
7. Output radial profiles with `profile_status=FULL_CASE_SPECIFIC_EOM`.

## Critical physics check
- If Case I or III-a is intended as an isotropic background deformation independent of epsilon, it may indeed be blind to leading anisotropic HEE at O(delta^2 epsilon^2). Then w,b need not differ at this order, but this must be stated as a selection rule, not shown as missing data.
- If Case III-b is a tensor portal with Z(Phi) multiplying the YM kinetic term, the w-equation should generally become

  (Z(u) P(u) w')' + Z(u) Q(u) w = 0

or whatever follows from the precise action. In that case w(u) should be recomputed. The current v98 prescription instead reweights only the metric source while holding w(u) fixed.

## Most relevant files in this package
- solvers/hsv_full_grid_hp_solver.py
- solvers/hsv_hee_solver.py
- solvers/D3_coupled_system_VERIFIED.py
- v98_scripts/hsv_v98_core.py
- v98_scripts/build_v98_full_d345_reference.py
- v98_scripts/derive_d3_full_general.py
- data/hsv_selected_radial_profile_plot_data.csv
- data/v98_FULL_D345_off_locus_HEE_all_cases.csv
- data/v98_case_source_rules.csv
- audit/radial_profile_vs_visible_audit.csv

## Do not repeat this mistake
Do not regenerate the radial-profile figure from `hsv_selected_radial_profile_plot_data.csv` as if it were a full dark-sector profile solution. It is a selected representative profile file and mostly uses the visible shape.
