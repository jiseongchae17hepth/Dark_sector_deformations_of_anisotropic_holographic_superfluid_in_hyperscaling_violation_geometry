# HSV v98 — Complete D=3,4,5 off-locus HEE dataset

## Result
- 10,290 rows = 3 dimensions × 49 (alpha,z) points × 10 cases × 7 widths.
- 10,220 rows contain finite HEE coefficients.
- The remaining 70 rows are the single repeated physical corner D=3, alpha=0, z=1, where m=alpha+z-1=0 eliminates the finite visible chemical profile.
- All 27 acceptance gates pass.

## Theory pipeline
1. Solve the order-epsilon visible Yang–Mills zero mode for every nondegenerate (D,alpha,z).
2. In D=3, solve the order-epsilon^2 temporal correction b2 by the Fredholm solvability condition, including the 2 b0' b2' electric source.
3. Solve the D=3 coupled linearized Einstein–scalar–supporting-U(1) response for N2, sigma2, phi2; solve the D=4,5 traceless Einstein radial response for the anisotropic metric function.
4. Evaluate the geodesic/RT functional at first order in the metric perturbation with endpoint-safe high-precision quadrature.

## Two HEE domains, stored separately
- `coeff_paper_small_width`: the original-paper small-width perturbative convention, retained unchanged wherever its asymptotic turning point lies outside the horizon.
- `coeff_finite_width_first_variation`: exact unperturbed black-brane turning point plus fixed-boundary first variation, used for 280 wide rows where the small-width estimate would lie at or inside the horizon.
- `coeff_O_delta2_epsilon2`: canonical complete column selecting the valid method row by row. No fitted or locus-derived multiplicative correction is used.

## Acceptance gates
| test                         |    expected |    measured |       error |   tolerance | status   |
|:-----------------------------|------------:|------------:|------------:|------------:|:---------|
| mu_D3_a2_z1                  | 4           | 4           | 1.05883e-11 |      2e-05  | PASS     |
| mu_D4_a0.5_z1                | 4           | 4           | 1.05883e-11 |      2e-05  | PASS     |
| mu_D5_a0_z1                  | 4           | 4           | 1.05883e-11 |      2e-05  | PASS     |
| mu_D5_a1_z1                  | 5.67955     | 5.67955     | 1.13205e-12 |      2e-05  | PASS     |
| D3_b2_mu2                    | 0.0105655   | 0.0105655   | 3.98059e-07 |      2e-05  | PASS     |
| D3_metric_profile_rel        | 0           | 1.60266e-05 | 1.60266e-05 |      0.0001 | PASS     |
| D3_HEE_L0p1                  | 0.000516871 | 0.000516867 | 7.87726e-06 |      5e-05  | PASS     |
| D4_response_profile_rel      | 0           | 2.7964e-06  | 2.7964e-06  |      0.0001 | PASS     |
| D4_HEE_W0p05                 | 8.02756e-07 | 8.027e-07   | 6.98909e-05 |      0.0002 | PASS     |
| D4_smallW_power              | 4           | 3.99663     | 0.00336769  |      0.01   | PASS     |
| D5_response_profile_rel      | 0           | 2.7964e-06  | 2.7964e-06  |      0.0001 | PASS     |
| D5_Eq65_leading_W4           | 0.222752    | 0.222671    | 0.000363734 |      0.001  | PASS     |
| D3_II_aDM_0p4_ratio          | 0.96        | 0.96        | 0           |      2e-10  | PASS     |
| D3_II_aDM_0p8_ratio          | 0.84        | 0.84        | 0           |      2e-10  | PASS     |
| D3_II_aDM_1p2_ratio          | 0.64        | 0.64        | 0           |      2e-10  | PASS     |
| D3_I_temporal_U1_qX_0p5_null | 1           | 1           | 0           |      2e-10  | PASS     |
| D3_III-a_Phi_0p4_null        | 1           | 1           | 0           |      2e-10  | PASS     |
| D4_II_aDM_0p4_ratio          | 0.96        | 0.96        | 0           |      2e-10  | PASS     |
| D4_II_aDM_0p8_ratio          | 0.84        | 0.84        | 0           |      2e-10  | PASS     |
| D4_II_aDM_1p2_ratio          | 0.64        | 0.64        | 0           |      2e-10  | PASS     |
| D4_I_temporal_U1_qX_0p5_null | 1           | 1           | 0           |      2e-10  | PASS     |
| D4_III-a_Phi_0p4_null        | 1           | 1           | 0           |      2e-10  | PASS     |
| D5_II_aDM_0p4_ratio          | 0.96        | 0.96        | 0           |      2e-10  | PASS     |
| D5_II_aDM_0p8_ratio          | 0.84        | 0.84        | 0           |      2e-10  | PASS     |
| D5_II_aDM_1p2_ratio          | 0.64        | 0.64        | 0           |      2e-10  | PASS     |
| D5_I_temporal_U1_qX_0p5_null | 1           | 1           | 0           |      2e-10  | PASS     |
| D5_III-a_Phi_0p4_null        | 1           | 1           | 0           |      2e-10  | PASS     |

## Claim scope
- Strict: visible baseline, minimal hidden-SU(2) mu_X=0 rescaling, and the stated leading-order isotropic-null selection rule.
- Prescription-labelled: hidden-SU(2) mu_X!=0 locked profile and tensor Z(Phi)-weighted stress with fixed onset profile.
- Ratios with tiny visible baselines are not clipped; absolute values and reliability flags are included.
