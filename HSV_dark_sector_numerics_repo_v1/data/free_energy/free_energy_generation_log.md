# v100 free-energy dataset update

Generated `hsv_free_energy_plot_data.csv` using a 1609-style probe Yang--Mills on-shell free-energy diagnostic.

Rows: 735. Base grid rows: 147.

Important scope: this is a probe thermodynamic diagnostic matching the original 1609 plotting convention, not a full holographically renormalized grand potential.

Formula basis:
- F_iso / V_d = - mu^2 m / 2, m=d theta+z+d-2.
- DeltaF is computed from the YM on-shell action density in the u-coordinate with finite small horizon vector amplitude epsilon=0.001.
- y_value = tanh(DeltaF / V_d), following the 1609 plotting convention.

Case rules:
- visible_baseline: direct nonlinear branch solve.
- case_I and case_IIIa: leading anisotropy-sector null rule; same probe free-energy diagnostic as visible baseline.
- case_II: minimal hidden SU(2) mu_X=0 scaling by 1-alpha_dm^2/4=0.84.
- case_IIIb: tensor-portal source-weight prescription using v98 HEE response ratio at W=0.10, flagged as prescription.

Solver successes: 146/147 base grid points. Degenerate points are retained with NaN.
