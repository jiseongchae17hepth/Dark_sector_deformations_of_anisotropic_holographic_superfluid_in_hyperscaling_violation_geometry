# HSV v166 real-solver and dense-CSV package

This package contains solver modules, solver workflow scripts, and dense CSV outputs used by the v166 manuscript package. It intentionally contains no figure-rendering scripts. The manuscript figures were rendered separately from the dense CSV tables; this package is meant for reproducing the numerical data layer.

Included content:

- `src/`: real solver/workflow modules.
- `scripts/`: solver and CSV-generation workflows, including dense scalar-BVP response generation.
- `data/`: dense CSV outputs used by the manuscript figures.
- `requirements.txt`: minimal Python dependencies.

Case-III-b note:

- `scripts/generate_v166_dense_scalar_and_response_csv.py` solves the sourced scalar BVP with `scipy.integrate.solve_bvp`.
- The dense Case-III-b response CSVs store the leading turning-point reduction `Phi^2(u_*(W))` of the full radial RT source.
- These CSVs are therefore the well-conditioned perturbative/short-width data used in the manuscript, not full metric-response quadratures at every grid point.

Excluded content:

- figure-rendering factories.
- image-writing scripts.
- the old v160 analytic-placeholder generator.
- raw unstable direct-subtraction RT tables.
