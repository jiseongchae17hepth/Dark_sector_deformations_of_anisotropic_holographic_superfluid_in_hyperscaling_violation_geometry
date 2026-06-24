#!/usr/bin/env python3
"""Lightweight validation of shipped HSV numerical CSV results.

This script does not regenerate the full grid. It verifies that the packaged
source-of-record CSVs are present and that the hard acceptance gates recorded
in v98_acceptance_tests.csv all passed.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "source_of_record"

required = [
    DATA / "v98_FULL_D345_off_locus_HEE_all_cases.csv",
    DATA / "v98_acceptance_tests.csv",
    DATA / "v98_case_source_rules.csv",
    DATA / "v98_dataset_dictionary.csv",
]
missing = [p for p in required if not p.exists()]
if missing:
    for p in missing:
        print(f"MISSING: {p}")
    sys.exit(1)

acc = pd.read_csv(DATA / "v98_acceptance_tests.csv")
failed = acc[acc["status"].astype(str).str.upper() != "PASS"]
print("Acceptance gates:", len(acc), "rows")
if len(failed):
    print(failed.to_string(index=False))
    sys.exit(2)
print("All recorded acceptance gates PASS.")

full = pd.read_csv(DATA / "v98_FULL_D345_off_locus_HEE_all_cases.csv")
print("Full HEE grid rows:", len(full))
print("Dimensions:", sorted(full["D"].dropna().unique().tolist()))
print("Cases:", sorted(full["case"].dropna().unique().tolist()))
print("Finite coeff rows:", int(np.isfinite(pd.to_numeric(full["coeff_O_delta2_epsilon2"], errors="coerce")).sum()))

rules = pd.read_csv(DATA / "v98_case_source_rules.csv")
print("Case source rules:")
print(rules.to_string(index=False))
