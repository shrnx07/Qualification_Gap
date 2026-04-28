"""
gap.py — Counterfactual qualification gap calculation
"""
import numpy as np
import pandas as pd
from src.schema import assign_bracket

FEATURES     = ["loan_amnt", "annual_inc", "dti", "emp_length_yrs"]
INCOME_STEPS = np.arange(0, 60_001, 2_000)
DTI_STEPS    = np.arange(0, 0.401,  0.02)

def compute_gap(row, model, scaler):
    base    = row[FEATURES].values.copy().astype(float)
    inc_idx = FEATURES.index("annual_inc")
    dti_idx = FEATURES.index("dti")
    gap_income = np.nan
    gap_dti    = np.nan

    for delta in INCOME_STEPS:
        cand = base.copy(); cand[inc_idx] += delta
        X_in = cand.reshape(1,-1)
        if scaler is not None: X_in = scaler.transform(X_in)
        if model.predict(X_in)[0] == 1: gap_income = delta; break

    for delta in DTI_STEPS:
        cand = base.copy(); cand[dti_idx] = max(0.0, cand[dti_idx]-delta)
        X_in = cand.reshape(1,-1)
        if scaler is not None: X_in = scaler.transform(X_in)
        if model.predict(X_in)[0] == 1: gap_dti = delta; break

    return {"gap_income_delta": gap_income, "gap_dti_delta": gap_dti,
            "flip_achieved": int(not (np.isnan(gap_income) and np.isnan(gap_dti)))}

def run_gap_analysis(denied_df, model, scaler, model_name, sample_n=None, random_state=42):
    df = denied_df.copy()
    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state)
    print(f"Computing gaps for {len(df):,} denied applicants using {model_name}...")
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0 and i > 0: print(f"  ... {i:,} done")
        gap = compute_gap(row, model, scaler)
        gap["applicant_id"] = int(row["applicant_id"])
        gap["model_name"]   = model_name
        gap["bracket_id"]   = int(row["bracket_id"])
        rows.append(gap)
    results_df = pd.DataFrame(rows)
    print(f"Done. Flip rate: {results_df['flip_achieved'].mean():.1%}")
    return results_df[["applicant_id","model_name","gap_income_delta","gap_dti_delta","flip_achieved","bracket_id"]]
