"""
gap.py
------
Counterfactual qualification gap calculation for the Qualification Gap project.

For each denied applicant, finds the minimum change in income or DTI that
would flip the model's prediction from Denied (0) to Approved (1).

This is the core analytical contribution of the project.

Usage:
    from src.gap import compute_gap, run_gap_analysis
    results_df = run_gap_analysis(denied_df, lr, scaler, model_name="logistic_regression")
"""

import numpy as np
import pandas as pd

from src.model   import FEATURES
from src.schema  import assign_bracket


# ── Default grid bounds ────────────────────────────────────────────────────────
# Chosen to represent realistic near-term changes an applicant could make.
# Income: up to $60k increase in $2k steps (~1-2 years of typical income growth)
# DTI: up to 40% reduction in 2% steps (aggressive but achievable via debt payoff)

DEFAULT_INCOME_STEPS = np.arange(0, 60_001, 2_000)   # $0 to $60k
DEFAULT_DTI_STEPS    = np.arange(0, 0.401,  0.02)    # 0.00 to 0.40 reduction


# ── Single-applicant gap ───────────────────────────────────────────────────────

def compute_gap(
    applicant_row: pd.Series,
    model,
    scaler,
    features: list = FEATURES,
    income_steps: np.ndarray = DEFAULT_INCOME_STEPS,
    dti_steps: np.ndarray    = DEFAULT_DTI_STEPS,
) -> dict:
    """
    Compute the qualification gap for a single denied applicant.

    Algorithm (grid search):
      1. Start from the applicant's actual feature values.
      2. Iteratively increase income in small steps until the model
         predicts Approved — record the minimum delta.
      3. Repeat for DTI reductions.
      4. Return both deltas and whether any flip was achieved.

    Parameters
    ----------
    applicant_row : Row from a denied-applicants DataFrame
    model         : Fitted sklearn classifier (LR or DT)
    scaler        : Fitted StandardScaler, or None for tree models
    features      : Feature list (must match model training order)
    income_steps  : Array of income deltas to try ($)
    dti_steps     : Array of DTI reductions to try (proportion)

    Returns
    -------
    dict with keys:
        gap_income_delta  — minimum income increase that flipped prediction (NaN if none)
        gap_dti_delta     — minimum DTI reduction that flipped prediction (NaN if none)
        flip_achieved     — 1 if either flip succeeded, 0 otherwise
    """
    base = applicant_row[features].values.copy().astype(float)
    inc_idx = features.index("annual_inc")
    dti_idx = features.index("dti")

    gap_income = np.nan
    gap_dti    = np.nan

    # ── Income grid search ──────────────────────────────────────────────────
    for delta in income_steps:
        candidate = base.copy()
        candidate[inc_idx] += delta

        X_in = candidate.reshape(1, -1)
        if scaler is not None:
            X_in = scaler.transform(X_in)

        if model.predict(X_in)[0] == 1:
            gap_income = delta
            break

    # ── DTI grid search ─────────────────────────────────────────────────────
    for delta in dti_steps:
        candidate = base.copy()
        candidate[dti_idx] = max(0.0, candidate[dti_idx] - delta)

        X_in = candidate.reshape(1, -1)
        if scaler is not None:
            X_in = scaler.transform(X_in)

        if model.predict(X_in)[0] == 1:
            gap_dti = delta
            break

    flip_achieved = int(not (np.isnan(gap_income) and np.isnan(gap_dti)))

    return {
        "gap_income_delta": gap_income,
        "gap_dti_delta":    gap_dti,
        "flip_achieved":    flip_achieved,
    }


# ── Batch gap calculation ──────────────────────────────────────────────────────

def run_gap_analysis(
    denied_df: pd.DataFrame,
    model,
    scaler,
    model_name: str,
    features: list = FEATURES,
    sample_n: int  = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run gap calculation over all (or a sample of) denied applicants.

    Parameters
    ----------
    denied_df    : DataFrame of denied applicants with applicant_id + features
    model        : Fitted classifier
    scaler       : Fitted StandardScaler or None
    model_name   : String label stored in gap_scores ('logistic_regression' etc.)
    sample_n     : If set, randomly sample this many rows (useful if full set is slow)
    random_state : Seed for sampling

    Returns
    -------
    DataFrame ready to INSERT into gap_scores table.
    """
    df = denied_df.copy()

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state)
        print(f"Sampled {sample_n:,} denied applicants (random_state={random_state})")

    print(f"Computing gaps for {len(df):,} denied applicants using {model_name} …")
    print("(This may take a few minutes. Progress logged every 1,000 rows.)")

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 1_000 == 0 and i > 0:
            print(f"  … {i:,} done")

        gap = compute_gap(row, model, scaler, features)
        gap["applicant_id"] = row["applicant_id"]
        gap["model_name"]   = model_name
        gap["bracket_id"]   = assign_bracket(row["annual_inc"])
        results.append(gap)

    results_df = pd.DataFrame(results)

    flip_rate = results_df["flip_achieved"].mean()
    print(f"\nDone. Flip achieved for {flip_rate:.1%} of denied applicants.")

    # Column order must match gap_scores schema
    return results_df[[
        "applicant_id", "model_name",
        "gap_income_delta", "gap_dti_delta", "gap_emp_delta" if "gap_emp_delta" in results_df.columns else "gap_income_delta",
        "flip_achieved", "bracket_id",
    ]] if "gap_emp_delta" in results_df.columns else results_df[[
        "applicant_id", "model_name",
        "gap_income_delta", "gap_dti_delta",
        "flip_achieved", "bracket_id",
    ]]


# ── Quick demo (single applicant) ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Gap calculation demo with a synthetic denied applicant:")
    print()

    # Fake a denied applicant: low income, high DTI
    demo_row = pd.Series({
        "applicant_id":  9999,
        "loan_amnt":     10_000,
        "annual_inc":    35_000,
        "dti":           0.42,
        "emp_length_yrs": 2,
        "annual_inc":    35_000,
    })

    print("Applicant profile:")
    print(f"  Income:       ${demo_row['annual_inc']:,.0f}")
    print(f"  DTI:          {demo_row['dti']:.0%}")
    print(f"  Emp length:   {demo_row['emp_length_yrs']} years")
    print()
    print("Run run_gap_analysis() with a real model to compute gaps.")
