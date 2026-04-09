"""
clean.py
--------
Data cleaning functions for the Qualification Gap project.

Handles two source files from the Lending Club dataset:
  - accepted_2007_to_2018Q4.csv
  - rejected_2007_to_2018Q4.csv

Note: The rejected file does not contain annual_inc — this column is
set to NaN for rejected applicants. Income bracket assignment for gap
analysis is therefore based on accepted applicants only.

Outputs clean, typed DataFrames with a shared column schema,
ready for loading into DuckDB via schema.py.
"""

import re
import numpy as np
import pandas as pd


# ── Column selections ──────────────────────────────────────────────────────────

ACCEPTED_COLS = [
    "loan_amnt", "annual_inc", "dti", "emp_length",
    "purpose", "loan_status", "addr_state", "issue_d",
]

REJECTED_RENAME = {
    "Amount Requested":     "loan_amnt",
    "Debt-To-Income Ratio": "dti",
    "Employment Length":    "emp_length",
    "State":                "addr_state",
    "Loan Title":           "purpose",
    "Application Date":     "app_date",
}

# Final shared schema
OUTPUT_COLS = [
    "loan_amnt", "annual_inc", "dti", "emp_length_yrs",
    "purpose", "addr_state", "issue_year", "issue_month", "outcome",
]


# ── Helper ─────────────────────────────────────────────────────────────────────

def parse_emp_length(s) -> int:
    if pd.isna(s):
        return -1
    s = str(s)
    if "< 1" in s:
        return 0
    match = re.search(r"(\d+)", s)
    if match:
        return min(int(match.group(1)), 10)
    return -1


def _basic_filters(df: pd.DataFrame, require_income: bool = False) -> pd.DataFrame:
    if require_income:
        df = df.dropna(subset=["annual_inc", "dti"])
        df = df[(df["annual_inc"] > 0) & (df["dti"] >= 0) & (df["dti"] < 100)]
    else:
        df = df.dropna(subset=["dti"])
        df = df[(df["dti"] >= 0) & (df["dti"] < 100)]
    return df


# ── Main cleaning functions ────────────────────────────────────────────────────

def clean_accepted(df: pd.DataFrame, min_year: int = 2015) -> pd.DataFrame:
    df = df[ACCEPTED_COLS].copy()
    df["issue_dt"]       = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df                   = df[df["issue_dt"].dt.year >= min_year].copy()
    df["issue_year"]     = df["issue_dt"].dt.year
    df["issue_month"]    = df["issue_dt"].dt.month
    df["emp_length_yrs"] = df["emp_length"].apply(parse_emp_length)
    df["dti"]            = pd.to_numeric(df["dti"], errors="coerce")
    df["outcome"]        = 1
    df = _basic_filters(df, require_income=True)
    return df[OUTPUT_COLS].reset_index(drop=True)


def clean_rejected(
    df: pd.DataFrame,
    min_year: int = 2015,
    sample_n: int = 500_000,
    random_state: int = 42,
) -> pd.DataFrame:
    df = df.rename(columns=REJECTED_RENAME).copy()
    df["issue_dt"]    = pd.to_datetime(df["app_date"], format="%Y-%m-%d", errors="coerce")
    df                = df[df["issue_dt"].dt.year >= min_year].copy()
    df["issue_year"]  = df["issue_dt"].dt.year
    df["issue_month"] = df["issue_dt"].dt.month

    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state)
        print(f"  Sampled {sample_n:,} rejected rows")

    df["emp_length_yrs"] = df["emp_length"].apply(parse_emp_length)
    df["dti"] = (
        df["dti"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df["dti"]        = pd.to_numeric(df["dti"], errors="coerce")
    df["annual_inc"] = np.nan
    df["outcome"]    = 0

    if "purpose" not in df.columns:
        df["purpose"] = "unknown"
    df["purpose"] = df["purpose"].fillna("unknown")

    df = _basic_filters(df, require_income=False)
    return df[OUTPUT_COLS].reset_index(drop=True)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)

    print("Loading accepted loans ...")
    acc_raw = pd.read_csv(
        "data/raw/accepted_2007_to_2018Q4.csv",
        usecols=ACCEPTED_COLS,
        low_memory=False,
    )
    acc = clean_accepted(acc_raw)
    print(f"  Accepted (2015+): {len(acc):,} rows")
    acc.to_parquet("data/processed/accepted.parquet", index=False)
    print("  Saved to data/processed/accepted.parquet")

    print("\nLoading rejected loans ...")
    rej_raw = pd.read_csv("data/raw/rejected_2007_to_2018Q4.csv", low_memory=False)
    rej = clean_rejected(rej_raw, sample_n=500_000)
    print(f"  Rejected (sampled): {len(rej):,} rows")
    rej.to_parquet("data/processed/rejected.parquet", index=False)
    print("  Saved to data/processed/rejected.parquet")

    print(f"\nDone. Denial rate in combined sample: {len(rej)/(len(acc)+len(rej)):.1%}")