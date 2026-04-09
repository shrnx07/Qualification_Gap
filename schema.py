"""
schema.py
---------
DuckDB schema definitions and data-loading logic for the Qualification Gap project.

Three tables:
  applicants      — one row per loan application (accepted + rejected)
  income_brackets — lookup table for aggregation
  gap_scores      — one row per (denied applicant × model) after gap calculation

Usage:
    import duckdb
    from src.schema import create_schema, load_applicants, assign_bracket

    con = duckdb.connect("db/lending.db")
    create_schema(con)
    load_applicants(con, accepted_df, rejected_df)
"""

import duckdb
import pandas as pd


# ── DDL statements ─────────────────────────────────────────────────────────────

CREATE_APPLICANTS = """
CREATE TABLE IF NOT EXISTS applicants (
    applicant_id   INTEGER PRIMARY KEY,
    loan_amnt      DOUBLE,
    annual_inc     DOUBLE,
    dti            DOUBLE,       -- debt-to-income ratio (e.g. 0.20 = 20%)
    emp_length_yrs INTEGER,      -- 0–10; -1 means missing
    purpose        VARCHAR,
    addr_state     VARCHAR(2),
    issue_year     INTEGER,
    issue_month    INTEGER,
    outcome        INTEGER NOT NULL  -- 1 = approved, 0 = denied
);
"""

CREATE_INCOME_BRACKETS = """
CREATE TABLE IF NOT EXISTS income_brackets (
    bracket_id   INTEGER PRIMARY KEY,
    label        VARCHAR NOT NULL,
    lower_bound  DOUBLE  NOT NULL,
    upper_bound  DOUBLE            -- NULL means no upper bound (top bracket)
);
"""

INSERT_BRACKETS = """
INSERT INTO income_brackets VALUES
    (1, 'Low (<$40k)',          0,      40000),
    (2, 'Lower-Mid ($40-70k)',  40000,  70000),
    (3, 'Upper-Mid ($70-100k)', 70000,  100000),
    (4, 'High (>$100k)',        100000, NULL);
"""

CREATE_GAP_SCORES = """
CREATE TABLE IF NOT EXISTS gap_scores (
    applicant_id      INTEGER REFERENCES applicants(applicant_id),
    model_name        VARCHAR NOT NULL,   -- 'logistic_regression' or 'decision_tree'
    gap_income_delta  DOUBLE,             -- minimum income increase to flip prediction
    gap_dti_delta     DOUBLE,             -- minimum DTI reduction to flip prediction
    gap_emp_delta     INTEGER,            -- minimum emp_length increase (optional)
    flip_achieved     INTEGER NOT NULL,   -- 1 if a flip was found, 0 if not
    bracket_id        INTEGER REFERENCES income_brackets(bracket_id),
    PRIMARY KEY (applicant_id, model_name)
);
"""


# ── Schema creation ────────────────────────────────────────────────────────────

def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create all tables in the connected DuckDB database.
    Safe to call multiple times (uses IF NOT EXISTS).
    Brackets are inserted only once.
    """
    con.execute(CREATE_APPLICANTS)
    con.execute(CREATE_INCOME_BRACKETS)

    # Only insert brackets if the table is empty
    count = con.execute("SELECT COUNT(*) FROM income_brackets").fetchone()[0]
    if count == 0:
        con.execute(INSERT_BRACKETS)

    con.execute(CREATE_GAP_SCORES)
    print("Schema created (or already exists).")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_applicants(
    con: duckdb.DuckDBPyConnection,
    accepted_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> None:
    """
    Combine accepted and rejected DataFrames, assign sequential IDs,
    and load into the applicants table.

    Idempotent: clears the table before loading so re-runs are safe.

    Parameters
    ----------
    con          : Active DuckDB connection
    accepted_df  : Output of clean.clean_accepted()
    rejected_df  : Output of clean.clean_rejected()
    """
    combined = pd.concat([accepted_df, rejected_df], ignore_index=True)
    combined.insert(0, "applicant_id", range(1, len(combined) + 1))

    # Wipe and reload for idempotency
    con.execute("DELETE FROM applicants")
    con.execute("INSERT INTO applicants SELECT * FROM combined")

    total = con.execute("SELECT COUNT(*) FROM applicants").fetchone()[0]
    approved = con.execute("SELECT COUNT(*) FROM applicants WHERE outcome=1").fetchone()[0]
    denied   = con.execute("SELECT COUNT(*) FROM applicants WHERE outcome=0").fetchone()[0]

    print(f"Loaded {total:,} applicants — {approved:,} approved, {denied:,} denied")
    print(f"Overall denial rate: {denied/total:.1%}")


# ── Bracket assignment (Python-side helper) ───────────────────────────────────

def assign_bracket(annual_inc: float) -> int:
    """
    Map annual income to income_brackets.bracket_id.

    Used when writing gap_scores from Python before loading into DuckDB.
    """
    if annual_inc < 40_000:
        return 1
    if annual_inc < 70_000:
        return 2
    if annual_inc < 100_000:
        return 3
    return 4


# ── Useful analytical queries ──────────────────────────────────────────────────

QUERY_DENIAL_RATE_BY_BRACKET = """
SELECT
    b.label,
    COUNT(*)                                          AS n_applicants,
    SUM(CASE WHEN a.outcome = 0 THEN 1 ELSE 0 END)   AS n_denied,
    ROUND(AVG(CASE WHEN a.outcome = 0 THEN 1.0 ELSE 0.0 END), 3) AS denial_rate
FROM applicants a
JOIN income_brackets b
  ON a.annual_inc >= b.lower_bound
 AND (b.upper_bound IS NULL OR a.annual_inc < b.upper_bound)
GROUP BY b.label, b.bracket_id
ORDER BY b.bracket_id;
"""

QUERY_GAP_BY_BRACKET = """
SELECT
    b.label,
    g.model_name,
    COUNT(*)                                AS n_denied,
    ROUND(AVG(g.gap_income_delta), 0)       AS avg_income_gap,
    ROUND(AVG(g.gap_dti_delta),    3)       AS avg_dti_gap,
    ROUND(AVG(g.flip_achieved),    3)       AS flip_rate
FROM gap_scores g
JOIN income_brackets b USING (bracket_id)
GROUP BY b.label, b.bracket_id, g.model_name
ORDER BY b.bracket_id, g.model_name;
"""


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    con = duckdb.connect("db/lending.db")
    create_schema(con)
    print("\nIncome brackets loaded:")
    print(con.execute("SELECT * FROM income_brackets").df())
