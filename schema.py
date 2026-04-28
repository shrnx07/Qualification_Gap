"""
schema.py — DuckDB schema definitions and loading logic
"""
import duckdb
import pandas as pd

CREATE_APPLICANTS = """
CREATE TABLE IF NOT EXISTS applicants (
    applicant_id   INTEGER PRIMARY KEY,
    loan_amnt      DOUBLE,
    annual_inc     DOUBLE,
    dti            DOUBLE,
    emp_length_yrs INTEGER,
    purpose        VARCHAR,
    addr_state     VARCHAR,
    issue_year     INTEGER,
    issue_month    INTEGER,
    outcome        INTEGER,
    bracket_id     INTEGER
);"""

CREATE_INCOME_BRACKETS = """
CREATE TABLE IF NOT EXISTS income_brackets (
    bracket_id   INTEGER PRIMARY KEY,
    label        VARCHAR NOT NULL,
    lower_bound  DOUBLE  NOT NULL,
    upper_bound  DOUBLE
);"""

INSERT_BRACKETS = """
INSERT INTO income_brackets VALUES
    (1, 'Low (<$40k)',          0,      40000),
    (2, 'Lower-Mid ($40-70k)',  40000,  70000),
    (3, 'Upper-Mid ($70-100k)', 70000,  100000),
    (4, 'High (>$100k)',        100000, NULL);"""

CREATE_GAP_SCORES = """
CREATE TABLE IF NOT EXISTS gap_scores (
    applicant_id     INTEGER,
    model_name       VARCHAR,
    gap_income_delta DOUBLE,
    gap_dti_delta    DOUBLE,
    flip_achieved    INTEGER,
    bracket_id       INTEGER,
    PRIMARY KEY (applicant_id, model_name)
);"""

def create_schema(con):
    con.execute(CREATE_APPLICANTS)
    con.execute(CREATE_INCOME_BRACKETS)
    count = con.execute("SELECT COUNT(*) FROM income_brackets").fetchone()[0]
    if count == 0:
        con.execute(INSERT_BRACKETS)
    con.execute(CREATE_GAP_SCORES)
    print("Schema ready.")

def assign_bracket(annual_inc):
    if annual_inc < 40_000:  return 1
    if annual_inc < 70_000:  return 2
    if annual_inc < 100_000: return 3
    return 4

QUERY_GAP_BY_BRACKET = """
SELECT
    b.label, g.model_name,
    COUNT(*)                          AS n,
    ROUND(AVG(g.gap_income_delta), 0) AS avg_income_gap,
    ROUND(AVG(g.gap_dti_delta),    3) AS avg_dti_gap,
    ROUND(AVG(g.flip_achieved),    3) AS flip_rate
FROM gap_scores g
JOIN income_brackets b ON b.bracket_id = g.bracket_id
GROUP BY b.label, b.bracket_id, g.model_name
ORDER BY b.bracket_id, g.model_name;"""

if __name__ == "__main__":
    con = duckdb.connect("db/lending.db")
    create_schema(con)
    print(con.execute("SELECT * FROM income_brackets").df())
