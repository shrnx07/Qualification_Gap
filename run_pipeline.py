"""
run_pipeline.py
---------------
End-to-end pipeline for the Qualification Gap project.

Outcome definition (loan_status from accepted file):
    Fully Paid                → outcome = 1 (approved / repaid)
    Charged Off, Default      → outcome = 0 (denied proxy / defaulted)
    Current, Late, Grace Period → dropped (outcome unknown)

This gives a genuine financial signal: features actually predict
repayment, giving meaningful AUC and valid gap calculations.
"""

import os, warnings
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report, roc_auc_score
from scipy.stats             import spearmanr

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
os.makedirs("db",      exist_ok=True)

FEATURES = ["loan_amnt", "annual_inc", "dti", "emp_length_yrs"]
BRACKET_LABELS = {
    1: "Low (<$40k)",
    2: "Lower-Mid ($40-70k)",
    3: "Upper-Mid ($70-100k)",
    4: "High (>$100k)",
}

def assign_bracket(inc):
    if inc < 40_000:  return 1
    if inc < 70_000:  return 2
    if inc < 100_000: return 3
    return 4

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and prepare data
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 1 — Loading and preparing data")
print("="*60)

ACCEPTED_COLS = [
    "loan_amnt", "annual_inc", "dti", "emp_length",
    "purpose", "loan_status", "addr_state", "issue_d",
]

print("Reading accepted CSV ...")
raw = pd.read_csv(
    "data/raw/accepted_2007_to_2018Q4.csv",
    usecols=ACCEPTED_COLS,
    low_memory=False,
)

# Filter to 2015+
raw["issue_dt"] = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")
raw = raw[raw["issue_dt"].dt.year >= 2015].copy()
raw["issue_year"]  = raw["issue_dt"].dt.year
raw["issue_month"] = raw["issue_dt"].dt.month

# Outcome from loan_status
APPROVED = {"Fully Paid"}
DENIED   = {"Charged Off", "Default"}
raw = raw[raw["loan_status"].isin(APPROVED | DENIED)].copy()
raw["outcome"] = raw["loan_status"].apply(lambda s: 1 if s in APPROVED else 0)

# Employment length
import re
def parse_emp(s):
    if pd.isna(s): return -1
    if "< 1" in str(s): return 0
    m = re.search(r"(\d+)", str(s))
    return min(int(m.group(1)), 10) if m else -1

raw["emp_length_yrs"] = raw["emp_length"].apply(parse_emp)
raw["dti"] = pd.to_numeric(raw["dti"], errors="coerce")

# Clean
raw = raw.dropna(subset=["annual_inc","dti"])
raw = raw[(raw["annual_inc"] > 0) & (raw["dti"] >= 0) & (raw["dti"] < 100)]
raw = raw[raw["emp_length_yrs"] >= 0].copy()
raw = raw.reset_index(drop=True)
raw.insert(0, "applicant_id", range(1, len(raw)+1))
raw["bracket_id"] = raw["annual_inc"].apply(assign_bracket)

approved = (raw["outcome"]==1).sum()
denied   = (raw["outcome"]==0).sum()
print(f"  Total usable  : {len(raw):,}")
print(f"  Fully Paid    : {approved:,}")
print(f"  Charged Off   : {denied:,}")
print(f"  Denial rate   : {denied/len(raw):.1%}")

print("\n  Denial rate by income bracket:")
for bid, label in BRACKET_LABELS.items():
    sub = raw[raw["bracket_id"]==bid]
    dr  = (sub["outcome"]==0).mean()
    print(f"    {label:28s}: {dr:.1%}  (n={len(sub):,})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load into DuckDB
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 2 — Loading into DuckDB")
print("="*60)

con = duckdb.connect("db/lending.db")
for tbl in ["gap_scores","applicants","income_brackets"]:
    con.execute(f"DROP TABLE IF EXISTS {tbl}")

con.execute("""
CREATE TABLE income_brackets (
    bracket_id INTEGER PRIMARY KEY, label VARCHAR NOT NULL,
    lower_bound DOUBLE NOT NULL,    upper_bound DOUBLE)""")
con.execute("""
INSERT INTO income_brackets VALUES
    (1,'Low (<$40k)',0,40000),(2,'Lower-Mid ($40-70k)',40000,70000),
    (3,'Upper-Mid ($70-100k)',70000,100000),(4,'High (>$100k)',100000,NULL)""")

keep_cols = ["applicant_id","loan_amnt","annual_inc","dti",
             "emp_length_yrs","purpose","addr_state",
             "issue_year","issue_month","outcome","bracket_id"]
df = raw[keep_cols].copy()

con.execute("""
CREATE TABLE applicants (
    applicant_id INTEGER PRIMARY KEY, loan_amnt DOUBLE,
    annual_inc DOUBLE, dti DOUBLE, emp_length_yrs INTEGER,
    purpose VARCHAR, addr_state VARCHAR, issue_year INTEGER,
    issue_month INTEGER, outcome INTEGER, bracket_id INTEGER)""")
con.execute("INSERT INTO applicants SELECT * FROM df")

con.execute("""
CREATE TABLE gap_scores (
    applicant_id INTEGER, model_name VARCHAR,
    gap_income_delta DOUBLE, gap_dti_delta DOUBLE,
    flip_achieved INTEGER, bracket_id INTEGER,
    PRIMARY KEY (applicant_id, model_name))""")

print(f"  Loaded {con.execute('SELECT COUNT(*) FROM applicants').fetchone()[0]:,} records")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train models
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 3 — Training models")
print("="*60)

X = df[FEATURES]; y = df["outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_sc, y_train)

dt = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
dt.fit(X_train, y_train)

model_results = []
for name, model, Xt in [("Logistic Regression", lr, X_test_sc),
                         ("Decision Tree",       dt, X_test)]:
    preds = model.predict(Xt)
    proba = model.predict_proba(Xt)[:,1]
    rep   = classification_report(y_test, preds, output_dict=True)
    auc   = roc_auc_score(y_test, proba)
    model_results.append({
        "Model":     name,
        "Accuracy":  round(rep["accuracy"],4),
        "Precision": round(rep["weighted avg"]["precision"],4),
        "Recall":    round(rep["weighted avg"]["recall"],4),
        "AUC":       round(auc,4),
    })
    print(f"  {name}: Accuracy={rep['accuracy']:.4f}  AUC={auc:.4f}")

model_table = pd.DataFrame(model_results)
print("\n  Model Comparison Table:")
print(model_table.to_string(index=False))

coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": lr.coef_[0]})
print("\n  LR Coefficients (annual_inc should be +, dti should be -):")
print(coef_df.sort_values("Coefficient", ascending=False).to_string(index=False))

fig, ax = plt.subplots(figsize=(20,8))
plot_tree(dt, feature_names=FEATURES, class_names=["Denied","Approved"],
          max_depth=3, filled=True, ax=ax)
fig.savefig("figures/decision_tree.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved figures/decision_tree.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Gap calculation
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 4 — Computing qualification gaps")
print("="*60)

INCOME_STEPS = np.arange(0, 60_001, 2_000)
DTI_STEPS    = np.arange(0, 0.401,  0.02)

def compute_gap(row, model, sc):
    base    = row[FEATURES].values.copy().astype(float)
    inc_idx = FEATURES.index("annual_inc")
    dti_idx = FEATURES.index("dti")
    gap_income = np.nan
    gap_dti    = np.nan
    for delta in INCOME_STEPS:
        cand = base.copy(); cand[inc_idx] += delta
        X_in = cand.reshape(1,-1)
        if sc is not None: X_in = sc.transform(X_in)
        if model.predict(X_in)[0] == 1: gap_income = delta; break
    for delta in DTI_STEPS:
        cand = base.copy(); cand[dti_idx] = max(0.0, cand[dti_idx]-delta)
        X_in = cand.reshape(1,-1)
        if sc is not None: X_in = sc.transform(X_in)
        if model.predict(X_in)[0] == 1: gap_dti = delta; break
    return {
        "gap_income_delta": gap_income,
        "gap_dti_delta":    gap_dti,
        "flip_achieved":    int(not (np.isnan(gap_income) and np.isnan(gap_dti))),
    }

denied_df = df[df["outcome"]==0].copy()
parts = []
for bid in [1,2,3,4]:
    sub = denied_df[denied_df["bracket_id"]==bid]
    parts.append(sub.sample(n=min(1250, len(sub)), random_state=42))
denied_sample = pd.concat(parts).reset_index(drop=True)
print(f"  Sample: {len(denied_sample):,} denied applicants (up to 1,250 per bracket)")

con.execute("DELETE FROM gap_scores")

for model_name, model, sc in [("logistic_regression", lr, scaler),
                                ("decision_tree",       dt, None)]:
    print(f"\n  Running {model_name} ...")
    rows = []
    for i, (_, row) in enumerate(denied_sample.iterrows()):
        if i % 500 == 0 and i > 0: print(f"    ... {i:,} done")
        gap = compute_gap(row, model, sc)
        gap["applicant_id"] = int(row["applicant_id"])
        gap["model_name"]   = model_name
        gap["bracket_id"]   = int(row["bracket_id"])
        rows.append(gap)
    gaps_df = pd.DataFrame(rows)[[
        "applicant_id","model_name",
        "gap_income_delta","gap_dti_delta",
        "flip_achieved","bracket_id"]]
    con.execute("INSERT INTO gap_scores SELECT * FROM gaps_df")
    print(f"    Flip rate: {gaps_df['flip_achieved'].mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Figures & tables
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 5 — Generating figures and tables")
print("="*60)

summary = con.execute("""
    SELECT b.label, b.bracket_id, g.model_name,
        COUNT(*)                          AS n,
        ROUND(AVG(g.gap_income_delta), 0) AS avg_income_gap,
        ROUND(AVG(g.gap_dti_delta),    3) AS avg_dti_gap,
        ROUND(AVG(g.flip_achieved),    3) AS flip_rate
    FROM gap_scores g
    JOIN income_brackets b USING (bracket_id)
    GROUP BY b.label, b.bracket_id, g.model_name
    ORDER BY b.bracket_id, g.model_name
""").df()

print("\n  CORE RESULTS TABLE:")
print(summary[["label","model_name","avg_income_gap","avg_dti_gap","flip_rate"]].to_string(index=False))

bracket_order = list(BRACKET_LABELS.values())
lr_sum = summary[summary["model_name"]=="logistic_regression"].copy()
lr_sum = lr_sum.set_index("label").reindex(bracket_order).reset_index()

# Figure 1
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(lr_sum["label"], lr_sum["avg_income_gap"],
              color="steelblue", edgecolor="white")
ax.bar_label(bars, fmt="$%.0f", padding=4, fontsize=9)
ax.set_title("Average Income Gap to Qualification by Income Bracket\n(Logistic Regression)", fontsize=12)
ax.set_ylabel("Average Income Increase Needed ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
plt.xticks(rotation=15, ha="right"); plt.tight_layout()
plt.savefig("figures/fig1_income_gap_by_bracket.png", dpi=150); plt.close()
print("  Saved figures/fig1_income_gap_by_bracket.png")

# Figure 2
raw_gaps = con.execute("""
    SELECT b.label, g.gap_income_delta, b.bracket_id
    FROM gap_scores g JOIN income_brackets b USING (bracket_id)
    WHERE g.model_name='logistic_regression' AND g.flip_achieved=1
    ORDER BY b.bracket_id
""").df()
fig, ax = plt.subplots(figsize=(10,5))
if len(raw_gaps) > 0:
    sns.boxplot(data=raw_gaps, x="label", y="gap_income_delta",
                order=bracket_order, palette="Blues", ax=ax)
    ax.set_title("Distribution of Income Gap by Bracket\n(Logistic Regression — flipped applicants only)", fontsize=12)
    ax.set_ylabel("Income Increase Needed ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
plt.xticks(rotation=15, ha="right"); plt.tight_layout()
plt.savefig("figures/fig2_gap_distribution.png", dpi=150); plt.close()
print("  Saved figures/fig2_gap_distribution.png")

# Figure 3
pivot = summary.pivot(index="label", columns="model_name", values="avg_income_gap")
pivot = pivot.reindex(bracket_order)
fig, ax = plt.subplots(figsize=(10,5))
pivot.plot.bar(ax=ax, color=["coral","steelblue"], edgecolor="white")
ax.set_title("Average Income Gap: Logistic Regression vs Decision Tree\n(Sensitivity Analysis)", fontsize=12)
ax.set_ylabel("Average Income Increase Needed ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
plt.xticks(rotation=15, ha="right"); plt.tight_layout()
plt.savefig("figures/fig3_sensitivity.png", dpi=150); plt.close()
print("  Saved figures/fig3_sensitivity.png")

lr_vals = summary[summary["model_name"]=="logistic_regression"]["avg_income_gap"].values
dt_vals = summary[summary["model_name"]=="decision_tree"]["avg_income_gap"].values
try:
    rho, p = spearmanr(lr_vals, dt_vals)
    print(f"\n  Spearman rho = {rho:.3f}  (p={p:.4f})")
except:
    rho, p = float("nan"), float("nan")
    print("  Spearman: insufficient variation to compute")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"\n  Records     : {con.execute('SELECT COUNT(*) FROM applicants').fetchone()[0]:,}")
print(f"  Gap scores  : {con.execute('SELECT COUNT(*) FROM gap_scores').fetchone()[0]:,}")
print(f"\n  Model Performance:")
print(model_table.to_string(index=False))
print(f"\n  Core Finding (Logistic Regression):")
print(lr_sum[["label","avg_income_gap","avg_dti_gap","flip_rate"]].to_string(index=False))
print(f"\n  Spearman rho = {rho:.3f}")
print("\n  Figures saved to figures/")
print("  Ready for your advisor meeting.")
