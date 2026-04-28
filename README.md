# Quantifying Financial Exclusion: The Qualification Gap in Peer-to-Peer Lending

**Author:** Sai Sharan Balagopal — Beloit College  
**Faculty Supervisor:** Prof. Disha Shende — Data Analytics, Data Science & Economics  
**Course:** 0.5-unit Independent Study  
**Semester:** Spring 2026

---

## Overview

This project introduces and quantifies the **Qualification Gap** — the minimum change in a denied applicant's financial profile required to flip a credit model's prediction from denied to approved — and examines whether this gap varies disproportionately across income brackets in the peer-to-peer lending market.

Using 835,999 Lending Club loan records (2015–2018), we train two classifiers to predict loan repayment outcomes and apply a counterfactual grid search to compute income and DTI gaps for 5,000 stratified denied applicants.

**Core finding:** Low-income denied applicants require a **47.1% income increase** to reach the predicted approval boundary, compared to just **6.1%** for high-income denied applicants — a **7.7x proportional disparity**.

---

## Research Question

> *How does the Qualification Gap disproportionately vary across income tiers in the peer-to-peer lending market?*

---

## Repository Structure

```
qualification-gap/
├── run_pipeline.py              # End-to-end pipeline — run this first
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── clean.py                 # Data cleaning and feature engineering
│   ├── schema.py                # DuckDB schema definitions and loading
│   ├── model.py                 # Model training and evaluation
│   └── gap.py                   # Counterfactual grid search (core method)
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_model.ipynb           # Model training and evaluation
│   ├── 03_gap.ipynb             # Gap calculation and core results
│   └── 04_sensitivity.ipynb     # Sensitivity and robustness checks
│
├── data/
│   ├── raw/                     # Place Kaggle CSVs here (gitignored)
│   └── processed/               # Cleaned parquet files (gitignored)
│
├── db/
│   └── lending.db               # DuckDB database (gitignored)
│
├── figures/
│   ├── fig1_income_gap_by_bracket.png
│   ├── fig2_gap_distribution.png
│   ├── fig3_sensitivity.png
│   ├── fig4_proportional_gap.png
│   └── decision_tree.png
│
└── report/
    └── technical_report_draft.md
```

---

## Data Requirements

**Source:** [Lending Club Loan Data (2007–2018) — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Download and place in `data/raw/`:
- `accepted_2007_to_2018Q4.csv` (~1.6 GB)
- `rejected_2007_to_2018Q4.csv` (optional — not used in final pipeline)

> Raw data is not committed to this repository due to file size.

**Outcome variable construction:**
- `Fully Paid` → outcome = 1 (approved proxy)
- `Charged Off` / `Default` → outcome = 0 (denied proxy)
- `Current`, `Late`, `Grace Period` → excluded (unknown final outcome)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/qualification-gap.git
cd qualification-gap

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\Activate.ps1    # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place Kaggle CSVs in data/raw/
mkdir -p data/raw
# mv ~/Downloads/accepted_2007_to_2018Q4.csv data/raw/
```

---

## Running the Pipeline

Run the full end-to-end pipeline with one command:

```bash
python run_pipeline.py
```

**Runtime:** approximately 10–15 minutes on a standard laptop.

**What it does:**
1. Loads and filters accepted loans (2015–2018)
2. Constructs outcome variable from `loan_status`
3. Loads 835,999 records into DuckDB
4. Trains Logistic Regression and Decision Tree classifiers
5. Runs counterfactual gap calculation on 5,000 stratified denied applicants
6. Generates all four figures
7. Prints the core results table

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Data loading, distributions, denial rates by income bracket |
| `02_model.ipynb` | Feature selection rationale, model training, ROC curves, coefficient analysis |
| `03_gap.ipynb` | Counterfactual grid search, absolute and proportional gap results, Figures 1 and 4 |
| `04_sensitivity.ipynb` | LR vs Decision Tree comparison, Spearman rank correlation, Figure 3 |

Run notebooks in order using Jupyter Lab:

```bash
jupyter lab
```

---

## Methods

**Classification models**
- Logistic Regression (`class_weight='balanced'`)
- Decision Tree (max_depth=5, `class_weight='balanced'`)
- Features: `loan_amnt`, `annual_inc`, `dti`, `emp_length_yrs`

**Counterfactual Grid Search (Qualification Gap)**
Based on the algorithmic recourse framework of Wachter et al. (2017). For each denied applicant:
- Increase income in $2,000 steps up to $60,000 → find minimum flip delta
- Reduce DTI in 2% steps up to 40% → find minimum flip delta

**Sensitivity Analysis**
- Both models run identically; Spearman rank correlation measures bracket-ordering agreement
- Spearman ρ = 0.80 → finding is robust to model choice

---

## Key Results

| Income Bracket | Avg Income Gap | % Income Increase | Flip Rate |
|---|---|---|---|
| Low (<$40k) | $14,193 | **47.1%** | 69.0% |
| Lower-Mid ($40–70k) | $12,755 | **24.6%** | 65.9% |
| Upper-Mid ($70–100k) | $10,527 | **13.2%** | 63.8% |
| High (>$100k) | $7,464 | **6.1%** | 77.0% |

**Model performance:** AUC ≈ 0.61 for both models (limited by masked FICO score in public dataset)

**Sensitivity:** Spearman ρ = 0.80 — both models agree on bracket ordering

---

## Future Work

- Add FICO score via synthetic imputation from correlated variables
- Apply methodology to HMDA mortgage data for regulatory-grade analysis
- Extend counterfactual search using Wachter et al. (2017) optimization (continuous gaps)
- Stratify by race and gender alongside income for intersectional analysis
- Oaxaca–Blinder decomposition to formally separate endowment vs. structural components

---

## References

- Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual Explanations without Opening the Black Box. *Harvard Journal of Law & Technology, 31*, 841–887.
- Karimi, A. H., Schölkopf, B., & Valera, I. (2021). Algorithmic Recourse: From Counterfactual Explanations to Interventions. *FAccT 2021*.
- King, R. G., & Levine, R. (1993). Finance and Growth: Schumpeter Might Be Right. *Quarterly Journal of Economics, 108*(3), 717–737.

---

*This project was completed as a 0.5-unit independent study at Beloit College, Spring 2026.*
