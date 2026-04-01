# Measuring the Qualification Gap in Loan Approval

**Author:** Second-semester Freshman, Beloit College  
**Supervisor:** Prof. Disha Shende — Data Analytics / Data Science / Economics  
**Course:** 0.5-unit Independent Study  

---

## Project Summary

This project asks a deceptively simple question: among people who were *denied* a loan, how close were they to getting approved — and does that closeness vary by income bracket?

The answer requires building a complete analytical system: clean raw data, design a relational schema, train a classification model, and run a structured what-if calculation to produce a concrete measure called the **Qualification Gap**.

The Qualification Gap for a denied applicant is the **minimum change in income or debt-to-income ratio** that would have flipped the model's prediction from Denied to Approved. Aggregating those gaps across income brackets reveals whether lower-income denied applicants are structurally farther from qualification — or whether the barriers are roughly uniform across income levels.

---

## Data Source

**Lending Club Loan Data (2007–2018) via Kaggle**  
https://www.kaggle.com/datasets/wordsforthewise/lending-club

Files used:
- `accepted_2007_to_2018Q4.csv` (filtered to 2015–2018)
- `rejected_2007_to_2018Q4.csv` (filtered to 2015–2018)

> Raw data is not committed to this repo. Download from Kaggle and place in `data/raw/`.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/qualification-gap.git
cd qualification-gap

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

---

## Repository Structure

```
qualification-gap/
├── data/
│   ├── raw/          # Original CSVs from Kaggle — never modified
│   └── processed/    # Cleaned parquet files output by clean.py
├── db/
│   └── lending.db    # DuckDB database file (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb        # Week 1: Exploratory analysis
│   ├── 02_model.ipynb      # Week 2: Model training & evaluation
│   ├── 03_gap.ipynb        # Week 3: Gap calculation
│   └── 04_sensitivity.ipynb # Week 4: Sensitivity analysis
├── src/
│   ├── clean.py      # Data cleaning functions
│   ├── schema.py     # DuckDB schema + loading logic
│   ├── model.py      # Model training and evaluation
│   └── gap.py        # Counterfactual grid search
├── figures/          # All saved plots
├── report/           # Technical report drafts
├── requirements.txt
└── README.md
```

---

## Running the Pipeline

Run notebooks in order:

```
notebooks/01_eda.ipynb       → EDA + cleaning
notebooks/02_model.ipynb     → Train logistic regression + decision tree
notebooks/03_gap.ipynb       → Compute qualification gaps
notebooks/04_sensitivity.ipynb → Compare models, sensitivity analysis
```

---

## Key Findings

*(To be updated after Week 3 analysis is complete)*

---

## Limitations

- Uses Lending Club **personal loan** data as a proxy for credit decisions — findings are specific to this platform and cannot be directly generalised to mortgage markets.
- Classification models (logistic regression, decision tree) are statistical proxies for the lender's actual decision algorithm.
- Grid search bounds reflect assumptions about realistic financial changes — different bounds will produce different flip rates.
- Dataset restricted to 2015–2018 and may not reflect current lending conditions.

---

## References

- Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual Explanations without Opening the Black Box. *Harvard Journal of Law & Technology, 31*, 841–887.
- Karimi, A. H., Schölkopf, B., & Valera, I. (2021). Algorithmic Recourse: From Counterfactual Explanations to Interventions. *FAccT 2021*.
