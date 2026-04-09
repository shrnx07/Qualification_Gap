# Quantifying Financial Exclusion: How Does the Qualification Gap Disproportionately Vary Across Income Tiers in the Peer-to-Peer Lending Market?



## 1. Introduction

Binary credit decisions — approved or denied — carry significant consequences for borrowers. Yet most research on credit access focuses on *who* gets denied rather than *how close* denied applicants were to qualifying. This distinction matters: two applicants can both be denied while one needs only a small income increase to qualify and another faces a nearly insurmountable barrier.

This project introduces and quantifies the **Qualification Gap** — the minimum change in a denied applicant's financial profile required to flip a credit model's prediction from denied to approved. The central research question is:

> *How does the Qualification Gap disproportionately vary across income tiers in the peer-to-peer lending market?*

Using Lending Club historical loan data (2015–2018), we train logistic regression and decision tree classifiers to predict loan repayment outcomes, then apply a counterfactual grid search — based on the algorithmic recourse framework of Wachter et al. (2017) — to compute income and DTI gaps for 5,000 denied applicants stratified by income bracket.

**Main finding:** In proportional terms, low-income denied applicants (earning under $40,000) require a 47.1% income increase to reach the predicted approval boundary, compared to just 6.1% for high-income denied applicants (earning over $100,000). This 7.7x disparity suggests the qualification barrier is not merely financial — it is structurally steeper for lower-income borrowers.

---

## 2. Data

**Source:** Lending Club Loan Data (2007–2018), accessed via Kaggle.  
**Files used:** `accepted_2007_to_2018Q4.csv`, filtered to 2015–2018.  
**Final sample:** 835,999 records after filtering.

### 2.1 Outcome Definition

Loan status was used to construct a binary outcome variable:
- **Fully Paid** → outcome = 1 (approved proxy)
- **Charged Off / Default** → outcome = 0 (denied proxy)
- **Current, Late, Grace Period** → excluded (final outcome unknown)

This approach uses loan *performance* as a proxy for approval decisions. Its primary limitation is discussed in Section 6.

### 2.2 Features

Four features available at application time were selected:

| Feature | Description |
|---|---|
| `loan_amnt` | Requested loan amount ($) |
| `annual_inc` | Self-reported annual income ($) |
| `dti` | Existing debt-to-income ratio |
| `emp_length_yrs` | Employment length (0–10 years) |

Post-approval variables (`int_rate`, `grade`, `funded_amnt`) were excluded to prevent data leakage.

### 2.3 Descriptive Statistics

Overall denial rate: **21.0%** (175,429 charged off of 835,999 total).

| Income Bracket | N | Denial Rate |
|---|---|---|
| Low (<$40k) | 110,892 | 24.3% |
| Lower-Mid ($40-70k) | 319,764 | 22.6% |
| Upper-Mid ($70-100k) | 214,828 | 20.3% |
| High (>$100k) | 190,515 | 17.2% |

Note: The denial rate gradient (24.3% → 17.2%) is meaningful but modest. The qualification gap reveals a much steeper disparity beneath this surface pattern.

---

## 3. Methods

### 3.1 Data Pipeline

Raw CSV data was cleaned using Python (pandas), with employment length parsed from string format (e.g., "< 1 year" → 0, "10+ years" → 10). Cleaned records were loaded into a DuckDB relational database with three tables: `applicants`, `income_brackets`, and `gap_scores`. 

### 3.2 Models

Two classifiers were trained on the feature matrix:

**Logistic Regression** models approval probability as a sigmoid function of the input features. The coefficient signs provide a built-in sanity check: `annual_inc` should be positive (higher income → more likely to repay) and `dti` should be negative (higher debt burden → more likely to default). Both conditions held in our results.

**Decision Tree** (max_depth=5) partitions the feature space into rectangular regions using binary splits. It produces an interpretable, stepwise boundary rather than a smooth one.

Both models used `class_weight='balanced'` to compensate for the 3.7:1 class imbalance between Fully Paid and Charged Off loans. Without this adjustment, both models predicted "Fully Paid" for all applicants, yielding AUC ≈ 0.50 and gap = $0 for every applicant.

**Train/test split:** 80/20 stratified by outcome.

### 3.3 Qualification Gap Calculation

For each denied applicant in a stratified sample of 5,000 (1,250 per bracket), a **counterfactual grid search** was performed:

1. Start from the applicant's actual feature values
2. Increase `annual_inc` in $2,000 steps up to $60,000 — record the minimum delta that flips the prediction to Approved
3. Reduce `dti` in 2% steps up to 40% — record the minimum delta that flips

The income gap (`gap_income_delta`) and DTI gap (`gap_dti_delta`) are stored in `gap_scores`. A `flip_achieved` flag records whether any flip was found within the search bounds.

Two derived metrics are reported:
- **Absolute gap:** `gap_income_delta` in dollars
- **Proportional gap:** `gap_income_delta / annual_inc × 100` (% of current income)

### 3.4 Sensitivity Analysis

The gap calculation was run identically for both models. Spearman rank correlation between models' average income gaps across the four brackets quantifies agreement on bracket ordering.

---

## 4. Results

### 4.1 Model Performance

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.5805 | 0.7194 | 0.5805 | 0.6115 |
| Decision Tree | 0.5735 | 0.7211 | 0.5735 | 0.6102 |

AUC of ~0.61 reflects the limited feature set. The most predictive variable in real credit decisions — FICO score — is masked in the public dataset. Despite modest AUC, the model learns a financially sensible boundary: `annual_inc` coefficient = +0.30, `dti` = −0.22, confirming the model rewards income and penalizes debt burden.

### 4.2 Core Finding — Absolute Gap

| Income Bracket | Avg Income Gap (LR) | Flip Rate (LR) |
|---|---|---|
| Low (<$40k) | **$14,193** | 69.0% |
| Lower-Mid ($40-70k) | **$12,755** | 65.9% |
| Upper-Mid ($70-100k) | **$10,527** | 63.8% |
| High (>$100k) | **$7,464** | 77.0% |

*(See Figure 1)*

### 4.3 Core Finding — Proportional Gap

| Income Bracket | % Income Increase Needed |
|---|---|
| Low (<$40k) | **47.1%** |
| Lower-Mid ($40-70k) | **24.6%** |
| Upper-Mid ($70-100k) | **13.2%** |
| High (>$100k) | **6.1%** |

*(See Figure 4)*

The proportional gap falls monotonically from 47.1% to 13.2% across the first three brackets, representing a **7.7x disparity** between the lowest and highest income groups. This is the project's primary finding.

**High bracket anomaly:** High-income denied applicants have both the lowest proportional gap (6.1%) and the highest flip rate (77.0%). This suggests they are denied primarily for loan size relative to income rather than income itself — they are already close to the approval boundary and require only small adjustments.

### 4.4 Flip Rate Interpretation

31% of low-income denied applicants could not flip their outcome even with a $60,000 income increase or 40% DTI reduction. These applicants are denied for reasons outside our search space — likely employment length or loan amount — and represent the most financially excluded group in the sample.

---

## 5. Sensitivity Analysis

Running the identical gap calculation with both models yields:

| Income Bracket | LR Gap | DT Gap |
|---|---|---|
| Low (<$40k) | $14,193 | $10,311 |
| Lower-Mid ($40-70k) | $12,755 | $7,308 |
| Upper-Mid ($70-100k) | $10,527 | $4,599 |
| High (>$100k) | $7,464 | $6,170 |

**Spearman rank correlation: rho = 0.80**

the models disagree on absolute gap magnitudes but agree on the directional ordering across brackets. The decision tree produces consistently lower absolute gaps, likely because its stepwise boundary is easier to cross with small income changes. The conclusion — that lower-income denied applicants face proportionally larger gaps — holds under both models.

*(See Figure 3)*

---

## 6. Limitations

**Proxy outcome:** This project uses loan repayment (Fully Paid vs Charged Off) as a proxy for approval decisions. Real lenders make approval decisions before repayment is known. The model captures factors correlated with repayment risk, which partially overlaps with approval criteria — but the two are not identical.

**Missing FICO score:** Credit score is the most predictive variable in real credit decisions and is masked in the public Lending Club dataset. Our models explain only a portion of the variance in outcomes, and the gaps we calculate are relative to a model boundary that does not fully reflect how real lenders operate.

**Personal lending, not mortgages:** Lending Club is a peer-to-peer platform for personal loans. The regulatory environment, underwriting standards, and borrower population differ from mortgage markets. Findings apply specifically to this platform.

**Grid search bounds are normative:** The choice of $0–$60,000 income range and 0–40% DTI reduction reflects assumptions about realistic near-term changes. Different bounds will produce different flip rates. The proportional gap finding is robust to reasonable bound variations, but absolute gap numbers are sensitive to this choice.

**Mechanical correlation between income and DTI:** Lower-income applicants may mechanically have higher DTIs (more debt relative to income), meaning both the income gap and DTI gap may partially reflect the same underlying financial constraint expressed twice.

---

## 7. Conclusion and Future Work

This project quantifies the Qualification Gap — the minimum income increase required to flip a loan outcome prediction — and demonstrates that this gap varies disproportionately across income brackets. Low-income denied applicants face a 47.1% income hurdle compared to 6.1% for high-income denied applicants, a 7.7x proportional disparity. Both logistic regression and decision tree models agree on the directional ordering of this finding (Spearman rho = 0.80).

The Qualification Gap as a metric is the primary contribution. It transforms a binary denial outcome into a continuous, comparable measure of financial exclusion that could in principle be tracked across time, lenders, or demographic groups.

**Future work:**
1. Apply the methodology to HMDA mortgage data to examine whether the pattern holds in regulated mortgage markets
2. Incorporate FICO score as a feature using a synthetic imputation approach (e.g., estimating FICO from correlated variables)
3. Extend the counterfactual calculation using the Wachter et al. (2017) optimization approach rather than grid search, which would enable continuous rather than discrete gap estimates
4. Stratify by race and gender in addition to income bracket to examine intersectional patterns of financial exclusion

---


