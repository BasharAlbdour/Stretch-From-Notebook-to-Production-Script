# Tree vs. Linear Disagreement Analysis

## Sample Details

- **Test-set index:** 4060
- **True label:** 0
- **RF predicted P(churn=1):** 0.5998
- **LR predicted P(churn=1):** 0.1700
- **Probability difference:** 0.4299

## Feature Values

- **tenure:** 36.0
- **monthly_charges:** 20.0
- **total_charges:** 1077.33
- **num_support_calls:** 2.0
- **senior_citizen:** 0.0
- **has_partner:** 0.0
- **has_dependents:** 0.0
- **contract_months:** 1.0

## Structural Explanation

The Random Forest flags this customer as high-risk (P=0.60) while Logistic Regression
sees low risk (P=0.17), despite a true label of 0. The RF likely captured a threshold
interaction between contract_months=1 (month-to-month contract) and has_partner=0,
has_dependents=0 — a combination the training data associates with churn even at
moderate tenure. Logistic Regression cannot express this conjunction: it assigns a
fixed negative weight to low monthly_charges=20 which pulls the score down linearly,
missing the non-monotonic pattern where low charges + short contract + no anchor
(partner/dependents) is actually a risky profile the tree learned as a specific decision path.
