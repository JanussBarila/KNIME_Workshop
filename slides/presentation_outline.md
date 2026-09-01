# 5–10 Minute Presentation Outline

## Slide 1 — Objective

- Identify the strongest churn signals.
- Prioritize customers for retention action.
- Translate model output into an operational pilot.

## Slide 2 — Data and method

- 7,043 anonymized customer records.
- Cleaning, imputation, encoding, scaling, and stratified train/test split.
- Logistic Regression compared with Random Forest.

## Slide 3 — Model selection

- Logistic Regression test ROC-AUC: **0.842**.
- Random Forest test ROC-AUC: **0.820**, with near-perfect train AUC and clear overfitting.
- Logistic Regression selected for better generalization and interpretability.
- Operating threshold: **0.28**, selected by holdout F1.

## Slide 4 — Operating performance

- Recall: **0.781**.
- Precision: **0.518**.
- Accuracy: **0.749**.
- The threshold prioritizes finding likely churners over minimizing review volume.

## Slide 5 — Main churn signals

- Contract structure and tenure.
- Billing and price exposure.
- Security and technical-support services.
- Internet-service type and payment method.

## Slide 6 — High-value segment

- Rule: TotalCharges >= 3000, tenure >= 24, and a one- or two-year contract.
- Segment size: **1,605 customers (22.8%)**.
- Flagged at risk: **59 customers (3.7%)**.

## Slide 7 — Retention actions

1. Contract-renewal protection
2. Bill-optimization review
3. Support and security trial
4. Automatic-payment incentive

## Slide 8 — Decision ask

- Approve a controlled retention pilot.
- Use a holdout group to measure incremental impact.
- Track save rate, retained margin, offer acceptance, and cost per saved customer.
- Recalibrate the threshold from capacity and economic results.

