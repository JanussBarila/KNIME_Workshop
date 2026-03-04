# 5–10 Minute Presentation Outline

## Slide 1 — Objective & Business Context
**Speaker notes:**
- Goal: identify churn drivers and prioritize retention actions.
- Why now: churn reduction has immediate revenue and CLV impact.
- Deliverable: risk model + high-value targeting strategy.

## Slide 2 — Data & Method
**Speaker notes:**
- Dataset: 7,043 customer records with demographics, services, billing, and churn label.
- Data prep: cleaned `TotalCharges`, handled missing values, stratified train/test split.
- Models: Logistic Regression and Random Forest.

## Slide 3 — Model Performance
**Speaker notes:**
- Logistic test AUC: 0.856; Random Forest test AUC: 0.863.
- Selected model: Random Forest.
- Operating threshold: 0.30 (best F1 balance).

## Slide 4 — Main Churn Drivers
**Speaker notes:**
- Biggest factors: month-to-month contracts, low tenure, high monthly charges.
- Missing tech support/security increases churn risk.
- Electronic check payment method has elevated churn.

## Slide 5 — High Value Customer Definition
**Speaker notes:**
- Final definition: TotalCharges >= 3000, tenure >= 24, contract is one- or two-year.
- Size: 22.8% of customer base.
- At-risk within high-value: 1.1% at p(churn) >= 0.30.

## Slide 6 — Recommended Retention Actions
**Speaker notes:**
- 4 targeted incentives:
  1) contract renewal protection,
  2) bill optimization,
  3) support/security uplift,
  4) auto-pay migration incentive.
- Map each incentive to the driver it addresses.

## Slide 7 — Next-Month Execution Plan
**Speaker notes:**
- Week-by-week rollout: target list, launch, monitor, recalibrate.
- KPI tracking: save rate, churn reduction, ROI, high-value churn trend.

## Slide 8 — Decision Ask
**Speaker notes:**
- Approve 30-day pilot with threshold 0.30.
- Approve incentive budget and channel allocation.
- Commit to monthly model refresh and KPI review cadence.
