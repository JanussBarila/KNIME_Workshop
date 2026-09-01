# Churn Risk Recommendations

## Main churn signals

Across model diagnostics and descriptive analysis, the strongest recurring signals are:

1. Month-to-month contracts and short tenure
2. Higher price exposure and cumulative charges
3. Missing online security or technical support
4. Fiber-optic service patterns that may reflect a value or service-expectation gap
5. Electronic-check payments

Descriptive churn rates reinforce these patterns: month-to-month customers, fiber-optic users, electronic-check users, and customers without technical support all show elevated churn.

## At-risk definition

A customer is flagged at risk when predicted churn probability is **at least 0.28**. This threshold maximized F1 on the holdout set and produced:

- Recall: **0.781**
- Precision: **0.518**
- ROC-AUC: **0.842**

The threshold favors recall, which is appropriate when the retention team can review a broader queue and the cost of missing a likely churner is high.

## High-value exposure

The selected high-value rule requires:

- `TotalCharges >= 3000`
- `tenure >= 24`
- `Contract` of one or two years

Results:

- High-value customers: **1,605 of 7,043 (22.8%)**
- High-value customers flagged at risk: **59 of 1,605 (3.7%)**

## Recommended interventions

1. **Contract renewal protection** — offer a loyalty credit or price lock for a 12–24 month renewal.
2. **Bill optimization review** — right-size high-charge accounts before price pressure becomes cancellation intent.
3. **Premium care and security trial** — give at-risk customers without support or security a time-limited service trial.
4. **Automatic-payment incentive** — encourage electronic-check users to move to automatic payment.

## Operating recommendation

Rank the contact queue by:

1. High-value flag
2. Predicted churn probability
3. Margin or ARPU

Start with a controlled pilot. Measure incremental save rate, retained margin, offer acceptance, and cost per saved customer against a holdout group before scaling.

