# Email Memo to Management

**Subject:** Churn Risk Analysis — Findings, High-Value Exposure, and Pilot Recommendation

## Executive summary

The analysis covers 7,043 anonymized customer records and compares Logistic Regression with Random Forest. Logistic Regression was selected because it generalized better on the holdout set (ROC-AUC **0.842** versus **0.820**) and is easier to explain. The Random Forest achieved near-perfect train performance but showed clear overfitting.

For operational targeting, a probability threshold of **0.28** was selected by holdout F1. At this point, recall is **0.781** and precision is **0.518**. The threshold intentionally captures a broad share of likely churners; it should be recalibrated if contact capacity or intervention cost changes.

The analysis consistently highlights contract structure, tenure, billing exposure, missing security or support services, and payment method. These are useful targeting signals, but they should not be interpreted as causal effects without controlled testing.

## High-value exposure

The selected high-value rule combines cumulative spend, tenure, and commitment:

- TotalCharges >= 3000
- Tenure >= 24 months
- One- or two-year contract

This segment contains **1,605 customers (22.8%)**. The model flags **59 customers (3.7%)** of the segment at or above the 0.28 churn threshold.

## Recommended actions

1. Offer renewal price protection or a loyalty credit.
2. Review and right-size high-charge plans.
3. Provide a time-limited premium support and security trial.
4. Encourage electronic-check users to move to automatic payment.

The operating queue should rank customers by high-value status, predicted churn probability, and then margin or ARPU.

## Pilot design

Run a controlled pilot with a holdout group. Track:

- Incremental save rate
- Retained contribution margin
- Offer acceptance by intervention
- Cost per saved customer
- High-value churn trend

This analysis is pilot-ready decision support, not a production deployment. Before scaling, add out-of-time validation, probability calibration, drift monitoring, bias checks, and campaign-uplift measurement.

