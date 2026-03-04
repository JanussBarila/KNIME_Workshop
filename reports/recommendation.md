# Churn Risk Recommendations

## (a) Account characteristics most linked to churn probability

Across both Logistic Regression and Random Forest diagnostics, the strongest churn-linked characteristics are:

1. **Contract structure (largest business driver):**
   - Month-to-month customers have much higher churn propensity.
   - One- and two-year contracts are strongly protective.
2. **Tenure:**
   - Short-tenure accounts are substantially more likely to churn.
3. **Price pressure:**
   - Higher `MonthlyCharges` is associated with higher churn risk.
4. **Service experience / value realization:**
   - No `OnlineSecurity` and no `TechSupport` increase risk.
   - Fiber optic accounts show elevated churn rates in descriptive data, likely reflecting pricing expectations and service-value mismatch.
5. **Payment behavior:**
   - Electronic check users exhibit the highest observed churn rates.

Descriptive churn rates reinforce these model patterns:
- Month-to-month churn rate: **42.7%** (vs 11.3% one-year, 2.8% two-year)
- Fiber optic churn rate: **41.9%**
- Electronic check churn rate: **45.3%**
- No tech support churn rate: **41.6%**

## (b) Proportion of high value customers at risk of terminating

### At-risk threshold choice
We define “at risk” as predicted churn probability **>= 0.30**.

**Reasoning:**
- Threshold selected by maximizing **F1** on holdout data, balancing precision and recall.
- At 0.30, recall is prioritized (0.767), which is suitable when retention teams prefer to capture more potential churners and can tolerate some false positives.

### Estimate for selected high value definition
High value definition selected:
- `TotalCharges >= 3000`
- `tenure >= 24`
- `Contract in {One year, Two year}`

Results:
- High value customers: **22.8%** of base (1,605 / 7,043)
- High value customers at risk (p >= 0.30): **1.1%** of high value segment (17 / 1,605)

This indicates the core value base is relatively stable, but a small subset still merits proactive save actions.

## (c) Incentives for at-risk high value customers

1. **Contract Renewal Protection Offer**
   - Target: At-risk high value accounts with contract nearing renewal or weaker commitment signals.
   - Offer: 12–24 month renewal with loyalty credit and price lock.
   - Rationale: Contract length is the strongest protective factor.

2. **Bill Optimization Bundle**
   - Target: High monthly-charge accounts with elevated churn risk.
   - Offer: Personalized plan review + right-sized bundle + temporary discount taper (e.g., 3 months).
   - Rationale: Reduces price shock while preserving long-term value.

3. **Premium Care + Security Add-on**
   - Target: At-risk customers lacking TechSupport/OnlineSecurity.
   - Offer: Complimentary 90-day premium support and security pack trial.
   - Rationale: Model signals these add-ons materially reduce churn propensity.

4. **Payment Method Retention Nudge**
   - Target: Electronic check users in high-risk buckets.
   - Offer: Incentive to switch to auto-pay (small recurring bill credit).
   - Rationale: Payment behavior is strongly associated with churn risk and can be operationally changed quickly.

## Implementation note
Prioritize intervention queue by:
1) high-value flag, then 2) predicted churn probability descending, then 3) margin/ARPU.
This concentrates retention spend where business value and risk are both highest.
