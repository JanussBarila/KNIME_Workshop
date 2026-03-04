# Email Memo to Management

**Subject:** Churn Risk Analysis – Findings, High-Value Exposure, and Next-Month Action Plan

## Executive Summary
We completed an end-to-end churn modeling exercise on the customer base (7,043 accounts) and built two predictive models: Logistic Regression and Random Forest. Both models performed strongly; Random Forest provided the best holdout discrimination (Test ROC AUC: **0.863**, vs **0.856** for Logistic Regression). For operational targeting, we selected a churn-risk threshold of **0.30**, chosen by F1 optimization to balance precision and recall.

The analysis shows churn risk is most strongly associated with: (1) month-to-month contracts, (2) low tenure, (3) higher monthly charges, and (4) absence of value-protection services such as tech support and online security. Descriptive patterns confirm these findings, including notably high churn among month-to-month customers and electronic check users.

We defined high-value customers using a conservative and defensible lifetime-value lens: customers with high cumulative spend, established tenure, and annual/multi-year contracts. Under this definition, high-value accounts represent **22.8%** of the base, and only **1.1%** of that segment are currently predicted at high churn risk. This is good news: the highest-value core is generally stable, but there is a small, actionable at-risk subset where targeted retention can protect disproportionate revenue.

## What We Found (Drivers of Churn)
### 1) Contract model is the clearest lever
Customers on month-to-month contracts churn at much higher rates than those on one- or two-year terms. In both model families, contract-related variables rank among the most important predictors. This confirms that commitment structure itself influences churn behavior.

### 2) Tenure acts as a “stability moat”
Tenure is the top or near-top signal in both models. Newer customers are much more vulnerable, while long-tenure customers churn far less. This suggests onboarding and early-life experience remain critical windows for churn prevention.

### 3) Price pressure matters
Higher monthly bills are associated with higher churn risk. This does not mean “high spend equals bad customer”; rather, customers paying more appear more sensitive to perceived value gaps, billing friction, or competitive alternatives.

### 4) Value-add services reduce exits
Customers without online security and/or tech support are more likely to churn. This is a practical intervention point: service bundles can improve retention if offered to the right users before cancellation intent solidifies.

### 5) Payment behavior is an operational signal
Electronic check users show materially higher churn than customers on automatic bank transfer or card payments. Payment method may reflect engagement and convenience, and it is a tractable area for rapid intervention.

## What We Recommend
### A. Targeting strategy
Use a two-step prioritization for monthly retention operations:
1. **Segment gate:** high-value customers first (selected definition below).
2. **Risk ranking:** within high-value customers, sort by predicted churn probability descending.

**Selected high-value definition:**
- TotalCharges >= 3000
- Tenure >= 24 months
- Contract = One year or Two year

This definition aligns with long-term revenue and relationship quality, and avoids conflating temporary high bills with durable value.

### B. Incentive playbook (next month)
Deploy 3–4 offers tied directly to model-identified churn drivers:

1. **Renewal protection offer:** price-lock + loyalty credit for contract extension.
2. **Bill optimization intervention:** personalized plan review and right-sized package for high-charge, high-risk users.
3. **Support/security uplift:** 90-day premium support + online security trial for customers lacking these services.
4. **Auto-pay migration incentive:** recurring bill credit for switching from electronic check to automatic payment.

### C. Threshold and trade-off policy
Retain threshold at **0.30** for month-one rollout. It captures a high share of likely churners (high recall), suitable for proactive save programs. If contact capacity becomes constrained, move threshold upward in a controlled A/B pilot.

## Expected Impact and How to Operationalize Next Month
### Expected impact
- **Near-term:** Better prioritization of save offers to likely churners instead of broad, low-yield outreach.
- **Revenue protection:** Focus on high-value at-risk accounts protects disproportionate lifetime value.
- **Program efficiency:** Incentives linked to specific churn drivers should improve acceptance and reduce discount leakage.

### 30-day operational plan
**Week 1:**
- Finalize target list using model score + high-value flag.
- Align retention scripts, offers, and channels with the four incentive tracks.

**Week 2:**
- Launch campaign to high-risk cohorts.
- Route highest-risk/high-value accounts to specialist retention queue.

**Week 3:**
- Monitor early KPIs: contact rate, offer acceptance, churn-save rate, and net revenue impact.
- Start threshold sensitivity check (0.30 vs 0.35) in a limited split test.

**Week 4:**
- Report outcomes to leadership.
- Calibrate offer mix and targeting rules for month-two deployment.

### KPI dashboard to track monthly
- Churn rate (overall and by segment)
- Save rate among contacted high-risk customers
- Retention campaign ROI (incremental margin protected)
- High-value segment churn and migration trends
- Offer acceptance by incentive type

In summary, the model is production-ready for decision support, the high-value base is mostly stable, and a focused, driver-specific retention program can be implemented immediately with measurable business impact next month.
