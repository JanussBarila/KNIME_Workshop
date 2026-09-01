# High-Value Customer Definition

## Selected rule

```python
def is_high_value_customer(row: dict) -> bool:
    return (
        row["TotalCharges"] >= 3000
        and row["tenure"] >= 24
        and row["Contract"] in {"One year", "Two year"}
    )
```

The rule combines cumulative revenue, established tenure, and contract commitment. It is intentionally conservative and easy for commercial teams to explain and reproduce.

## Coverage and risk

- High-value customers: **1,605 of 7,043 (22.8%)**
- Selected churn threshold: **0.28**
- High-value customers flagged at risk: **59 of 1,605 (3.7%)**

This segment should be used as a prioritization gate rather than a complete customer-lifetime-value model. A production definition would also include contribution margin, service costs, discounts, payment behavior, and expected future revenue.

## Alternatives considered

- Current high monthly charges plus premium-service use
- Very high cumulative spend and long tenure
- A continuous CLV score rather than a binary rule

The selected rule was preferred for this project because it balances business relevance with transparency.

