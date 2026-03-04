# High Value Customer Definition

## Proposed function (selected)

```python
def is_high_value_customer(row: dict) -> bool:
    return (
        row["TotalCharges"] >= 3000
        and row["tenure"] >= 24
        and row["Contract"] in {"One year", "Two year"}
    )
```

### Why this is defensible
This rule identifies customers with:
1. **Large cumulative revenue** (`TotalCharges >= 3000`),
2. **Proven relationship longevity** (`tenure >= 24`), and
3. **Contract commitment** (annual or multi-year contract), which is typically aligned with higher lifetime value and lower servicing volatility.

This captures economically meaningful, stable customers that are expensive to lose.

---

## Alternative definitions considered

### Alternative A (selected): Lifetime-value stability rule
- `TotalCharges >= 3000`
- `tenure >= 24`
- `Contract in {One year, Two year}`

**Coverage and risk estimate:**
- High value customers: **22.8%** of base (1,605 of 7,043)
- Predicted high-risk among high value (threshold = 0.30): **1.1%** (17 of 1,605)

### Alternative B: Current ARPU + premium service intensity
- `MonthlyCharges >= 75`
- `tenure >= 12`
- `InternetService == Fiber optic`

**Coverage and risk estimate:**
- High value customers: **29.7%**
- Predicted high-risk among high value: **47.4%**

### Alternative C: Ultra-high spend long-tenure segment
- `TotalCharges >= 5000`
- `tenure >= 36`

**Coverage and risk estimate:**
- High value customers: **16.1%**
- Predicted high-risk among high value: **11.8%**

---

## Final choice
We select **Alternative A** for management reporting because it aligns best with classical CLV logic (revenue depth + tenure + commitment) and gives a clear, conservative target segment for retention protection.

For campaign design, Alternative B can be used as an **“expansion risk” segment** because it surfaces more potentially churning, high-bill users.
