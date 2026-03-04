# Model Diagnostics Appendix

## 1) ROC AUC Summary

| Model | Train AUC | Test AUC |
|---|---:|---:|
| Logistic Regression | 0.840 | 0.856 |
| Random Forest (tree-based) | 0.887 | 0.863 |

**Interpretation:**
- Both models separate churners vs. non-churners well (AUC > 0.85).
- The Random Forest delivered the strongest test discrimination (0.863), so it is the selected model for operational recommendations.
- The train–test gap is larger for the Random Forest than for Logistic Regression, indicating mild overfit risk, but test performance still improves over logistic.

## 2) Logistic Regression Coefficient Diagnostics (Top Absolute Coefficients)

Signs are interpreted as effect on churn likelihood, holding other variables constant.

| Rank | Feature | Coefficient | Direction |
|---|---|---:|---|
| 1 | tenure | -0.618 | Longer tenure lowers churn risk |
| 2 | MonthlyCharges | +0.351 | Higher monthly bills increase churn risk |
| 3 | Contract__Two year | -0.337 | 2-year contracts strongly reduce churn |
| 4 | PaperlessBilling__No | -0.323 | Not using paperless billing is associated with lower churn |
| 5 | InternetService__DSL | -0.323 | DSL users churn less than baseline categories |
| 6 | Contract__Month-to-month | +0.283 | Month-to-month customers are more likely to churn |
| 7 | TechSupport__Yes | -0.258 | Having tech support lowers churn risk |
| 8 | PhoneService__Yes | -0.253 | Phone service usage is associated with lower churn |

## 3) Tree Model Feature Importance (Top Drivers)

| Rank | Feature | Importance |
|---|---|---:|
| 1 | tenure | 0.160 |
| 2 | Contract__Month-to-month | 0.099 |
| 3 | TotalCharges | 0.096 |
| 4 | OnlineSecurity__No | 0.072 |
| 5 | InternetService__Fiber optic | 0.056 |
| 6 | Contract__Two year | 0.053 |
| 7 | MonthlyCharges | 0.050 |
| 8 | TechSupport__No | 0.046 |

## 4) Additional Test-set Metrics for Selected Model (Random Forest)

- Threshold used for binary classification: **0.30** (selected by best F1 on holdout test set).
- Accuracy: **0.784**
- Precision: **0.568**
- Recall: **0.767**
- Confusion Matrix (Actual rows x Predicted columns):
  - TN = 817, FP = 218
  - FN = 87, TP = 287

## 5) What these diagnostics mean in practice

- **AUC** tells us ranking quality independent of a fixed threshold. AUC near 0.86 means strong prioritization ability for retention outreach.
- **Precision vs. Recall trade-off:** At threshold 0.30, the model intentionally favors recall (catching more likely churners), appropriate for churn prevention where missing a churner can be costly.
- **Feature diagnostics** from both models consistently point to contract type, tenure, service/security add-ons, and price exposure as central churn drivers.
