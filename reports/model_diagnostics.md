# Model Diagnostics Appendix

## Model comparison

| Model | Train ROC-AUC | Test ROC-AUC | Assessment |
| --- | ---: | ---: | --- |
| Logistic Regression | 0.849 | 0.842 | Best holdout performance; selected |
| Random Forest | 1.000 | 0.820 | Strong train performance but clear overfitting |

Logistic Regression was selected because it generalized better to unseen data and offers more interpretable coefficients. The Random Forest remains useful as a nonlinear benchmark, but its train–test gap is too large to prefer for this analysis.

## Selected-model operating point

The probability threshold was selected by maximizing F1 across values from 0.20 to 0.80 on the holdout set.

- Threshold: **0.28**
- Accuracy: **0.749**
- Precision: **0.518**
- Recall: **0.781**
- ROC-AUC: **0.842**
- F1: **0.623**
- Confusion matrix: TN = 763, FP = 272, FN = 82, TP = 292

The threshold intentionally favors recall because missing a likely churner can be more costly than reviewing a false positive. A production threshold should ultimately be selected from contact capacity, intervention cost, customer value, and expected save rate.

## Main signals

The models and descriptive analysis consistently highlight:

1. Tenure and contract structure
2. Monthly and cumulative charges
3. Internet-service type
4. Online security and technical support
5. Electronic-check payments

Coefficient signs in the multivariate Logistic Regression should not be read as isolated causal effects. Correlated billing and service variables can change coefficient direction after controlling for the rest of the feature set.

## Practical interpretation

The model is suitable for prioritizing a retention queue in a portfolio setting. Production use would require out-of-time validation, cost-based threshold selection, probability calibration, drift monitoring, bias checks, and measurement of campaign uplift.

