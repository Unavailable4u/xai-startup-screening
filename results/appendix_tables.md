**Appendix Table 6. Leave-one-country-out AUC by held-out country [95% CI]**

| Features | Model | Bangladesh | India | Pakistan |
|---|---|---|---|---|
| Neutral | Logistic | 0.58 [0.45, 0.71] | 0.56 [0.40, 0.71] | 0.69 [0.49, 0.86] |
| Neutral | Random Forest | 0.64 [0.52, 0.77] | 0.57 [0.40, 0.72] | 0.72 [0.53, 0.89] |
| Neutral | XGBoost | 0.61 [0.48, 0.74] | 0.44 [0.27, 0.60] | 0.73 [0.54, 0.88] |
| Neutral + sector | Logistic | 0.61 [0.45, 0.75] | 0.88 [0.78, 0.96] | 0.93 [0.84, 0.99] |
| Neutral + sector | Random Forest | 0.63 [0.47, 0.77] | 0.84 [0.73, 0.94] | 0.98 [0.94, 1.00] |
| Neutral + sector | XGBoost | 0.59 [0.44, 0.74] | 0.81 [0.69, 0.92] | 0.98 [0.94, 1.00] |
| Sector only | Logistic | 0.62 [0.47, 0.75] | 0.95 [0.89, 0.99] | 0.97 [0.91, 1.00] |
| Sector only | Random Forest | 0.63 [0.48, 0.77] | 0.89 [0.80, 0.97] | 0.97 [0.91, 1.00] |
| Sector only | XGBoost | 0.63 [0.48, 0.77] | 0.93 [0.85, 0.98] | 0.98 [0.95, 1.00] |

*Note.* Models are trained on the other two countries. Intervals are percentile bootstrap intervals over the held-out firms only. Unfunded firms in the held-out country: Bangladesh 23, India 20, Pakistan 16.

**Appendix Table 8. Within-country cross-validated AUC [95% CI]**

| Features | Model | Bangladesh | India | Pakistan |
|---|---|---|---|---|
| Neutral | Logistic | 0.57 [0.42, 0.71] | 0.62 [0.48, 0.77] | 0.52 [0.32, 0.73] |
| Neutral | Random Forest | 0.66 [0.53, 0.78] | 0.58 [0.43, 0.72] | 0.69 [0.52, 0.85] |
| Neutral | XGBoost | 0.67 [0.54, 0.80] | 0.56 [0.41, 0.70] | 0.67 [0.49, 0.83] |
| Neutral + sector | Logistic | 0.56 [0.39, 0.71] | 0.94 [0.87, 0.98] | 0.96 [0.89, 0.99] |
| Neutral + sector | Random Forest | 0.67 [0.54, 0.80] | 0.93 [0.86, 0.98] | 0.95 [0.88, 0.99] |
| Neutral + sector | XGBoost | 0.71 [0.58, 0.84] | 0.88 [0.77, 0.96] | 0.94 [0.86, 0.99] |
| Sector only | Logistic | 0.53 [0.36, 0.69] | 0.92 [0.84, 0.98] | 0.97 [0.92, 1.00] |
| Sector only | Random Forest | 0.56 [0.41, 0.72] | 0.94 [0.87, 0.99] | 0.95 [0.89, 0.99] |
| Sector only | XGBoost | 0.61 [0.46, 0.76] | 0.89 [0.79, 0.96] | 0.94 [0.88, 0.99] |

*Note.* 5-fold cross-validation repeated 20 times within each country. Intervals are bootstrap intervals over firms; repeated folds are not independent.

**Appendix Table 9. Pooled AUC in the primary sample and in all firms with a funding label [95% CI]**

| Features | Model | Primary sample (n = 188) | All labelled firms (n = 211) |
|---|---|---|---|
| Neutral | Logistic | 0.58 [0.48, 0.66] | 0.65 [0.57, 0.73] |
| Neutral | Random Forest | 0.65 [0.56, 0.73] | 0.70 [0.63, 0.77] |
| Neutral | XGBoost | 0.62 [0.53, 0.72] | 0.67 [0.60, 0.75] |
| Neutral + sector | Logistic | 0.83 [0.76, 0.90] | 0.84 [0.78, 0.90] |
| Neutral + sector | Random Forest | 0.85 [0.79, 0.90] | 0.84 [0.79, 0.90] |
| Neutral + sector | XGBoost | 0.84 [0.78, 0.90] | 0.83 [0.77, 0.89] |
| Sector only | Logistic | 0.85 [0.79, 0.91] | 0.84 [0.78, 0.90] |
| Sector only | Random Forest | 0.84 [0.78, 0.90] | 0.83 [0.77, 0.89] |
| Sector only | XGBoost | 0.86 [0.79, 0.91] | 0.84 [0.77, 0.89] |

*Note.* The primary sample has 188 firms (founded 2010-2023, independent). The sensitivity sample has all 211 firms with a binary funding label. Pooled estimates include country indicators.
