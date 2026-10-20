# Analysis plan v1 (DRAFT: freeze only after the conditions in Section 11 are met)

Project: XAI startup screening and simulated capital allocation, South Asia (Bangladesh, India, Pakistan)
Data: Startup_Data_MODEL_READY_EXPANDED_v2 (recalculated copy), compiled 2026-10-19/20
Decision D9 (agreed): pooled analysis with country as a control; within-country results; held-out-country results as the MAIN test.

## 1. Hypotheses (from Study_Design sheet)
- H1: founder human-capital attributes (education, STEM, industry experience) predict external funding.
- H2: non-linear ML (RF, XGBoost) outperforms regularised logistic regression at this sample size.
- H3: prestige signals (education level, serial-entrepreneur status) add little explanatory power after controlling for fundamentals.
- H4: a SHAP-filtered rule-based screening policy achieves a higher hit rate than heuristic, econometric and black-box policies.
- H5 (exploratory): accelerator participation is associated with funding after covariate adjustment.
Expected signs and pass/fail criteria are written in Section 8 BEFORE final runs.

## 2. Sample and frame
- Primary frame: independent startups (entity_type starts with "Independent") founded 2010-2023, with a non-missing funding label.
- Sensitivity sample: all rows (includes 12 Indian/Pakistani firms founded before 2010, 11 of them unfunded, and 8 non-independent Bangladesh rows).
- Rows are used only after the human audit (Section 10) has confirmed their label and status.
- Current counts (2026-10-20, before audit): frame = 147 labelled, 18 unfunded (BD 9, IN 5, PK 4); all rows = 170 labelled, 34 unfunded.

## 3. Outcomes
- Primary: any_external_funding (1 = funded; 0 = no disclosed funding found after >= 36 months; "not an audited fact").
- Secondary: follow_on_funding_received.
- Descriptive/survival: event_closed, duration_months (Kaplan-Meier by country; Cox only with <= 3 covariates, stratified by country).

## 4. Features
- CORE (primary; low missingness in all three countries): age_years, product_stage, n_founders, gdp_growth_at_founding, inflation_at_founding, foreign_hq_flag, is_b2b, is_b2c, is_marketplace, is_subscription_saas, is_hardware_iot, is_transaction_commission, sector_group.
- CORE_no_age: same without age_years (age is the variable most exposed to sampling artefacts).
- Founder-attribute features (edu_max_level, any_stem_founder, serial_entrepreneur, industry_exp_max_years) are used ONLY where observed (mostly Bangladesh) and for H1/H3 within Bangladesh; they are 63-100% missing for India/Pakistan.
- Country dummies (is_IN, is_PK) are included in pooled models only.
- Never used: every column in config.FORBIDDEN; Set C (team_size_mid, traction_composite) is diagnostic only.

## 5. Analyses (script: scripts/run_main_analysis.py)
1. Pooled + country control: repeated stratified 5-fold CV, stratified by country x label; report AUC overall and within each country (bootstrap 95% CI).
2. Within-country CV: run only if a country has >= 8 unfunded startups in the sample; otherwise reported as "not estimable".
3. Leave-one-country-out (MAIN generalisation test): train on two countries, test on the third; AUC with bootstrap CI; flag as low-power if the held-out country has < 10 unfunded.
Models: logistic L2, Random Forest, XGBoost (fixed hyperparameters in models.py; imputation inside folds).
Metrics: AUC (primary), PR-AUC for the unfunded class, Brier score. Accuracy is not reported.
Repeats: 20 in final runs.

## 6. Explainability (H3, H4)
SHAP (TreeExplainer) on RF/XGBoost; global importance, beeswarm, 2-3 local examples; bootstrap stability of top-5 ranking. SHAP is descriptive of the model, not causal.

## 7. Portfolio simulation (H4) and accelerator analysis (H5)
- Simulation: k_frac = 0.30, tau = 0.40, fixed now; investors = heuristic, econometric, black-box, XAI, random; paired comparisons on identical folds; run within the primary frame.
- AIPW for accelerator effect: exploratory; requires overlap plot and balance table; reported only if accelerator_participated is observed for >= 100 labelled startups and propensity scores are not at the clipping bounds.

## 8. Pre-specified interpretation rules
- A model "has predictive value" only if the held-out-country AUC lower 95% bound exceeds 0.5 in at least two of the three countries.
- H2 is supported only if RF or XGBoost beats logistic by >= 0.03 AUC in the pooled analysis with a paired-bootstrap CI that excludes 0.
- H3 is supported if prestige features rank in the bottom half of SHAP importance AND removing them changes AUC by less than 0.02.
- If results are null, the paper reports them as such and frames the contribution as an audit of what observable signals can and cannot predict funding at this sample size.

## 9. Sensitivity analyses
ALL ROWS sample; CORE_no_age; excluding rows added after the audit sample if audit error rate > 10%; secondary outcome; temporal split (train founded <= 2016, test >= 2017) if enough unfunded cases; dropping each country in turn.

## 10. Data audit (required before freeze)
- A second coder (not the compiler) checks, against the cited source: all unfunded rows, all closures, and a random 20% of the remaining added rows.
- Record each check in Data_Notes (startup_id, field, old, new, source, who, date). If the error rate in the audit sample exceeds 10%, audit all added rows.
- State in the paper how rows were compiled, including any automated assistance, and that "unfunded" means no disclosed funding found.

## 11. Freeze conditions (do not tag analysis-plan-v1 until all are true)
- [ ] Audit complete and corrections applied.
- [ ] In-frame unfunded startups: >= 40 overall and >= 12 per country (see quotas in docs/data_collection_protocol.md).
- [ ] Founder attributes collected for the new rows, or the paper restricts H1/H3 to Bangladesh.
- [ ] Co-authors have signed off on Sections 2-9.

## Appendix: baseline smoke test (1 CV repeat, NOT for citation)
Primary frame, CORE features: pooled AUC 0.61-0.69 (95% CIs include 0.5 for logistic and XGBoost); within Bangladesh 0.48-0.64; leave-one-country-out AUC 0.23-0.49 (at or below chance). All-rows sample: pooled 0.77-0.80, held-out India 0.78-0.82, Pakistan 0.68-0.72, Bangladesh 0.58-0.66. The higher all-rows values are driven by pre-2010 unfunded firms, which fall outside the intended frame.
