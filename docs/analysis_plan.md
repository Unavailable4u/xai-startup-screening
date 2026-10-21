# Analysis plan v1.1 (DRAFT: freeze only when Section 11 is satisfied)

Project: XAI startup screening and simulated capital allocation, South Asia (Bangladesh, India, Pakistan)
Data: Startup_Data_MODEL_READY_EXPANDED_v3 (recalculated copy `..._v3_recalc.xlsx`), 225 startups
Decisions: D9 pooled + country control, within-country, leave-one-country-out as the main generalisation test;
D10 sector/business-model features reported as a SEPARATE feature group; D11 complete-case rule for founder-attribute hypotheses (H1, H3, H4).

## 1. Hypotheses
- H1: founder human-capital attributes (education, STEM, industry experience) predict external funding.
- H2: non-linear ML (RF, XGBoost) outperforms regularised logistic regression at this sample size.
- H3: prestige signals (education level, serial-entrepreneur status) retain little additive explanatory power after controlling for fundamentals.
- H4: a SHAP-filtered rule-based screening policy achieves a higher hit rate than heuristic, econometric and black-box policies.
- H5 (exploratory): accelerator participation is associated with funding after covariate adjustment.

## 2. Sample and frame
- Primary frame: independent startups (entity_type starts with "Independent") founded 2010-2023 with a non-missing funding label.
- Current counts (2026-10-21): frame = 188 labelled, 59 unfunded (Bangladesh 23, India 20, Pakistan 16). All rows = 225 (211 labelled).
- Rows enter the final analysis only after the human audit (Section 10).
- **Known design features (to be stated in the paper as limitations):**
  1. Case/control sampling: unfunded startups were drawn from directory listings (Tracxn, BASIS, etc.) and funded startups mostly from documented well-known cases; the two groups come from different sampling sources.
  2. Sector is confounded with that sampling source (IT/SaaS 74% unfunded; e-commerce 6%; fintech 8%).
  3. Informative missingness: founder attributes were required for unfunded rows but are missing for most funded India/Pakistan rows (Bangladesh: 87% observed for unfunded vs 50% for funded).
  4. Closures were purposively searched, mostly among funded Indian and Pakistani firms (27 of 31 closures).
  5. Macro variables at founding act partly as a calendar-year proxy.

## 3. Outcomes
- Primary: any_external_funding (0 = no disclosed funding found after >= 36 months; not an audited fact).
- Secondary: follow_on_funding_received.
- Survival: descriptive only (see Section 7).

## 4. Feature groups (D10)
| Group | Contents |
|---|---|
| neutral | age_years, product_stage, n_founders, gdp_growth_at_founding, inflation_at_founding |
| neutral+sector | neutral + is_b2b, is_b2c, is_marketplace, is_subscription_saas, is_hardware_iot, is_transaction_commission, sector_group |
| sector_only | the business-model flags + sector_group |
| founder (D11) | edu_max_level, any_stem_founder, serial_entrepreneur, industry_exp_max_years; Bangladesh complete cases only |
Pooled models add country dummies (is_IN, is_PK). Every column in config.FORBIDDEN is excluded; Set C variables are diagnostic only. Results are ALWAYS reported for neutral and neutral+sector, so readers can see how much sector drives performance.

## 5. Analyses
1. `run_main_analysis.py`: pooled + country control (CV stratified by country x label; AUC overall and per country), within-country CV (only if >= 8 unfunded), leave-one-country-out. Models: logistic L2, RF, XGBoost; imputation inside folds; 20 repeats in the final run. Metrics: AUC (bootstrap 95% CI), PR-AUC for unfunded, Brier. No accuracy.
2. H2: paired bootstrap of AUC(RF or XGBoost) - AUC(logistic) on pooled out-of-fold predictions.
3. `run_founder_hypotheses.py` (H1, H3): Bangladesh frame rows with all four founder attributes observed. Reports the selection diagnostic, CV AUC for controls only / founder only / both, paired differences, and standardised odds ratios with bootstrap CIs.
4. `run_shap.py`: SHAP for XGBoost on (a) pooled neutral, (b) pooled neutral+sector, (c) Bangladesh founder rows; rank of prestige features.
5. `run_sim.py` (H4): four investors + random baseline, Bangladesh founder-complete rows only (prestige needs founder data); k_frac = 0.30, tau = 0.40 fixed in advance. Exploratory because n is about 53.
6. `run_survival.py`: Kaplan-Meier by country (descriptive); stratified Cox with at most 2 covariates (exploratory).
7. H5: AIPW (`run_aipw.py`) only if accelerator_participated is observed for >= 100 labelled frame startups and propensity scores are not at the clipping bounds. Otherwise omit.

## 6. Interpretation rules (fixed before final runs)
- Predictive value: the held-out-country AUC lower 95% bound must exceed 0.5 in at least two of three countries, in the **neutral** group, to claim generalisable screening ability. Results that depend on sector are reported as "sector-driven, confounded with sampling".
- H1 supported if founder attributes raise AUC over controls by >= 0.05 with a paired CI excluding 0.
- H2 supported if RF or XGBoost beats logistic by >= 0.03 AUC with a paired CI excluding 0.
- H3 supported if adding prestige features to fundamentals changes AUC by < 0.02 AND both prestige features rank in the bottom half of SHAP importance.
- H4 supported if the XAI policy's hit rate exceeds black-box and heuristic with a paired CI excluding 0. Folds are correlated across repeats, so CIs are optimistic; say so.
- A null result is reported as a null result.

## 7. Survival (descriptive)
Closures were searched among funded firms in India and Pakistan, so closure rates are not comparable across countries or between funded and unfunded firms (a funded-vs-unfunded hazard ratio is confounded by this design and by immortal-time bias). Report KM curves and counts only, with that caveat; do not interpret Cox hazard ratios causally.

## 8. Sensitivity analyses
ALL ROWS sample; each country dropped in turn; neutral without age_years; excluding rows that fail the audit; secondary outcome; temporal split (train founded <= 2016, test >= 2017) if enough unfunded cases; funded rows added from the same directories (Section 11, item 3) as a replacement sample.

## 9. Reporting
Every table and figure is generated by scripts into results/. Report all three feature groups, all three countries, and the "not estimable" cells explicitly.

## 10. Data audit (required before freeze)
- `python scripts/make_audit_sample.py` writes docs/audit/audit_sample.csv: all unfunded rows, all closures, and a random 20% of other added rows (seed 42).
- A second coder (not the compiler) opens each source_citation and fills the blank columns. `python scripts/summarize_audit.py` reports the error rate (Wilson 95% CI). If it exceeds 10%, audit all added rows.
- The paper states how rows were compiled, including any automated assistance, and that "unfunded" means no disclosed funding found.

## 11. Freeze conditions
- [ ] Audit complete; corrections applied; error rate documented.
- [ ] Founder attributes (education, STEM, serial entrepreneur, industry experience) collected for the funded India and Pakistan frame rows (about 63 rows), so H1, H3 and H4 can be tested beyond Bangladesh.
- [ ] Funded startups sampled from the SAME directory listings as the unfunded ones, filling the sector x funded cells (see protocol addendum 2).
- [ ] Co-authors have signed off on Sections 1-9.

## Appendix: baseline smoke test (v3 data, 1-3 CV repeats; NOT for citation)
- Pooled AUC: neutral 0.59-0.65; neutral+sector 0.82-0.84; sector only 0.84-0.86.
- Leave-one-country-out, neutral: Bangladesh 0.58-0.64, India 0.44-0.57, Pakistan 0.70-0.73. With sector: India 0.81-0.88, Pakistan 0.93-0.98.
- H2: RF - logistic = +0.02 to +0.04, XGBoost - logistic = 0.00 to +0.03; all paired CIs include 0.
- Bangladesh founder rows (53, 20 unfunded): founder attributes add +0.10 to +0.14 AUC over controls (CIs include 0); prestige adds < 0.01 over fundamentals; education odds ratio 2.19 per SD (95% CI 1.34-3.80); STEM founder 0.32 (the unfunded set has 100% STEM founders, consistent with IT-agency sampling).
- SHAP (Bangladesh founder rows): education ranks 4 of 9, serial entrepreneur 7 of 9, so the SHAP part of the H3 rule is not met for education.
- Simulation (Bangladesh founder rows, 5 repeats): black-box = XAI = 0.87, heuristic 0.81, econometric 0.80, random 0.56.
- Survival (frame, 28 closures: BD 2, IN 16, PK 10): country differences are design artefacts; Cox funded HR of about 3 is not interpretable (see Section 7).
