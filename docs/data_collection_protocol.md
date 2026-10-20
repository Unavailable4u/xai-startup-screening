# Data collection protocol (v1)

1. Population / sampling frame: (define; e.g. startups founded 2012-2022 listed in X)
2. Random draw: seed = ____, date = ____
3. Required fields per row: see `Add_Rejected_Template` columns
4. Rule for any_external_funding = 0: no funding/grant/accelerator support in any listed source AND >= 36 months old at baseline. Otherwise leave blank.
5. Never guess; leave blank if not found. Every row has source_citation + date_collected.
6. Second coder re-codes a random 20% blind; report Cohen's kappa; resolve disagreements.
7. Version: bump data tag (data-v2, ...) and record MD5 in Version_Log.

## Addendum 2026-10-20: quotas for the primary frame (independent startups founded 2010-2023)
Current unfunded in frame: BD 9, IN 5, PK 4 (total 18). Targets before freezing the analysis plan:

| Country | Unfunded now | Minimum target | Notes |
|---|---|---|---|
| Bangladesh | 9 | 15 | Prioritise independent startups, not subsidiaries |
| India | 5 | 15 | Sample from a listed frame, not famous cases |
| Pakistan | 4 | 12 | Smaller ecosystem; accept a lower quota |
| Total | 18 | 42+ | A rough power guide: with 40 unfunded and 4x as many funded, a true AUC of 0.70 gives a 95% CI half-width of about 0.08 |

Also collect founder attributes (education, STEM, serial entrepreneur, industry experience) for ALL new rows; they are the variables H1 and H3 depend on.
Match funded and unfunded rows on founding-year range (both 2010-2023) so age is not a sampling artefact.
