# Data collection protocol: addendum 2 (2026-10-21)

Quotas for the v3 frame are met (unfunded: BD 23, IN 20, PK 16). The remaining weaknesses are about HOW cases and controls were sampled. Priorities:

## 1. Founder attributes for funded India and Pakistan rows (highest priority)
Founder data is observed for 4 of 42 funded Indian and 1 of 25 funded Pakistani frame rows (about 63 rows missing). Collect for every funded IN/PK frame row: edu_max_level, any_stem_founder, serial_entrepreneur, industry_exp_max_years, using the same coding rules as the unfunded rows. Record the source for each.

## 2. Funded startups sampled from the same listings as the unfunded ones
Current frame counts: IT/SaaS 26 unfunded vs 9 funded; e-commerce 2 vs 31; fintech 2 vs 23. Fill the thin cells by drawing from the SAME directory listings (Tracxn, BASIS, Future Startup, etc.), in listed order, with skip logs, for both funded and unfunded startups:

| Sector group | Funded now | Unfunded now | Collect to reach (each) |
|---|---|---|---|
| IT / SaaS / AI / IoT | 9 | 26 | funded >= 20 |
| E-commerce | 31 | 2 | unfunded >= 10 |
| Fintech | 23 | 2 | unfunded >= 10 |
| HealthTech, EdTech | 31 | 11 | unfunded >= 12 |

Rule: for every listing, record how many entries were skipped and why (founders undocumented, outside frame, funding unclear). Do NOT skip entries because founder data is hard to find; mark those fields as missing.

## 3. Coding reminders
- Funded/unfunded label: use the two-part check from v3 (directory profile + targeted funding-news search); record the evidence.
- Keep debt (bank loans) out of the funding definition, as decided in v3; flag these rows.
- Closures: also search for closures among UNFUNDED startups, so closure is not found only among funded firms.

## 4. Audit
Run `python scripts/make_audit_sample.py`, give docs/audit/audit_sample.csv to a second coder, then `python scripts/summarize_audit.py`.
