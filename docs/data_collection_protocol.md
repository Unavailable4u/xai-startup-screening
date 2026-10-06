# Data collection protocol (v1)

1. Population / sampling frame: (define; e.g. startups founded 2012-2022 listed in X)
2. Random draw: seed = ____, date = ____
3. Required fields per row: see `Add_Rejected_Template` columns
4. Rule for any_external_funding = 0: no funding/grant/accelerator support in any listed source AND >= 36 months old at baseline. Otherwise leave blank.
5. Never guess; leave blank if not found. Every row has source_citation + date_collected.
6. Second coder re-codes a random 20% blind; report Cohen's kappa; resolve disagreements.
7. Version: bump data tag (data-v2, ...) and record MD5 in Version_Log.
