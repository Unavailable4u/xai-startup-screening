# v2 data patch (6 Oct 2026 workbook: Startup_Data_MODEL_READY_EXPANDED_v2.xlsx)

Copy these files into your repo, overwriting the old ones:

    src/xai_capital/data.py        (rebuilds macro columns by country-year; guards against uncached formulas; adds country)
    src/xai_capital/config.py      (points to the v2 recalculated workbook; adds CORE set and country dummies)
    scripts/run_country_checks.py  (pooled vs within-country vs leave-one-country-out)
    data/raw/Startup_Data_MODEL_READY_EXPANDED_v2_recalc.xlsx

Why a "recalc" workbook: the v2 file was saved without calculated values for its formula cells,
so pandas read age_years, n_founders and female_founder_share as empty for all Bangladesh rows,
and the Bangladesh macro formulas broke when Macro_Lookup was restructured. The recalculated
copy has the values filled in; data.py rebuilds the macro columns from the long-format lookup.
