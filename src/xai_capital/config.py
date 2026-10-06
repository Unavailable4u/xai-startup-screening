"""Central configuration: paths, column groups, random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
MODEL_READY = RAW / "Startup_Data_MODEL_READY_BD-ST-001_to_100.xlsx"
SEED = 42

TARGETS = {
    "primary": "any_external_funding",          # 72 yes / 14 no (86 usable)
    "secondary": "follow_on_funding_received",  # 70 yes / 11 no (81 usable)
}

# Set A: attributes knowable at/near founding (cleanest for "screening")
NUM_A = ["edu_max_level", "any_stem_founder", "serial_entrepreneur",
         "industry_exp_max_years", "n_founders", "female_founder_share",
         "product_stage", "age_years", "gdp_growth_at_founding",
         "inflation_at_founding", "foreign_hq_flag",
         "is_b2b", "is_b2c", "is_marketplace", "is_subscription_saas",
         "is_hardware_iot", "is_transaction_commission"]
CAT_A = ["sector_group"]

# Set B: A + ecosystem signals that may overlap with funding itself
EXTRA_B = ["accelerator_participated", "prior_grant_flag"]

# Set C (diagnostic only, never headline): current-snapshot variables
EXTRA_C = ["team_size_mid", "traction_composite"]

# Columns that must never enter a funding-prediction model
FORBIDDEN = ["follow_on_funding_usd", "follow_on_year", "funding_amount_basis",
             "n_accepted_investment_rows", "has_vc_pe", "has_angel",
             "has_accelerator_investor", "has_corporate", "has_government",
             "has_grant_dfi", "gdp_growth_at_funding", "inflation_at_funding",
             "evaluation_score_1to5", "survival_status", "event_closed",
             "acquired_flag", "duration_months", "any_external_funding",
             "follow_on_funding_received"]
