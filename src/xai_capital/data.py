import numpy as np
import pandas as pd
from .config import MODEL_READY

COUNTRY = {"BD": "Bangladesh", "IN": "India", "PK": "Pakistan"}

def load_startups(only_independent: bool = False, only_country: str | None = None) -> pd.DataFrame:
    df = pd.read_excel(MODEL_READY, sheet_name="Model_Ready_Startups")
    # Guard: formula cells saved without cached values read as NaN in pandas.
    if df["age_years"].isna().mean() > 0.3:
        raise RuntimeError("age_years is mostly empty: the workbook was saved without "
                           "calculated values. Use the *_recalc.xlsx file (see docs).")
    df["country"] = df["startup_id"].str[:2].map(COUNTRY)
    # Macro columns are rebuilt from the long-format Macro_Lookup (country, year).
    # The workbook formulas still point at the old Bangladesh-only layout.
    ml = pd.read_excel(MODEL_READY, sheet_name="Macro_Lookup").set_index(["country", "year"])
    for src, dst in (("gdp_growth_pct", "gdp_growth_at_founding"),
                     ("inflation_cpi_pct", "inflation_at_founding")):
        df[dst] = [ml[src].get((c, int(y)), np.nan) if pd.notna(y) else np.nan
                   for c, y in zip(df["country"], df["founding_year"])]
    df["is_IN"] = (df["country"] == "India").astype(int)
    df["is_PK"] = (df["country"] == "Pakistan").astype(int)
    if only_independent:
        df = df[df["entity_type"].str.startswith("Independent")]
    if only_country:
        df = df[df["country"] == only_country]
    return df.reset_index(drop=True)

def labelled(df: pd.DataFrame, target: str):
    d = df[df[target].notna()].copy()
    return d, d[target].astype(int)


def in_frame(df: pd.DataFrame, lo: int = 2010, hi: int = 2023) -> pd.DataFrame:
    """Primary analysis frame (decision D9): independent startups founded lo..hi."""
    keep = df["entity_type"].str.startswith("Independent") & df["founding_year"].between(lo, hi)
    return df[keep].reset_index(drop=True)

def boot_auc(y, p, n_boot=1000, seed=42):
    """AUC with percentile bootstrap CI (resampling startups)."""
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            b.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(b, 2.5), np.percentile(b, 97.5)


def paired_boot_diff(y, p1, p2, n_boot=1000, seed=42):
    """AUC(p1) - AUC(p2) with paired percentile bootstrap CI (same resampled startups)."""
    from sklearn.metrics import roc_auc_score
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            diffs.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return (roc_auc_score(y, p1) - roc_auc_score(y, p2), *np.percentile(diffs, [2.5, 97.5]))
