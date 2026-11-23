"""Regression test for the SHAP filter (review item 7): the missing-value criterion must be able to fire."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from xai_capital.models import make_models
from xai_capital.simulate import _xai_filter, PRESTIGE


def _data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"edu_max_level": rng.integers(0, 3, n).astype(float),
                      "serial_entrepreneur": rng.integers(0, 2, n).astype(float),
                      "age_years": rng.normal(6, 2, n), "n_founders": rng.integers(1, 5, n).astype(float)})
    y = pd.Series((X["n_founders"] + rng.normal(0, 1, n) > 2.5).astype(int))
    return X, y


def test_filter_allows_clean_rows_and_flags_nan_attribution():
    X, y = _data()
    num = list(X.columns)
    model = make_models(num, [])["xgb"].fit(X.iloc[10:], y.iloc[10:])
    clean = X.iloc[:10].copy()
    assert _xai_filter(model, clean, 0.99).all()                 # nothing dominated at a lax threshold
    nan = clean.copy(); nan.loc[nan.index[:5], "n_founders"] = np.nan
    allowed = _xai_filter(model, nan, 0.30)
    assert allowed.sum() < _xai_filter(model, clean, 0.30).sum() # NaN-attributed mass now lowers eligibility


def test_prestige_columns_are_named_as_in_the_plan():
    assert PRESTIGE == ["edu_max_level", "serial_entrepreneur"]
