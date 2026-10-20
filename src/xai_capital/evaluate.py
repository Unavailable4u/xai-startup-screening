import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from .config import SEED

def repeated_cv_oof(model, X, y, n_splits=5, n_repeats=20, strata=None):
    """Out-of-fold probabilities, one row per repeat."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=SEED)
    oof = np.full((n_repeats, len(y)), np.nan)
    split_on = y if strata is None else strata   # e.g. country x label
    for k, (tr, te) in enumerate(rskf.split(X, split_on)):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[k // n_splits, te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof

def summarize(oof, y, n_boot=1000):
    """Unfunded (class 0) is treated as the 'positive' class for PR-AUC."""
    y = np.asarray(y)
    aucs = [roc_auc_score(y, p) for p in oof]
    prs = [average_precision_score(1 - y, 1 - p) for p in oof]
    brs = [brier_score_loss(y, p) for p in oof]
    pm = oof.mean(axis=0)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], pm[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc_mean": np.mean(aucs), "auc_sd_over_repeats": np.std(aucs),
            "auc_boot_lo": lo, "auc_boot_hi": hi,
            "prauc_unfunded": np.mean(prs), "brier": np.mean(brs),
            "base_rate_unfunded": float(1 - y.mean())}
