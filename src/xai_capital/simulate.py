"""Four simulated investors choosing k startups from each held-out fold."""
import numpy as np, pandas as pd, shap
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from .config import SEED

PRESTIGE = ["edu_max_level", "serial_entrepreneur"]   # pre-registered 'signal' proxies

def _heuristic_score(X):
    return X[PRESTIGE].fillna(0).sum(axis=1).to_numpy() + 1e-6 * np.random.default_rng(SEED).random(len(X))

def _xai_filter(xgb_pipe, X_test, tau):
    """Flag candidates whose |SHAP| mass is dominated by prestige proxies or
    missing-value indicators. Threshold tau is chosen BEFORE seeing results."""
    pre, clf = xgb_pipe.named_steps["pre"], xgb_pipe.named_steps["clf"]
    Xt = pre.transform(X_test)
    names = [c.split("__", 1)[-1] for c in pre.get_feature_names_out()]
    sv = np.abs(shap.TreeExplainer(clf).shap_values(Xt))
    bad = [i for i, n in enumerate(names) if n in PRESTIGE or n.endswith("_missing")]
    share = sv[:, bad].sum(axis=1) / np.maximum(sv.sum(axis=1), 1e-12)
    return share <= tau          # True = allowed

def run(models, X, y, outcome, k_frac=0.3, tau=0.4, n_repeats=5):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    rows, rng = [], np.random.default_rng(SEED)
    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        k = max(1, int(round(k_frac * len(te))))
        logit = clone(models["logit_l2"]).fit(Xtr, y.iloc[tr])
        xgb = clone(models["xgb"]).fit(Xtr, y.iloc[tr])
        p_log = logit.predict_proba(Xte)[:, 1]
        p_xgb = xgb.predict_proba(Xte)[:, 1]
        allowed = _xai_filter(xgb, Xte, tau)
        p_xai = np.where(allowed, p_xgb, -1.0)
        picks = {"random": rng.permutation(len(te))[:k],
                 "heuristic": np.argsort(-_heuristic_score(Xte))[:k],
                 "econometric": np.argsort(-p_log)[:k],
                 "black_box": np.argsort(-p_xgb)[:k],
                 "xai": np.argsort(-p_xai)[:k]}
        oc = outcome.iloc[te].to_numpy()
        for name, idx in picks.items():
            rows.append({"fold": fold, "investor": name, "k": k,
                         "hit_rate": float(np.nanmean(oc[idx]))})
    return pd.DataFrame(rows)
