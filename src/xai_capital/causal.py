"""Exploratory AIPW (doubly robust) estimate of an accelerator 'effect'."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from .config import SEED

def aipw_ate(X, t, y, n_boot=500, clip=(0.05, 0.95)):
    X = SimpleImputer(strategy="median").fit_transform(X)
    t, y = np.asarray(t, int), np.asarray(y, float)

    def _est(idx, seed):
        Xi, ti, yi = X[idx], t[idx], y[idx]
        e, m1, m0 = np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx))
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xi, ti):
            ps = LogisticRegression(C=0.5, max_iter=2000).fit(Xi[tr], ti[tr])
            e[te] = ps.predict_proba(Xi[te])[:, 1]
            for arm, out in ((1, m1), (0, m0)):
                trk = tr[ti[tr] == arm]
                rf = RandomForestClassifier(200, min_samples_leaf=3,
                                            random_state=seed).fit(Xi[trk], yi[trk])
                p = rf.predict_proba(Xi[te])
                out[te] = p[:, 1] if p.shape[1] == 2 else float(rf.classes_[0])
        e = np.clip(e, *clip)
        psi = (m1 - m0 + ti * (yi - m1) / e - (1 - ti) * (yi - m0) / (1 - e))
        return psi.mean(), e

    ate, e = _est(np.arange(len(y)), SEED)
    rng = np.random.default_rng(SEED)
    boots = []
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < t[i].sum() < len(i):
            boots.append(_est(i, SEED + b)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"ate": ate, "ci_lo": lo, "ci_hi": hi, "n": len(y),
            "n_treated": int(t.sum()), "ps_min": e.min(), "ps_max": e.max()}
