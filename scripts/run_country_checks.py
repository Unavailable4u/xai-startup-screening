"""Pooled vs. within-country vs. leave-one-country-out checks (run after run_audit)."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from xai_capital import config as C
from xai_capital.data import load_startups, labelled
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof, summarize

REP = int(os.environ.get("XAI_REPEATS", 3))
MODELS = ("logit_l2", "rf", "xgb")
df = load_startups()
tgt = C.TARGETS["primary"]
d, y = labelled(df, tgt)
rows = []

def cv(name, num, cat, data, yy):
    for mn, mod in make_models(num, cat).items():
        if mn in MODELS:
            s = summarize(repeated_cv_oof(mod, data[num + cat], yy, 5, REP), yy)
            rows.append({"analysis": name, "model": mn, "n": len(yy), "n_unfunded": int((yy == 0).sum()),
                         "auc": round(s["auc_mean"], 3), "lo": round(s["auc_boot_lo"], 3),
                         "hi": round(s["auc_boot_hi"], 3)})

cv("pooled SetA", C.NUM_A, C.CAT_A, d, y)
cv("pooled SetA + country dummies", C.NUM_A + C.COUNTRY_DUMMIES, C.CAT_A, d, y)
cv("pooled CORE", C.CORE, C.CAT_A, d, y)
cv("pooled CORE without age_years", [c for c in C.CORE if c != "age_years"], C.CAT_A, d, y)
cv("age_years only", ["age_years"], [], d, y)
cv("country dummies only", C.COUNTRY_DUMMIES, [], d, y)
for ctry in ["Bangladesh", "India", "Pakistan"]:
    sub = d[d.country == ctry].reset_index(drop=True)
    cv(f"{ctry} only, CORE", C.CORE, C.CAT_A, sub, sub[tgt].astype(int))
for held in ["Bangladesh", "India", "Pakistan"]:       # leave-one-country-out
    tr, te = d[d.country != held], d[d.country == held]
    for mn, mod in make_models(C.CORE, C.CAT_A).items():
        if mn in MODELS:
            p = clone(mod).fit(tr[C.CORE + C.CAT_A], tr[tgt].astype(int)).predict_proba(te[C.CORE + C.CAT_A])[:, 1]
            rows.append({"analysis": f"LOCO test={held} (CORE)", "model": mn, "n": len(te),
                         "n_unfunded": int((te[tgt] == 0).sum()), "auc": round(roc_auc_score(te[tgt].astype(int), p), 3)})
res = pd.DataFrame(rows)
C.TABLES.mkdir(parents=True, exist_ok=True)
res.to_csv(C.TABLES / "country_checks.csv", index=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(res.to_string())
