"""Main analysis for decision D9: pooled (country control) + within-country + leave-one-country-out."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.base import clone
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame, boot_auc
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof

REP = int(os.environ.get("XAI_REPEATS", 10))
MIN_UNFUNDED = 8                      # below this, within-country CV is not estimable
MODELS = ("logit_l2", "rf", "xgb")
TGT = C.TARGETS["primary"]
FEATS = {"CORE": C.CORE, "CORE_no_age": [c for c in C.CORE if c != "age_years"]}
COUNTRIES = ["Bangladesh", "India", "Pakistan"]
raw = load_startups()
samples = {"FRAME (primary)": in_frame(raw), "ALL ROWS (sensitivity)": raw}
rows = []

def add(sample, fs, analysis, model, test_name, y, p, extra=None):
    auc, lo, hi = boot_auc(y, p)
    rows.append({"sample": sample, "features": fs, "analysis": analysis, "model": model,
                 "test_set": test_name, "n": len(y), "n_unfunded": int((np.asarray(y) == 0).sum()),
                 "auc": round(auc, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), **(extra or {})})

for sname, data in samples.items():
    d, y = labelled(data, TGT)
    d = d.reset_index(drop=True); y = y.reset_index(drop=True)
    strata = d["country"] + "_" + y.astype(str)
    for fs, num in FEATS.items():
        # 1) pooled with country control; score overall and within each country
        num_p = num + C.COUNTRY_DUMMIES
        for mn, mod in make_models(num_p, C.CAT_A).items():
            if mn not in MODELS: continue
            oof = repeated_cv_oof(mod, d[num_p + C.CAT_A], y, 5, REP, strata=strata).mean(axis=0)
            add(sname, fs, "pooled + country control", mn, "all", y, oof)
            for c in COUNTRIES:
                m = (d.country == c).to_numpy()
                add(sname, fs, "pooled + country control", mn, c, y[m], oof[m])
        # 2) within-country CV
        for c in COUNTRIES:
            sub = d[d.country == c].reset_index(drop=True); ys = y[d.country == c].reset_index(drop=True)
            if (ys == 0).sum() < MIN_UNFUNDED:
                rows.append({"sample": sname, "features": fs, "analysis": "within-country", "model": "-",
                             "test_set": c, "n": len(ys), "n_unfunded": int((ys == 0).sum()),
                             "auc": np.nan, "note": f"not estimable (<{MIN_UNFUNDED} unfunded)"})
                continue
            for mn, mod in make_models(num, C.CAT_A).items():
                if mn not in MODELS: continue
                oof = repeated_cv_oof(mod, sub[num + C.CAT_A], ys, 5, REP).mean(axis=0)
                add(sname, fs, "within-country", mn, c, ys, oof)
        # 3) leave-one-country-out (train two countries, test the third)
        for held in COUNTRIES:
            tr, te = d[d.country != held], d[d.country == held]
            for mn, mod in make_models(num, C.CAT_A).items():
                if mn not in MODELS: continue
                p = clone(mod).fit(tr[num + C.CAT_A], y[tr.index]).predict_proba(te[num + C.CAT_A])[:, 1]
                add(sname, fs, "leave-one-country-out", mn, held, y[te.index], p)

res = pd.DataFrame(rows)
C.TABLES.mkdir(parents=True, exist_ok=True)
res.to_csv(C.TABLES / "main_analysis_D9.csv", index=False)
pd.set_option("display.width", 220, "display.max_rows", 400)
print(res[res.features == "CORE"].drop(columns=["features"]).to_string(index=False))
