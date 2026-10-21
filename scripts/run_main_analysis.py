"""Main analysis (D9 + D10): pooled w/ country control, within-country, leave-one-country-out,
for each feature group; paired H2 comparison (RF/XGB vs logistic)."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.base import clone
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame, boot_auc, paired_boot_diff
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof

REP = int(os.environ.get("XAI_REPEATS", 10))
MIN_UNFUNDED = 8
MODELS = ("logit_l2", "rf", "xgb")
TGT = C.TARGETS["primary"]
COUNTRIES = ["Bangladesh", "India", "Pakistan"]
raw = load_startups()
samples = {"FRAME (primary)": in_frame(raw)}
if os.environ.get("XAI_ALLROWS", "1") == "1":
    samples["ALL ROWS (sensitivity)"] = raw
rows, h2 = [], []

def add(sample, grp, analysis, model, test, y, p):
    a, lo, hi = boot_auc(y, p)
    rows.append({"sample": sample, "features": grp, "analysis": analysis, "model": model, "test_set": test,
                 "n": len(y), "n_unfunded": int((np.asarray(y) == 0).sum()),
                 "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})

for sname, data in samples.items():
    d, y = labelled(data, TGT)
    d = d.reset_index(drop=True); y = y.reset_index(drop=True)
    strata = d["country"] + "_" + y.astype(str)
    for grp, (num, cat) in C.FEATURE_GROUPS.items():
        npool = num + C.COUNTRY_DUMMIES
        oofs = {}
        for mn, mod in make_models(npool, cat).items():
            if mn not in MODELS: continue
            oofs[mn] = repeated_cv_oof(mod, d[npool + cat], y, 5, REP, strata=strata).mean(axis=0)
            add(sname, grp, "pooled + country control", mn, "all", y, oofs[mn])
            for c in COUNTRIES:
                m = (d.country == c).to_numpy(); add(sname, grp, "pooled + country control", mn, c, y[m], oofs[mn][m])
        for mn in ("rf", "xgb"):                                   # H2: paired difference vs logistic
            dlt, lo, hi = paired_boot_diff(y, oofs[mn], oofs["logit_l2"])
            h2.append({"sample": sname, "features": grp, "comparison": f"{mn} - logit_l2",
                       "delta_auc": round(dlt, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
        for c in COUNTRIES:                                        # within-country
            m = (d.country == c).to_numpy(); sub = d[m].reset_index(drop=True); ys = y[m].reset_index(drop=True)
            if (ys == 0).sum() < MIN_UNFUNDED:
                rows.append({"sample": sname, "features": grp, "analysis": "within-country", "model": "-",
                             "test_set": c, "n": len(ys), "n_unfunded": int((ys == 0).sum()), "auc": np.nan}); continue
            for mn, mod in make_models(num, cat).items():
                if mn in MODELS:
                    add(sname, grp, "within-country", mn, c, ys,
                        repeated_cv_oof(mod, sub[num + cat], ys, 5, REP).mean(axis=0))
        for held in COUNTRIES:                                     # leave-one-country-out
            tr, te = d[d.country != held], d[d.country == held]
            for mn, mod in make_models(num, cat).items():
                if mn in MODELS:
                    p = clone(mod).fit(tr[num + cat], y[tr.index]).predict_proba(te[num + cat])[:, 1]
                    add(sname, grp, "leave-one-country-out", mn, held, y[te.index], p)

res, h2 = pd.DataFrame(rows), pd.DataFrame(h2)
C.TABLES.mkdir(parents=True, exist_ok=True)
res.to_csv(C.TABLES / "main_analysis.csv", index=False); h2.to_csv(C.TABLES / "h2_paired_differences.csv", index=False)
pd.set_option("display.width", 220, "display.max_rows", 500)
print(res[res["sample"] == "FRAME (primary)"].drop(columns=["sample"]).to_string(index=False))
print("\nH2 paired differences:\n", h2.to_string(index=False))
