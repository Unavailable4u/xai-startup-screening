"""H1 and H3 on Bangladesh complete cases (founder attributes are observed mainly there)."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame, boot_auc, paired_boot_diff
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof

REP = int(os.environ.get("XAI_REPEATS", 10)); TGT = C.TARGETS["primary"]
fr = in_frame(load_startups()); bd = fr[(fr.country == "Bangladesh") & fr[TGT].notna()].reset_index(drop=True)
obs = bd[C.FOUNDER].notna().all(axis=1)
print("SELECTION DIAGNOSTIC (Bangladesh, frame): share with complete founder data")
print(bd.assign(complete=obs).groupby(TGT).complete.agg(["size", "sum", "mean"]).round(2).rename(
    index={0.0: "unfunded", 1.0: "funded"}).to_string())
d = bd[obs].reset_index(drop=True); y = d[TGT].astype(int)
print(f"\ncomplete cases: n={len(d)}, unfunded={int((y==0).sum())}")
print("means by label:\n", d.groupby(TGT)[C.FOUNDER].mean().round(2).to_string())

sets = {"controls only": C.NEUTRAL, "founder only": C.FOUNDER,
        "controls + founder": C.NEUTRAL + C.FOUNDER,
        "fundamentals": C.FUNDAMENTALS, "fundamentals + prestige": C.FUNDAMENTALS + [p for p in C.PRESTIGE if p not in C.FUNDAMENTALS]}
oof, rows = {}, []
for nm, num in sets.items():
    for mn in ("logit_l2", "rf"):
        o = repeated_cv_oof(make_models(num, [])[mn], d[num], y, 5, REP).mean(axis=0); oof[(nm, mn)] = o
        a, lo, hi = boot_auc(y, o); rows.append({"feature_set": nm, "model": mn, "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
perf = pd.DataFrame(rows); print("\nCV performance:\n", perf.to_string(index=False))

cmp = []
for mn in ("logit_l2", "rf"):
    for a, b, h in (("controls + founder", "controls only", "H1: founder attrs add to controls"),
                    ("fundamentals + prestige", "fundamentals", "H3: prestige adds to fundamentals")):
        dl, lo, hi = paired_boot_diff(y, oof[(a, mn)], oof[(b, mn)])
        cmp.append({"hypothesis": h, "model": mn, "delta_auc": round(dl, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                    "H3_rule_met(delta<0.02)": bool(dl < 0.02) if h.startswith("H3") else ""})
cmp = pd.DataFrame(cmp); print("\nPaired comparisons:\n", cmp.to_string(index=False))

# standardised logistic odds ratios with bootstrap CI (founder attributes + controls)
num = C.FOUNDER + ["age_years", "product_stage"]
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(d[num]), columns=num)
X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
rng = np.random.default_rng(42); coefs = []
for _ in range(500):
    i = rng.integers(0, len(y), len(y))
    if y.iloc[i].nunique() == 2:
        coefs.append(LogisticRegression(C=0.5, max_iter=2000).fit(X.iloc[i], y.iloc[i]).coef_[0])
coefs = np.array(coefs); full = LogisticRegression(C=0.5, max_iter=2000).fit(X, y).coef_[0]
orr = pd.DataFrame({"feature": num, "odds_ratio_per_SD": np.exp(full).round(2),
                    "ci_lo": np.exp(np.percentile(coefs, 2.5, axis=0)).round(2), "ci_hi": np.exp(np.percentile(coefs, 97.5, axis=0)).round(2)})
print("\nStandardised odds ratios (L2 logistic, bootstrap 95% CI):\n", orr.to_string(index=False))
C.TABLES.mkdir(parents=True, exist_ok=True)
perf.to_csv(C.TABLES / "founder_cv_performance.csv", index=False); cmp.to_csv(C.TABLES / "h1_h3_paired.csv", index=False)
orr.to_csv(C.TABLES / "founder_odds_ratios.csv", index=False)
