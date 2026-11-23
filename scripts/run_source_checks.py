"""Source-confound diagnostics (review item 1: Alvina, Sec. 3.2 / 3.6).

Question: funded firms come from news/Crunchbase-type sources, unfunded ones partly from directory listings.
Could the models be learning the SAMPLING SOURCE instead of funding?

Outputs (results/tables/):
  source_composition.csv      source x country x funding (and source x sector x funding)
  source_baselines.csv        pooled CV AUC: source-only vs country-only vs neutral / sector groups, with and without the source flag
  source_increment.csv        paired AUC differences: what do features add beyond the source flag?
  source_comparable.csv       the key models re-run WITHOUT the v3 directory rows (source-comparable subsample)
  source_comparable_h2.csv    H2 paired differences in that subsample
  source_identification.csv   among unfunded firms only: can the features tell v3-directory from other unfunded?

Run after run_audit. Env: XAI_REPEATS (default 20 = repeats behind the committed tables).
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.base import clone
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame, boot_auc, paired_boot_diff
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof

SEC = os.environ.get("XAI_SECTIONS", "2,3,4").split(",")   # sections 2-4 are model runs; 1 always runs
REP = int(os.environ.get("XAI_REPEATS", 20))
MODELS = ("logit_l2", "rf", "xgb")
TGT = C.TARGETS["primary"]
COUNTRIES = ["Bangladesh", "India", "Pakistan"]
C.TABLES.mkdir(parents=True, exist_ok=True)

raw = load_startups()
fr = in_frame(raw)
d, y = labelled(fr, TGT)
d = d.reset_index(drop=True); y = y.reset_index(drop=True)

# ---------------------------------------------------------------- 0. sanity: source flag == collection batch
batch = d["date_collected"].astype(str).str[:10]
assert ((batch == "2026-10-20") == (d["src_v3"] == 1)).all(), "src_v3 does not match the 2026-10-20 batch"

# ---------------------------------------------------------------- 1. composition tables
comp = (d.assign(funded=y).groupby(["sample_source", "country"])
          .agg(n=("funded", "size"), funded=("funded", "sum")).reset_index())
comp["unfunded"] = comp["n"] - comp["funded"]
comp["pct_unfunded"] = (100 * comp["unfunded"] / comp["n"]).round(1)
tot = (d.assign(funded=y).groupby("sample_source")
         .agg(n=("funded", "size"), funded=("funded", "sum")).reset_index())
tot["country"] = "ALL"; tot["unfunded"] = tot["n"] - tot["funded"]
tot["pct_unfunded"] = (100 * tot["unfunded"] / tot["n"]).round(1)
comp = pd.concat([comp, tot], ignore_index=True)
sec = (pd.crosstab([d["sector_group"]], [d["sample_source"], y]).rename(columns={0: "unfunded", 1: "funded"}))
sec.columns = [f"{a}|{b}" for a, b in sec.columns]
comp.to_csv(C.TABLES / "source_composition.csv", index=False)
sec.to_csv(C.TABLES / "source_composition_by_sector.csv")
pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_columns", 30)
print("== SOURCE x COUNTRY x FUNDING (primary frame, labelled) ==\n", comp.to_string(index=False))
print("\n== SECTOR x SOURCE x FUNDING ==\n", sec.to_string())
# per-country AUC of the fixed source rule (compare with the LOCO sector AUCs in Table 2)
pc = []
for cty in COUNTRIES:
    m = (d.country == cty).values
    a, lo, hi = boot_auc(y[m], (1 - d["src_v3"]).astype(float).values[m])
    pc.append({"country": cty, "n": int(m.sum()), "n_unfunded": int((y[m] == 0).sum()),
               "n_unfunded_v3": int(((d.src_v3 == 1) & m).sum()), "rule_auc": round(a, 3),
               "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
pd.DataFrame(pc).to_csv(C.TABLES / "source_rule_by_country.csv", index=False)
print("\n== FIXED SOURCE RULE, AUC BY COUNTRY ==\n", pd.DataFrame(pc).to_string(index=False))
nonv3_unf = ((d.src_v3 == 0) & (y == 0)).sum(); v3_unf = ((d.src_v3 == 1) & (y == 0)).sum()
print(f"\nUnfunded = {int((y==0).sum())}: {v3_unf} from v3 directory sample (no funded counterpart), "
      f"{nonv3_unf} from v1/v2 (same sources as the funded firms).")

def run_cv(data, yy, num, cat, tag, models=MODELS, strata=None):
    out, oofs = [], {}
    for mn, mod in make_models(num, cat).items():
        if mn not in models: continue
        oof = repeated_cv_oof(mod, data[num + cat], yy, 5, REP, strata=strata).mean(axis=0)
        oofs[mn] = oof
        a, lo, hi = boot_auc(yy, oof)
        out.append({"spec": tag, "model": mn, "n": len(yy), "n_unfunded": int((yy == 0).sum()),
                    "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    return out, oofs

strata = d["country"] + "_" + y.astype(str)
CD = C.COUNTRY_DUMMIES

# ---------------------------------------------------------------- 2. baselines with / without the source flag
if "2" in SEC:
    specs = {                                                # name: (numeric, categorical)
        "country only":                    (CD, []),
        "source flag only":                (C.SOURCE, []),
        "source flag + country":           (C.SOURCE + CD, []),
        "neutral (+country)":              (C.NEUTRAL + CD, []),
        "neutral + source flag":           (C.NEUTRAL + CD + C.SOURCE, []),
        "neutral+sector (+country)":       (C.NEUTRAL + C.SECTOR_BIZ + CD, C.CAT_A),
        "neutral+sector + source flag":    (C.NEUTRAL + C.SECTOR_BIZ + CD + C.SOURCE, C.CAT_A),
        "sector only (+country)":          (C.SECTOR_BIZ + CD, C.CAT_A),
        "sector only + source flag":       (C.SECTOR_BIZ + CD + C.SOURCE, C.CAT_A),
    }
    rows, oof_store = [], {}
    for name, (num, cat) in specs.items():
        r, o = run_cv(d, y, num, cat, name, strata=strata)
        rows += r; oof_store[name] = o
    # Fixed rule (no fitting): score = 1 - src_v3. A 1-parameter model fitted inside CV is distorted by the leave-out
    # anti-correlation of tied scores, so the source flag alone is reported as a fixed rule (analytic AUC = 0.5*(1+41/59)).
    rule = (1 - d["src_v3"]).astype(float).values
    a, lo, hi = boot_auc(y, rule)
    rows.append({"spec": "source flag only (fixed rule, no fitting)", "model": "rule", "n": len(y),
                 "n_unfunded": int((y == 0).sum()), "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    base = pd.DataFrame(rows); base.to_csv(C.TABLES / "source_baselines.csv", index=False)
    print("\n== POOLED CV AUC: does the source flag alone reproduce the 'sector' AUC? ==")
    print(base.pivot_table(index="spec", columns="model", values="auc", sort=False).to_string())
    print(base[base.model == "rule"].to_string(index=False))

    # increments beyond the source flag (paired bootstrap on out-of-fold predictions)
    oof_store["source flag rule"] = {m: rule for m in MODELS}   # fixed rule as the fair comparator
    inc = []
    for mn in MODELS:
        for a, b in (("neutral + source flag", "source flag rule"),
                     ("neutral+sector + source flag", "source flag rule"),
                     ("sector only + source flag", "source flag rule"),
                     ("neutral+sector (+country)", "source flag rule"),
                     ("sector only (+country)", "source flag rule"),
                     ("neutral (+country)", "source flag rule"),
                     ("neutral+sector + source flag", "neutral + source flag")):
            dl, lo, hi = paired_boot_diff(y, oof_store[a][mn], oof_store[b][mn])
            inc.append({"model": mn, "A": a, "minus_B": b, "delta_auc": round(dl, 3),
                        "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    inc = pd.DataFrame(inc); inc.to_csv(C.TABLES / "source_increment.csv", index=False)
    print("\n== PAIRED INCREMENTS (A minus B) ==\n", inc.to_string(index=False))

# ---------------------------------------------------------------- 3. source-comparable subsample (drop v3 directory rows)
if "3" in SEC:
    sub = d[d.src_v3 == 0].reset_index(drop=True); ys = y[d.src_v3 == 0].reset_index(drop=True)
    print(f"\n== SOURCE-COMPARABLE SUBSAMPLE: n={len(sub)}, unfunded={int((ys==0).sum())} "
          f"(BD {int(((sub.country=='Bangladesh')&(ys==0)).sum())}, IN {int(((sub.country=='India')&(ys==0)).sum())}, "
          f"PK {int(((sub.country=='Pakistan')&(ys==0)).sum())}) ==")
    rows, h2 = [], []
    for grp, (num, cat) in C.FEATURE_GROUPS.items():
        npool = num + CD
        r, o = run_cv(sub, ys, npool, cat, f"{grp} | pooled+country", strata=ys)  # label-only strata: PK has 4 unfunded
        for x in r: x["features"] = grp; x["analysis"] = "pooled + country control"; x["test_set"] = "all"
        rows += r
        for mn in ("rf", "xgb"):
            dl, lo, hi = paired_boot_diff(ys, o[mn], o["logit_l2"])
            h2.append({"features": grp, "comparison": f"{mn} - logit_l2", "delta_auc": round(dl, 3),
                       "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
        for held in COUNTRIES:                                           # leave-one-country-out
            tr, te = sub[sub.country != held], sub[sub.country == held]
            yte = ys[te.index]
            for mn, mod in make_models(num, cat).items():
                if mn not in MODELS: continue
                if len(np.unique(yte)) < 2: continue
                p = clone(mod).fit(tr[num + cat], ys[tr.index]).predict_proba(te[num + cat])[:, 1]
                a, lo, hi = boot_auc(yte, p)
                rows.append({"spec": f"{grp} | LOCO", "model": mn, "n": len(yte), "n_unfunded": int((yte == 0).sum()),
                             "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                             "features": grp, "analysis": "leave-one-country-out", "test_set": held})
    comp_df = pd.DataFrame(rows); comp_df.to_csv(C.TABLES / "source_comparable.csv", index=False)
    h2 = pd.DataFrame(h2); h2.to_csv(C.TABLES / "source_comparable_h2.csv", index=False)
    print(comp_df[comp_df.analysis == "pooled + control" if False else comp_df.analysis == "pooled + country control"]
          [["features", "model", "n", "n_unfunded", "auc", "ci_lo", "ci_hi"]].to_string(index=False))
    print("\nLOCO (source-comparable):\n", comp_df[comp_df.analysis == "leave-one-country-out"]
          [["features", "model", "test_set", "n", "n_unfunded", "auc", "ci_lo", "ci_hi"]].to_string(index=False))
    print("\nH2 paired (source-comparable):\n", h2.to_string(index=False))

# ---------------------------------------------------------------- 4. source identification among UNFUNDED firms
if "4" in SEC:
    unf = d[y == 0].reset_index(drop=True); ysrc = unf["src_v3"].astype(int)
    rows = []
    for grp, (num, cat) in C.FEATURE_GROUPS.items():
        npool = num + CD
        for mn, mod in make_models(npool, cat).items():
            if mn not in MODELS: continue
            oof = repeated_cv_oof(mod, unf[npool + cat], ysrc, 5, REP).mean(axis=0)
            a, lo, hi = boot_auc(ysrc, oof)
            rows.append({"features": grp, "model": mn, "n_unfunded": len(unf), "n_v3": int(ysrc.sum()),
                         "auc_predict_v3_vs_other_unfunded": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    ident = pd.DataFrame(rows); ident.to_csv(C.TABLES / "source_identification.csv", index=False)
    print("\n== AMONG UNFUNDED ONLY: AUC for telling v3-directory firms from other unfunded firms ==\n", ident.to_string(index=False))
