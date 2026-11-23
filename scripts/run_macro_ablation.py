"""Macroeconomic-feature ablation (review item 3: Alvina, Sec. 5.6; assigned task from Emon).

Concern: GDP growth and inflation at founding are country x year constants, so they act as country/cohort
identifiers rather than independent macro signals. Request: re-run the relevant models WITHOUT both variables,
keeping sample, CV splits (5-fold x REP, same seed, country x label strata), and hyper-parameters unchanged;
report AUC, uncertainty, leave-one-country-out performance, and SHAP rankings.

Sections (env XAI_SECTIONS, default 1-6):
  1 diagnostics: what do GDP/inflation encode?                     -> macro_cells.csv, macro_corr.csv
  2 pooled CV AUC, original vs no-macro, paired differences, H2    -> macro_pooled.csv, macro_paired.csv, macro_h2.csv
  3 leave-one-country-out + generalisation criterion               -> macro_loco.csv
  4 SHAP ranks (pooled neutral, neutral+sector, BD founder)        -> macro_shap.csv
  5 Bangladesh founder models (H1/H3) and simulation (H4)          -> macro_founder.csv, macro_sim_*.csv
  6 source-comparable subsample (v3 directory rows removed)        -> macro_nonv3.csv
Env: XAI_REPEATS (default 20 = repeats behind the committed tables); XAI_BUDGET seconds per call (default 230);
     out-of-fold predictions are cached in .cache_macro/ so an interrupted run can simply be repeated.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame, boot_auc, paired_boot_diff
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof
from xai_capital.explain import fit_and_shap, global_importance
from xai_capital.simulate import run as sim_run

SEC = os.environ.get("XAI_SECTIONS", "1,2,3,4,5,6").split(",")
REP = int(os.environ.get("XAI_REPEATS", 20)); SIM_REP = int(os.environ.get("XAI_SIM_REPEATS", 20))
T0 = time.time(); BUDGET = float(os.environ.get("XAI_BUDGET", 230))
CACHE = Path(".cache_macro"); CACHE.mkdir(exist_ok=True)
TGT = C.TARGETS["primary"]; CD = C.COUNTRY_DUMMIES; COUNTRIES = ["Bangladesh", "India", "Pakistan"]
MODELS = ("logit_l2", "rf", "xgb")
MACRO = ["gdp_growth_at_founding", "inflation_at_founding"]
NEUT_NM = [c for c in C.NEUTRAL if c not in MACRO]                  # age, product_stage, n_founders
pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 30)
C.TABLES.mkdir(parents=True, exist_ok=True)

fr = in_frame(load_startups())
d, y = labelled(fr, TGT); d = d.reset_index(drop=True); y = y.reset_index(drop=True)
strata = d["country"] + "_" + y.astype(str)

def budget_check():
    if time.time() - T0 > BUDGET:
        print(f"[budget reached] rerun the same command to resume ({len(list(CACHE.glob('*.npy')))} cached arrays)")
        sys.exit(0)

def cached_oof(tag, num, cat, data, yy, strat, mn):
    cf = CACHE / f"{tag}__{mn}__R{REP}.npy"
    if cf.exists(): return np.load(cf)
    budget_check()
    o = repeated_cv_oof(make_models(num, cat)[mn], data[num + cat], yy, 5, REP, strata=strat).mean(axis=0)
    np.save(cf, o); return o

# pooled specifications (country dummies added, as in run_main_analysis.py)
SPECS = {
    "neutral (original)":                (C.NEUTRAL + CD, []),
    "neutral, no macro":                 (NEUT_NM + CD, []),
    "neutral+sector (original)":         (C.NEUTRAL + C.SECTOR_BIZ + CD, C.CAT_A),
    "neutral+sector, no macro":          (NEUT_NM + C.SECTOR_BIZ + CD, C.CAT_A),
    "diag: macro only (+country)":       (MACRO + CD, []),
    "diag: age only (+country)":         (["age_years"] + CD, []),
    "diag: no macro, no age (+country)": (["product_stage", "n_founders"] + CD, []),
}
PAIRS = (("neutral (original)", "neutral, no macro"), ("neutral+sector (original)", "neutral+sector, no macro"))

# ------------------------------------------------------------------------------------------ 1. diagnostics
if "1" in SEC:
    g = d.assign(funded=y).groupby(["country", "founding_year"]).agg(
        n=("funded", "size"), funded_rate=("funded", "mean"),
        n_gdp_values=("gdp_growth_at_founding", "nunique"), n_infl_values=("inflation_at_founding", "nunique")).reset_index()
    g.to_csv(C.TABLES / "macro_cells.csv", index=False)
    print(f"== 1. Country x founding-year cells: {len(g)} cells for n={len(d)} firms (median {int(g.n.median())} firms/cell)")
    print(f"   Each cell has exactly one GDP value and one inflation value: "
          f"{bool((g.n_gdp_values == 1).all() and (g.n_infl_values == 1).all())}")
    print(f"   Distinct GDP values = {d.gdp_growth_at_founding.nunique()}, distinct inflation values = {d.inflation_at_founding.nunique()}")
    rows = []
    for c in COUNTRIES:
        s = d[d.country == c]; ys = y[d.country == c]
        rows.append({"country": c, "n": len(s),
                     "rho(gdp, founding_year)": round(spearmanr(s.gdp_growth_at_founding, s.founding_year)[0], 2),
                     "rho(infl, founding_year)": round(spearmanr(s.inflation_at_founding, s.founding_year)[0], 2),
                     "rho(gdp, funded)": round(spearmanr(s.gdp_growth_at_founding, ys)[0], 2),
                     "rho(infl, funded)": round(spearmanr(s.inflation_at_founding, ys)[0], 2),
                     "rho(founding_year, funded)": round(spearmanr(s.founding_year, ys)[0], 2)})
    cr = pd.DataFrame(rows); cr.to_csv(C.TABLES / "macro_corr.csv", index=False)
    print("\n   Within-country Spearman correlations:\n", cr.to_string(index=False))
    print("\n   Funded rate by founding-year band and country:")
    band = pd.cut(d.founding_year, [2009, 2014, 2018, 2023], labels=["2010-14", "2015-18", "2019-23"])
    print(d.assign(funded=y, band=band).pivot_table(index="band", columns="country", values="funded", aggfunc=["mean", "size"]).round(2).to_string())

# ------------------------------------------------------------------------------------------ 2. pooled CV
if "2" in SEC:
    rows, oofs = [], {}
    for name, (num, cat) in SPECS.items():
        for mn in MODELS:
            tag = f"pooled_{name.replace(' ', '_').replace(',', '').replace('+', 'p').replace(':', '').replace('(', '').replace(')', '')}"
            o = cached_oof(tag, num, cat, d, y, strata, mn); oofs[(name, mn)] = o
            a, lo, hi = boot_auc(y, o)
            row = {"spec": name, "model": mn, "n": len(y), "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)}
            for c in COUNTRIES:
                m = (d.country == c).to_numpy(); row[f"auc_{c}"] = round(boot_auc(y[m], o[m])[0], 3)
            rows.append(row)
    pooled = pd.DataFrame(rows); pooled.to_csv(C.TABLES / "macro_pooled.csv", index=False)
    print("\n== 2a. Pooled CV AUC (5-fold x %d, country x label strata) ==" % REP)
    print(pooled.pivot_table(index="spec", columns="model", values="auc", sort=False).to_string())
    pr = []
    for a, b in PAIRS:
        for mn in MODELS:
            dl, lo, hi = paired_boot_diff(y, oofs[(a, mn)], oofs[(b, mn)])
            pr.append({"original": a, "minus_ablated": b, "model": mn, "delta_auc": round(dl, 3),
                       "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    pr = pd.DataFrame(pr); pr.to_csv(C.TABLES / "macro_paired.csv", index=False)
    print("\n== 2b. Paired: original minus no-macro (positive = macro variables helped) ==\n", pr.to_string(index=False))
    h2 = []
    for name in ("neutral (original)", "neutral, no macro", "neutral+sector (original)", "neutral+sector, no macro"):
        for mn in ("rf", "xgb"):
            dl, lo, hi = paired_boot_diff(y, oofs[(name, mn)], oofs[(name, "logit_l2")])
            h2.append({"spec": name, "comparison": f"{mn} - logit_l2", "delta_auc": round(dl, 3),
                       "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    h2 = pd.DataFrame(h2); h2.to_csv(C.TABLES / "macro_h2.csv", index=False)
    print("\n== 2c. H2 paired differences (nonlinear - logistic) ==\n", h2.to_string(index=False))

# ------------------------------------------------------------------------------------------ 3. LOCO
if "3" in SEC:
    loco_specs = {"neutral (original)": (C.NEUTRAL, []), "neutral, no macro": (NEUT_NM, []),
                  "neutral+sector (original)": (C.NEUTRAL + C.SECTOR_BIZ, C.CAT_A),
                  "neutral+sector, no macro": (NEUT_NM + C.SECTOR_BIZ, C.CAT_A),
                  "diag: macro only": (MACRO, [])}
    rows, store = [], {}
    for name, (num, cat) in loco_specs.items():
        for mn in MODELS:
            for held in COUNTRIES:
                tr, te = d[d.country != held], d[d.country == held]
                p = clone(make_models(num, cat)[mn]).fit(tr[num + cat], y[tr.index]).predict_proba(te[num + cat])[:, 1]
                store[(name, mn, held)] = (y[te.index].to_numpy(), p)
                a, lo, hi = boot_auc(y[te.index], p)
                rows.append({"spec": name, "model": mn, "held_out": held, "n": len(te), "n_unfunded": int((y[te.index] == 0).sum()),
                             "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    loco = pd.DataFrame(rows); loco.to_csv(C.TABLES / "macro_loco.csv", index=False)
    print("\n== 3a. LOCO AUC ==")
    print(loco.pivot_table(index=["spec", "model"], columns="held_out", values="auc", sort=False).to_string())
    print("\n== 3b. Generalisation criterion (neutral groups): lower CI > 0.50 in >= 2 held-out countries ==")
    for name in ("neutral (original)", "neutral, no macro"):
        for mn in MODELS:
            ok = [c for c in COUNTRIES if loco[(loco.spec == name) & (loco.model == mn) & (loco.held_out == c)].ci_lo.iloc[0] > 0.5]
            print(f"   {name:22s}{mn:9s} lower bound > 0.50 in: {ok or 'none'}  -> {'MET' if len(ok) >= 2 else 'not met'}")
    pl = []
    for a, b in PAIRS:
        for mn in MODELS:
            for held in COUNTRIES:
                (ya, pa), (yb, pb) = store[(a, mn, held)], store[(b, mn, held)]
                dl, lo, hi = paired_boot_diff(pd.Series(ya), pa, pb)
                pl.append({"original": a, "minus_ablated": b, "model": mn, "held_out": held,
                           "delta_auc": round(dl, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    pl = pd.DataFrame(pl); pl.to_csv(C.TABLES / "macro_loco_paired.csv", index=False)
    print("\n== 3c. LOCO paired (original minus no-macro) ==\n", pl.to_string(index=False))

# ------------------------------------------------------------------------------------------ 4. SHAP
if "4" in SEC:
    bd = fr[(fr.country == "Bangladesh") & fr[C.FOUNDER].notna().all(axis=1)]
    cases = [("pooled neutral | original", fr, C.NEUTRAL + CD, []), ("pooled neutral | no macro", fr, NEUT_NM + CD, []),
             ("pooled neutral+sector | original", fr, C.NEUTRAL + C.SECTOR_BIZ + CD, C.CAT_A),
             ("pooled neutral+sector | no macro", fr, NEUT_NM + C.SECTOR_BIZ + CD, C.CAT_A),
             ("BD founder rows | original", bd, C.NEUTRAL + C.FOUNDER, []), ("BD founder rows | no macro", bd, NEUT_NM + C.FOUNDER, [])]
    out = []
    for tag, data, num, cat in cases:
        dd, yy = labelled(data, TGT)
        _, S = fit_and_shap(make_models(num, cat)["xgb"], dd[num + cat], yy)
        imp = global_importance(S)
        out.append(pd.DataFrame({"case": tag, "rank": range(1, len(imp) + 1), "feature": imp.index, "mean_abs_shap": imp.values.round(3)}))
    sh = pd.concat(out); sh.to_csv(C.TABLES / "macro_shap.csv", index=False)
    for pair in (cases[0:2], cases[2:4], cases[4:6]):
        a, b = (sh[sh.case == c[0]].head(7).reset_index(drop=True) for c in pair)
        print(f"\n== 4. SHAP top-7: {pair[0][0]}  vs  {pair[1][0]} ==")
        print(pd.concat([a[["rank", "feature", "mean_abs_shap"]], b[["feature", "mean_abs_shap"]].add_suffix("_nomacro")], axis=1).to_string(index=False))

# ------------------------------------------------------------------------------------------ 5. BD founder + sim
if "5" in SEC:
    bd = fr[(fr.country == "Bangladesh") & fr[C.FOUNDER].notna().all(axis=1)]
    db, yb = labelled(bd, TGT); db = db.reset_index(drop=True); yb = yb.reset_index(drop=True)
    sets = {"controls only | original": C.NEUTRAL, "controls only | no macro": NEUT_NM,
            "controls + founder | original": C.NEUTRAL + C.FOUNDER, "controls + founder | no macro": NEUT_NM + C.FOUNDER,
            # combined with the STEM exclusion of item 2 (cannot interpret a macro-free H1 gain without it)
            "controls + founder noSTEM | original": C.NEUTRAL + [c for c in C.FOUNDER if c != "any_stem_founder"],
            "controls + founder noSTEM | no macro": NEUT_NM + [c for c in C.FOUNDER if c != "any_stem_founder"]}
    oo, rows = {}, []
    for nm, num in sets.items():
        for mn in ("logit_l2", "rf"):
            o = cached_oof("bdfounder_" + nm.replace(" ", "_").replace("|", "").replace("+", "p"), num, [], db, yb, None, mn)
            oo[(nm, mn)] = o; a, lo, hi = boot_auc(yb, o)
            rows.append({"feature_set": nm, "model": mn, "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    for mn in ("logit_l2", "rf"):
        for v in ("original", "no macro"):
          for fs, lab in (("controls + founder", "with STEM"), ("controls + founder noSTEM", "no STEM")):
            dl, lo, hi = paired_boot_diff(yb, oo[(f"{fs} | {v}", mn)], oo[(f"controls only | {v}", mn)])
            rows.append({"feature_set": f"H1 gain ({lab}) | {v}", "model": mn,
                         "auc": round(dl, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    bdr = pd.DataFrame(rows); bdr.to_csv(C.TABLES / "macro_founder.csv", index=False)
    print("\n== 5a. Bangladesh founder models (n=53), AUC [95% CI]; H1 rows are paired AUC gains ==\n", bdr.to_string(index=False))

    res_all, summ = [], []
    for ver, ctrl in (("original", C.NEUTRAL), ("no macro", NEUT_NM)):
        num, cat = ctrl + C.SECTOR_BIZ + C.FOUNDER, C.CAT_A
        cf = CACHE / f"sim_{ver.replace(' ', '_')}_R{SIM_REP}.csv"
        if cf.exists(): r = pd.read_csv(cf)
        else:
            budget_check()
            r = sim_run(make_models(num, cat), db[num + cat], yb, outcome=yb, k_frac=0.3, tau=0.4, n_repeats=SIM_REP); r.to_csv(cf, index=False)
        piv = r.pivot_table(index="fold", columns="investor", values="hit_rate")
        s = piv.agg(["mean", "std"]).T.round(3); s["version"] = ver; summ.append(s.reset_index())
        rng = np.random.default_rng(42)
        for a, b in (("xai", "black_box"), ("xai", "heuristic"), ("xai", "econometric"), ("black_box", "heuristic"),
                     ("black_box", "econometric"), ("xai", "random")):
            diff = (piv[a] - piv[b]).to_numpy(); bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            res_all.append({"version": ver, "comparison": f"{a} - {b}", "mean_diff": round(diff.mean(), 3),
                            "ci_lo": round(np.percentile(bs, 2.5), 3), "ci_hi": round(np.percentile(bs, 97.5), 3)})
    summ = pd.concat(summ); summ.to_csv(C.TABLES / "macro_sim_summary.csv", index=False)
    pd.DataFrame(res_all).to_csv(C.TABLES / "macro_sim_paired.csv", index=False)
    print("\n== 5b. Simulated selection hit rates (top-30%%, base rate %.2f) ==\n" % yb.mean(),
          summ.pivot(index="investor", columns="version", values="mean").to_string())
    print("\n", pd.DataFrame(res_all).to_string(index=False))

# ------------------------------------------------------------------------------------------ 6. source-comparable
if "6" in SEC:
    m3 = (d.src_v3 == 0).to_numpy(); ds = d[m3].reset_index(drop=True); ys = y[m3].reset_index(drop=True)
    print(f"\n== 6. Source-comparable subsample (v3 directory rows removed): n={len(ds)}, unfunded={int((ys==0).sum())} ==")
    rows, o3 = [], {}
    for name in ("neutral (original)", "neutral, no macro", "neutral+sector (original)", "neutral+sector, no macro"):
        num, cat = SPECS[name]
        for mn in MODELS:
            o = cached_oof("nonv3_" + name.replace(" ", "_").replace(",", "").replace("+", "p").replace("(", "").replace(")", ""),
                           num, cat, ds, ys, ys, mn)
            o3[(name, mn)] = o; a, lo, hi = boot_auc(ys, o)
            rows.append({"spec": name, "model": mn, "n": len(ys), "n_unfunded": int((ys == 0).sum()),
                         "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    for a, b in PAIRS:
        for mn in MODELS:
            dl, lo, hi = paired_boot_diff(ys, o3[(a, mn)], o3[(b, mn)])
            rows.append({"spec": f"PAIRED {a} - {b}", "model": mn, "n": len(ys), "n_unfunded": int((ys == 0).sum()),
                         "auc": round(dl, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    nv = pd.DataFrame(rows); nv.to_csv(C.TABLES / "macro_nonv3.csv", index=False); print(nv.to_string(index=False))
