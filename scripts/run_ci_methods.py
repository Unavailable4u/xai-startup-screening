"""Uncertainty methodology checks (review item 10).

Current main-table intervals: percentile bootstrap over the EVALUATION firms of out-of-fold predictions that were
averaged across 20 CV repeats; LOCO intervals resample only the held-out country. They ignore (i) variability from
refitting on different training samples, (ii) split-to-split variability, and (iii) averaging effects.

Sections (env XAI_SECTIONS, default 1,2,3):
  1 averaging check: AUC of averaged predictions vs mean of per-repeat AUCs, spread across repeats, and a firm-bootstrap CI
    for the mean-of-repeats AUC                                              -> ci_averaging.csv
  2 whole-pipeline bootstrap, pooled models: resample firms (stratified country x label), refit with GROUPED folds
    (duplicates of a firm stay in one fold, so no leakage), AUC per replicate; paired H2 differences from the same
    replicates                                                               -> ci_pipeline_pooled.csv, ci_pipeline_paired.csv
  3 LOCO refit bootstrap: resample training firms AND held-out firms, refit; re-checks the lower-bound > 0.50
    transferability criterion                                                -> ci_loco_refit.csv
Env: XAI_BOOT (replicates, default 200), XAI_REPEATS (default 20), XAI_BUDGET (seconds per call, default 235).
Replicates are cached in .cache_ci/ so an interrupted run resumes.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame, boot_auc
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof

SEC = os.environ.get("XAI_SECTIONS", "1,2,3").split(","); B = int(os.environ.get("XAI_BOOT", 200))
REP = int(os.environ.get("XAI_REPEATS", 20)); T0 = time.time(); BUDGET = float(os.environ.get("XAI_BUDGET", 235))
CACHE = Path(".cache_ci"); CACHE.mkdir(exist_ok=True); (CACHE / ".gitignore").write_text("*\n")
TGT = C.TARGETS["primary"]; CD = C.COUNTRY_DUMMIES; COUNTRIES = ["Bangladesh", "India", "Pakistan"]; MODELS = ("logit_l2", "rf", "xgb")
MACRO = ["gdp_growth_at_founding", "inflation_at_founding"]; NEUT_NM = [c for c in C.NEUTRAL if c not in MACRO]
pd.set_option("display.width", 220, "display.max_rows", 200)
d, y = labelled(in_frame(load_startups()), TGT); d = d.reset_index(drop=True); y = y.reset_index(drop=True)
strata = d.country + "_" + y.astype(str); codes = strata.astype("category").cat.codes.to_numpy()
SPECS = {"neutral": (C.NEUTRAL + CD, []), "neutral+sector": (C.NEUTRAL + C.SECTOR_BIZ + CD, C.CAT_A),
         "sector_only": (C.SECTOR_BIZ + CD, C.CAT_A)}
MAIN = pd.read_csv(C.TABLES / "main_analysis.csv"); MAIN = MAIN[(MAIN["sample"] == "FRAME (primary)")]
def out_of_time():
    if time.time() - T0 > BUDGET: print("[budget reached] rerun the same command to resume"); sys.exit(0)

# ------------------------------------------------------------------------------------------ 1. averaging
if "1" in SEC:
    rows, rng = [], np.random.default_rng(42)
    for sname, (num, cat) in SPECS.items():
        for mn in MODELS:
            cf = CACHE / f"rep_{sname}_{mn}_R{REP}.npy"
            if cf.exists(): oof = np.load(cf)
            else:
                out_of_time(); oof = repeated_cv_oof(make_models(num, cat)[mn], d[num + cat], y, 5, REP, strata=strata); np.save(cf, oof)
            a_avg, lo, hi = boot_auc(y, oof.mean(axis=0))
            per = np.array([roc_auc_score(y, o) for o in oof])
            bs = []
            for _ in range(400):
                i = np.concatenate([rng.choice(np.flatnonzero(codes == c), (codes == c).sum()) for c in np.unique(codes)])
                if y.iloc[i].nunique() < 2: continue
                bs.append(np.mean([roc_auc_score(y.iloc[i], o[i]) for o in oof]))
            rows.append({"spec": sname, "model": mn, "auc_avg_pred": round(a_avg, 3), "ci_current": f"[{lo:.3f}, {hi:.3f}]",
                         "width_current": round(hi - lo, 3), "auc_mean_per_repeat": round(per.mean(), 3),
                         "sd_across_repeats": round(per.std(ddof=1), 3), "min_rep": round(per.min(), 3), "max_rep": round(per.max(), 3),
                         "avg_minus_mean": round(a_avg - per.mean(), 3),
                         "ci_firm_boot_mean_of_repeats": f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]",
                         "width_mean_of_repeats": round(np.percentile(bs, 97.5) - np.percentile(bs, 2.5), 3)})
    av = pd.DataFrame(rows); av.to_csv(C.TABLES / "ci_averaging.csv", index=False)
    print("== 1. Averaging across repeats vs per-repeat AUC ==\n", av.to_string(index=False))

# ------------------------------------------------------------------------------------------ 2. pipeline bootstrap (pooled)
def pipeline_replicate(num, cat, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(codes == c), (codes == c).sum()) for c in np.unique(codes)])
    Xb, yb = d[num + cat].iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    models = make_models(num, cat); res = {m: np.zeros(len(idx)) for m in MODELS}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xb, codes[idx], idx):   # groups = original firm
        for m in MODELS: res[m][te] = clone(models[m]).fit(Xb.iloc[tr], yb.iloc[tr]).predict_proba(Xb.iloc[te])[:, 1]
    return {m: roc_auc_score(yb, p) for m, p in res.items()}

if "2" in SEC:
    summ, pair = [], []
    for sname in ("neutral", "neutral+sector"):
        num, cat = SPECS[sname]; cf = CACHE / f"pipe_{sname}.csv"
        done = pd.read_csv(cf) if cf.exists() else pd.DataFrame(columns=["seed", *MODELS])
        while len(done) < B:
            out_of_time(); s = len(done); r = pipeline_replicate(num, cat, 1000 + s)
            done = pd.concat([done, pd.DataFrame([{"seed": s, **r}])], ignore_index=True); done.to_csv(cf, index=False)
        for mn in MODELS:
            pt = MAIN[(MAIN.features == sname) & (MAIN.model == mn) & (MAIN.analysis == "pooled + country control") & (MAIN.test_set == "all")].iloc[0]
            v = done[mn].to_numpy()
            summ.append({"spec": sname, "model": mn, "B": len(v), "auc_point_current": pt.auc,
                         "current_ci": f"[{pt.ci_lo:.3f}, {pt.ci_hi:.3f}]", "current_width": round(pt.ci_hi - pt.ci_lo, 3),
                         "pipeline_mean": round(v.mean(), 3), "pipeline_sd": round(v.std(ddof=1), 3),
                         "pipeline_pct_ci": f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]",
                         "pipeline_width": round(np.percentile(v, 97.5) - np.percentile(v, 2.5), 3),
                         "pipeline_sd_ci (point +- 1.96 sd)": f"[{pt.auc - 1.96 * v.std(ddof=1):.3f}, {pt.auc + 1.96 * v.std(ddof=1):.3f}]",
                         "width_ratio": round((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / (pt.ci_hi - pt.ci_lo), 2)})
        h2 = pd.read_csv(C.TABLES / "h2_paired_differences.csv"); h2 = h2[(h2["sample"] == "FRAME (primary)") & (h2.features == sname)]
        for mn in ("rf", "xgb"):
            dv = (done[mn] - done["logit_l2"]).to_numpy(); r0 = h2[h2.comparison.str.startswith(mn)].iloc[0]
            pair.append({"spec": sname, "comparison": f"{mn} - logit_l2", "point_current": r0.delta_auc,
                         "current_ci": f"[{r0.ci_lo:.3f}, {r0.ci_hi:.3f}]",
                         "pipeline_sd": round(dv.std(ddof=1), 3),
                         "pipeline_pct_ci": f"[{np.percentile(dv, 2.5):.3f}, {np.percentile(dv, 97.5):.3f}]",
                         "pipeline_sd_ci (point +- 1.96 sd)": f"[{r0.delta_auc - 1.96 * dv.std(ddof=1):.3f}, {r0.delta_auc + 1.96 * dv.std(ddof=1):.3f}]",
                         "share_replicates_with_diff>=0.03": round((dv >= 0.03).mean(), 3)})
    ps = pd.DataFrame(summ); pp = pd.DataFrame(pair); ps.to_csv(C.TABLES / "ci_pipeline_pooled.csv", index=False); pp.to_csv(C.TABLES / "ci_pipeline_paired.csv", index=False)
    print("\n== 2a. Whole-pipeline bootstrap (grouped folds), pooled AUC ==\n", ps.to_string(index=False))
    print("\n== 2b. Paired H2 differences ==\n", pp.to_string(index=False))

# ------------------------------------------------------------------------------------------ 3. LOCO refit bootstrap
def loco_replicate(num, cat, held, seed):
    rng = np.random.default_rng(seed); out = {}
    tr_all, te_all = np.flatnonzero((d.country != held).to_numpy()), np.flatnonzero((d.country == held).to_numpy())
    def res(ix):   # stratified resample of a set of firms by country x label
        return np.concatenate([rng.choice(ix[codes[ix] == c], (codes[ix] == c).sum()) for c in np.unique(codes[ix])])
    tr, te = res(tr_all), res(te_all)
    if y.iloc[te].nunique() < 2: return None
    for m, mod in make_models(num, cat).items():
        if m in MODELS: out[m] = roc_auc_score(y.iloc[te], clone(mod).fit(d[num + cat].iloc[tr], y.iloc[tr]).predict_proba(d[num + cat].iloc[te])[:, 1])
    return out

if "3" in SEC:
    rows = []
    for sname, num in (("neutral (original)", C.NEUTRAL), ("neutral, no macro", NEUT_NM)):
        cf = CACHE / f"loco_{sname.split()[0]}_{'orig' if 'original' in sname else 'nomacro'}.csv"
        done = pd.read_csv(cf) if cf.exists() else pd.DataFrame(columns=["seed", "held", *MODELS])
        while len(done) < B * 3:
            out_of_time(); s = len(done) // 3
            for h in COUNTRIES:
                r = loco_replicate(num, [], h, 5000 + s)
                done = pd.concat([done, pd.DataFrame([{"seed": s, "held": h, **(r or {m: np.nan for m in MODELS})}])], ignore_index=True)
            done.to_csv(cf, index=False)
        ml = C.TABLES / "macro_loco.csv"; ml = pd.read_csv(ml) if ml.exists() else None
        for mn in MODELS:
            n_ok = n_ok_c = 0
            for h in COUNTRIES:
                v = done[done.held == h][mn].dropna().to_numpy()
                if "original" in sname:
                    m0 = MAIN[(MAIN.features == "neutral") & (MAIN.model == mn) & (MAIN.analysis == "leave-one-country-out") & (MAIN.test_set == h)].iloc[0]
                    pt, cur = float(m0.auc), f"[{m0.ci_lo:.3f}, {m0.ci_hi:.3f}]"
                else:
                    pt, cur = (float(ml[(ml.spec == "neutral, no macro") & (ml.model == mn) & (ml.held_out == h)].auc.iloc[0]) if ml is not None else np.nan), ""
                lo, hi = np.percentile(v, [2.5, 97.5]); sd = v.std(ddof=1); clo = pt - 1.96 * sd
                n_ok += lo > 0.5; n_ok_c += clo > 0.5
                rows.append({"spec": sname, "model": mn, "held_out": h, "B": len(v), "point": pt, "current_ci": cur,
                             "refit_mean": round(v.mean(), 3), "refit_sd": round(sd, 3), "refit_pct_ci": f"[{lo:.3f}, {hi:.3f}]",
                             "refit_centered_ci (point +- 1.96 sd)": f"[{clo:.3f}, {pt + 1.96 * sd:.3f}]",
                             "lower>0.5 (pct)": bool(lo > 0.5), "lower>0.5 (centered)": bool(clo > 0.5)})
            rows.append({"spec": sname, "model": mn, "held_out": "CRITERION", "B": B,
                         "lower>0.5 (pct)": f"{n_ok} of 3 -> {'MET' if n_ok >= 2 else 'not met'}",
                         "lower>0.5 (centered)": f"{n_ok_c} of 3 -> {'MET' if n_ok_c >= 2 else 'not met'}"})
    lo = pd.DataFrame(rows); lo.to_csv(C.TABLES / "ci_loco_refit.csv", index=False)
    print("\n== 3. LOCO with refit bootstrap (train and test firms resampled) ==\n", lo.to_string(index=False))
