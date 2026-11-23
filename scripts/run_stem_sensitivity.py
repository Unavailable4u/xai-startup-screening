"""STEM-exclusion sensitivity analysis (review item 2: Alvina, Sec. 5.5; assigned task from Emon).

Request: determine whether the reported founder-model AUC improvements (H1 / H3), odds ratios, SHAP ranks and
simulated selection are robust when the STEM-founder indicator is excluded. Sample (Bangladesh complete cases, n=53),
CV splits (5-fold x REP, same seed), model configurations and hyper-parameters are IDENTICAL to the original analysis;
the ONLY change is dropping `any_stem_founder` from the feature sets.

Sections (env XAI_SECTIONS, default all):
  1 descriptives: STEM 2x2, exact test, STEM-only AUC, STEM x source (links to item 1)
  2 CV performance and paired comparisons, original vs no-STEM  -> stem_cv_performance.csv, stem_paired.csv
  3 standardised odds ratios, original vs no-STEM               -> stem_odds_ratios.csv
  4 SHAP ranks, original vs no-STEM                             -> stem_shap.csv
  5 simulated selection (H4), original vs no-STEM               -> stem_sim_summary.csv, stem_sim_paired.csv
Env: XAI_REPEATS (default 20 = repeats behind the committed founder tables); XAI_SIM_REPEATS (default 20)
     Section 2 caches out-of-fold predictions in .cache_stem/ and can be resumed if it hits XAI_BUDGET seconds.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame, boot_auc, paired_boot_diff
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof
from xai_capital.explain import fit_and_shap, global_importance
from xai_capital.simulate import run as sim_run

SEC = os.environ.get("XAI_SECTIONS", "1,2,3,4,5").split(",")
REP = int(os.environ.get("XAI_REPEATS", 20))   # committed founder tables correspond to 20 repeats
SIM_REP = int(os.environ.get("XAI_SIM_REPEATS", 20))
T0 = time.time(); BUDGET = float(os.environ.get("XAI_BUDGET", 230))
from pathlib import Path
CACHE = Path(".cache_stem"); CACHE.mkdir(exist_ok=True)
TGT = C.TARGETS["primary"]; STEM = "any_stem_founder"
pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_columns", 30)
C.TABLES.mkdir(parents=True, exist_ok=True)

fr = in_frame(load_startups())
bd = fr[(fr.country == "Bangladesh") & fr[TGT].notna()]
d = bd[bd[C.FOUNDER].notna().all(axis=1)].reset_index(drop=True)
y = d[TGT].astype(int)
FOUNDER_NS = [c for c in C.FOUNDER if c != STEM]
FUND_NS = [c for c in C.FUNDAMENTALS if c != STEM]
PRES_ADD = [p for p in C.PRESTIGE if p not in C.FUNDAMENTALS]
print(f"Bangladesh complete cases: n={len(d)}, unfunded={int((y==0).sum())}, funded={int((y==1).sum())}")

# ------------------------------------------------------------------------------------------ 1. descriptives
if "1" in SEC:
    print("\n== 1. STEM x funding (complete cases) ==")
    tab = pd.crosstab(d[STEM], y).rename(index={0: "no STEM founder", 1: "STEM founder"},
                                         columns={0: "unfunded", 1: "funded"})
    print(tab.to_string())
    a, b = int(tab.loc["STEM founder", "funded"]), int(tab.loc["STEM founder", "unfunded"])
    c_, d_ = int(tab.loc["no STEM founder", "funded"]), int(tab.loc["no STEM founder", "unfunded"])
    _, p = fisher_exact([[a, b], [c_, d_]])
    haldane = ((a + .5) * (d_ + .5)) / ((b + .5) * (c_ + .5))      # OR of funding for STEM vs non-STEM, 0.5 correction
    se = np.sqrt(1/(a+.5) + 1/(b+.5) + 1/(c_+.5) + 1/(d_+.5))
    print(f"Fisher exact p = {p:.4f}. Zero cell: unfunded & no-STEM = {d_}.")
    print(f"Unadjusted OR(funded | STEM) with 0.5 correction = {haldane:.3f} "
          f"[{np.exp(np.log(haldane)-1.96*se):.3f}, {np.exp(np.log(haldane)+1.96*se):.3f}]  (exact OR is 0: complete separation in one cell)")
    rule = 1 - d[STEM].astype(float)                    # score = 'no STEM'
    a_, lo, hi = boot_auc(y, rule.values)
    print(f"STEM alone as a fixed rule (no fitting): AUC = {a_:.3f} [{lo:.3f}, {hi:.3f}]")
    print("\nSTEM x sector (complete cases):"); print(pd.crosstab(d.sector_group, [y, d[STEM]]).to_string())
    print("\nSource x funding x STEM (complete cases):")
    print(pd.crosstab(d.sample_source, [y, d[STEM]], margins=True).to_string())
    r = (1 - d["src_v3"]).astype(float).values
    a_, lo, hi = boot_auc(y, r)
    print(f"\nSource rule (v3 => unfunded) inside these 53 firms: AUC = {a_:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"(v3 firms = {int(d.src_v3.sum())} of {int((y==0).sum())} unfunded)")
    print("Funded firms: ", d[y == 1].sample_source.value_counts().to_dict())
    pd.DataFrame({"measure": ["fisher_p", "or_haldane", "stem_rule_auc", "source_rule_auc"],
                  "value": [p, haldane, boot_auc(y, rule.values)[0], boot_auc(y, r)[0]]}
                 ).round(4).to_csv(C.TABLES / "stem_descriptives.csv", index=False)

# ------------------------------------------------------------------------------------------ 2. CV performance
if "2" in SEC:
    sets = {   # name: (original features, no-STEM features)
        "controls only":           (C.NEUTRAL, C.NEUTRAL),
        "founder only":            (C.FOUNDER, FOUNDER_NS),
        "controls + founder":      (C.NEUTRAL + C.FOUNDER, C.NEUTRAL + FOUNDER_NS),
        "fundamentals":            (C.FUNDAMENTALS, FUND_NS),
        "fundamentals + prestige": (C.FUNDAMENTALS + PRES_ADD, FUND_NS + PRES_ADD),
        "STEM only":               ([STEM], [STEM]),
        "controls + STEM only":    (C.NEUTRAL + [STEM], C.NEUTRAL + [STEM]),
    }
    oof, rows = {}, []
    for nm, (orig, ns) in sets.items():
        for ver, feats in (("with STEM", orig), ("no STEM", ns)):
            if nm in ("STEM only", "controls + STEM only") and ver == "no STEM": continue
            for mn in ("logit_l2", "rf"):
                cf = CACHE / f"{nm.replace(' ', '_').replace('+', 'p')}__{ver.replace(' ', '_')}__{mn}__R{REP}.npy"
                if cf.exists():
                    o = np.load(cf)
                else:
                    if time.time() - T0 > BUDGET:
                        print(f"[budget reached] rerun the same command to resume ({len(list(CACHE.glob('*.npy')))} cached)"); sys.exit(0)
                    o = repeated_cv_oof(make_models(feats, [])[mn], d[feats], y, 5, REP).mean(axis=0); np.save(cf, o)
                oof[(nm, ver, mn)] = o
                a, lo, hi = boot_auc(y, o)
                rows.append({"feature_set": nm, "version": ver, "model": mn, "n_features": len(feats),
                             "auc": round(a, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    perf = pd.DataFrame(rows); perf.to_csv(C.TABLES / "stem_cv_performance.csv", index=False)
    print("\n== 2a. CV AUC, original vs no-STEM (same splits, same hyper-parameters) ==")
    print(perf.pivot_table(index="feature_set", columns=["model", "version"], values="auc", sort=False).to_string())
    old = pd.read_csv(C.TABLES / "founder_cv_performance.csv")
    chk = perf[perf.version == "with STEM"].merge(old, on=["feature_set", "model"], suffixes=("", "_committed"))
    print("\nReproduction check vs committed founder_cv_performance.csv (max abs AUC difference): "
          f"{(chk.auc - chk.auc_committed).abs().max():.3f}")

    cmp = []
    def add(label, A, B, mn):
        dl, lo, hi = paired_boot_diff(y, oof[(A[0], A[1], mn)], oof[(B[0], B[1], mn)])
        cmp.append({"comparison": label, "model": mn, "delta_auc": round(dl, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    for mn in ("logit_l2", "rf"):
        add("H1 (with STEM): controls+founder - controls", ("controls + founder", "with STEM"), ("controls only", "with STEM"), mn)
        add("H1 (no STEM):   controls+founder - controls", ("controls + founder", "no STEM"), ("controls only", "no STEM"), mn)
        add("H3 (with STEM): fundamentals+prestige - fundamentals", ("fundamentals + prestige", "with STEM"), ("fundamentals", "with STEM"), mn)
        add("H3 (no STEM):   fundamentals+prestige - fundamentals", ("fundamentals + prestige", "no STEM"), ("fundamentals", "no STEM"), mn)
        add("STEM's share of H1: [controls+STEM only] - controls", ("controls + STEM only", "with STEM"), ("controls only", "with STEM"), mn)
        add("STEM contribution: controls+founder (with) - (no STEM)", ("controls + founder", "with STEM"), ("controls + founder", "no STEM"), mn)
        add("STEM contribution: founder only (with) - (no STEM)", ("founder only", "with STEM"), ("founder only", "no STEM"), mn)
        add("STEM contribution: fundamentals (with) - (no STEM)", ("fundamentals", "with STEM"), ("fundamentals", "no STEM"), mn)
    cmp = pd.DataFrame(cmp); cmp.to_csv(C.TABLES / "stem_paired.csv", index=False)
    print("\n== 2b. Paired AUC differences (95% bootstrap CI) ==\n", cmp.to_string(index=False))

# ------------------------------------------------------------------------------------------ 3. odds ratios
if "3" in SEC:
    def odds(feats, label):
        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(d[feats]), columns=feats)
        X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
        rng = np.random.default_rng(42); co = []
        for _ in range(500):
            i = rng.integers(0, len(y), len(y))
            if y.iloc[i].nunique() == 2:
                co.append(LogisticRegression(C=0.5, max_iter=2000).fit(X.iloc[i], y.iloc[i]).coef_[0])
        co = np.array(co); full = LogisticRegression(C=0.5, max_iter=2000).fit(X, y).coef_[0]
        return pd.DataFrame({"version": label, "feature": feats, "odds_ratio_per_SD": np.exp(full).round(2),
                             "ci_lo": np.exp(np.percentile(co, 2.5, axis=0)).round(2),
                             "ci_hi": np.exp(np.percentile(co, 97.5, axis=0)).round(2)})
    orr = pd.concat([odds(C.FOUNDER + ["age_years", "product_stage"], "with STEM"),
                     odds(FOUNDER_NS + ["age_years", "product_stage"], "no STEM")])
    orr.to_csv(C.TABLES / "stem_odds_ratios.csv", index=False)
    print("\n== 3. Standardised odds ratios (L2 logistic C=0.5, 500 bootstrap draws) ==\n", orr.to_string(index=False))
    print("\ncorr(STEM, education) =", round(d[[STEM, 'edu_max_level']].corr().iloc[0, 1], 2),
          "| corr(STEM, industry_exp) =", round(d[[STEM, 'industry_exp_max_years']].corr().iloc[0, 1], 2))

# ------------------------------------------------------------------------------------------ 4. SHAP
if "4" in SEC:
    out = []
    for ver, feats in (("with STEM", C.NEUTRAL + C.FOUNDER), ("no STEM", C.NEUTRAL + FOUNDER_NS)):
        _, sv = fit_and_shap(make_models(feats, [])["xgb"], d[feats], y)
        imp = global_importance(sv)
        out.append(pd.DataFrame({"version": ver, "rank": range(1, len(imp) + 1), "feature": imp.index,
                                 "mean_abs_shap": imp.values.round(3)}))
    sh = pd.concat(out); sh.to_csv(C.TABLES / "stem_shap.csv", index=False)
    print("\n== 4. XGBoost SHAP ranks, Bangladesh founder rows ==\n",
          sh.pivot(index="rank", columns="version", values=["feature", "mean_abs_shap"]).to_string())

# ------------------------------------------------------------------------------------------ 5. simulation
if "5" in SEC:
    res_all, summ = [], []
    for ver, founder in (("with STEM", C.FOUNDER), ("no STEM", FOUNDER_NS)):
        num, cat = C.NEUTRAL + C.SECTOR_BIZ + founder, C.CAT_A
        cf = CACHE / f"sim_{ver.replace(' ', '_')}_R{SIM_REP}.csv"
        if cf.exists():
            r = pd.read_csv(cf)
        else:
            r = sim_run(make_models(num, cat), d[num + cat], y, outcome=y, k_frac=0.3, tau=0.4, n_repeats=SIM_REP)
            r.to_csv(cf, index=False)
            if ver == "with STEM" and os.environ.get("XAI_SIM_ONE", "0") == "1":
                print("[cached 'with STEM'; rerun to do 'no STEM']"); sys.exit(0)
        piv = r.pivot_table(index="fold", columns="investor", values="hit_rate")
        s = piv.agg(["mean", "std"]).T.round(3); s["version"] = ver; summ.append(s.reset_index())
        rng = np.random.default_rng(42)
        for a, b in (("xai", "black_box"), ("xai", "heuristic"), ("xai", "econometric"), ("black_box", "heuristic"),
                     ("black_box", "econometric"), ("xai", "random")):
            diff = (piv[a] - piv[b]).to_numpy(); bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            res_all.append({"version": ver, "comparison": f"{a} - {b}", "mean_diff": round(diff.mean(), 3),
                            "ci_lo": round(np.percentile(bs, 2.5), 3), "ci_hi": round(np.percentile(bs, 97.5), 3)})
    summ = pd.concat(summ); pd.DataFrame(res_all).to_csv(C.TABLES / "stem_sim_paired.csv", index=False)
    summ.to_csv(C.TABLES / "stem_sim_summary.csv", index=False)
    print("\n== 5. Simulated selection hit rates (top-30%%, base rate %.2f) ==\n" % y.mean(),
          summ.pivot(index="investor", columns="version", values="mean").to_string())
    print("\nPaired differences:\n", pd.DataFrame(res_all).to_string(index=False))
