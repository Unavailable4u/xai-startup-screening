"""Check that numbers reported in the manuscript match the data and the committed result tables (review item 9).

Claims below are the values that should appear in the FINAL manuscript. Where the preprint v1 text differed from the
data, the old value is noted in a comment (these were the mismatches found in review). Run:  make check
Exit status 1 if any claim fails. Writes results/tables/manuscript_number_check.csv.
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from xai_capital import config as C
from xai_capital.data import load_startups, in_frame

T = C.TARGETS["primary"]; R = C.TABLES
raw = load_startups(); fr = in_frame(raw); d = fr[fr[T].notna()]
rows = []

def chk(section, what, claimed, actual, tol=0.0):
    ok = bool(abs(float(claimed) - float(actual)) <= tol + 1e-9)
    rows.append({"section": section, "claim": what, "manuscript": claimed, "computed": round(float(actual), 4),
                 "status": "PASS" if ok else "FAIL"})

def r2(section, what, claimed, actual): chk(section, what, claimed, actual, tol=0.0051)   # reported to 2 decimals

# ------------------------------------------------------------------ Section 3: sample and data counts
chk("3.1", "master dataset entries", 225, len(raw))
chk("3.2", "entries with a binary funding label", 211, raw[T].notna().sum())
chk("3.2", "primary analytic sample", 188, len(d))
for c, n in (("Bangladesh", 89), ("India", 58), ("Pakistan", 41)): chk("3.2", f"{c} startups", n, (d.country == c).sum())
chk("3.2", "funded", 129, d[T].sum()); chk("3.2", "no disclosed funding", 59, (d[T] == 0).sum())
for c, f, u, cl in (("Bangladesh", 66, 23, 2), ("India", 38, 20, 16), ("Pakistan", 25, 16, 10)):   # Table 1
    s = d[d.country == c]
    chk("T1", f"{c} funded", f, s[T].sum()); chk("T1", f"{c} unfunded", u, (s[T] == 0).sum())
    chk("T1", f"{c} closures", cl, (s.event_closed == 1).sum())
chk("3.6", "closures in India + Pakistan", 26, ((d.event_closed == 1) & d.country.isin(["India", "Pakistan"])).sum())
chk("3.6", "entries with no recorded accelerator participation (of 225)", 143, raw.accelerator_participated.isna().sum())
chk("3.6", "entries missing highest industry experience (of 225)", 115, raw.industry_exp_max_years.isna().sum())
chk("3.6", "entries missing number of founders (of 225)", 22, raw.n_founders.isna().sum())   # v1 said 117: no such column; 22 for n_founders
chk("3.6", "entries missing education level (of 225)", 93, raw.edu_max_level.isna().sum())
chk("3.6", "entries missing STEM indicator (of 225)", 99, raw.any_stem_founder.isna().sum())
chk("3.6", "entries missing serial-entrepreneur indicator (of 225)", 114, raw.serial_entrepreneur.isna().sum())
chk("3.3", "minimum age of unfunded firms >= 36 months (years)", 3.0, min(3.0, d[d[T] == 0].age_years.min()))
for s, pct in (("IT/SaaS/AI/IoT", 74), ("Other", 59), ("Media", 50), ("AgriTech", 43), ("EdTech", 38),
               ("HealthTech", 14), ("Logistics/Travel", 9), ("Fintech", 8), ("E-commerce", 6)):          # Figure 1
    g = d[d.sector_group == s]; chk("F1", f"{s} % unfunded", pct, 100 * (g[T] == 0).mean(), tol=0.51)
# founder completeness: ALL four founder fields present.  v1 text: India 16/20 and 4/38, Pakistan 1/25 (looser rule)
ok = d[C.FOUNDER].notna().all(axis=1)
for c, u, un, f, fn in (("Bangladesh", 20, 23, 33, 66), ("India", 15, 20, 0, 38), ("Pakistan", 12, 16, 0, 25)):
    s = d.country == c
    chk("3.6", f"{c} unfunded with complete founder data (of {un})", u, ok[s & (d[T] == 0)].sum())
    chk("3.6", f"{c} funded with complete founder data (of {fn})", f, ok[s & (d[T] == 1)].sum())
bd = d[(d.country == "Bangladesh") & ok]
chk("5.5", "Bangladesh complete cases", 53, len(bd)); chk("5.5", "  funded", 33, bd[T].sum()); chk("5.5", "  unfunded", 20, (bd[T] == 0).sum())
r2("5.5", "mean education level, funded", 1.79, bd[bd[T] == 1].edu_max_level.mean())
r2("5.5", "mean education level, unfunded", 1.25, bd[bd[T] == 0].edu_max_level.mean())
chk("5.5", "mean industry experience, funded (yrs)", 12.5, bd[bd[T] == 1].industry_exp_max_years.mean(), tol=0.051)   # v1 text said 12.6
chk("5.5", "mean industry experience, unfunded (yrs)", 8.2, bd[bd[T] == 0].industry_exp_max_years.mean(), tol=0.051)
chk("5.5", "share of funded complete cases with STEM founder", 0.67, bd[bd[T] == 1].any_stem_founder.mean(), tol=0.005)
chk("5.5", "unfunded complete cases with STEM founder (share)", 1.0, bd[bd[T] == 0].any_stem_founder.mean())
chk("5.7", "sample financing proportion", 0.62, bd[T].mean(), tol=0.005)

# ------------------------------------------------------------------ Table 2, 5.2, 5.3
M = pd.read_csv(R / "main_analysis.csv"); F = M[M["sample"] == "FRAME (primary)"]
def pooled(f, m): return F[(F.features == f) & (F.model == m) & (F.analysis == "pooled + country control") & (F.test_set == "all")].iloc[0]
def loco(f, m, c): return F[(F.features == f) & (F.model == m) & (F.analysis == "leave-one-country-out") & (F.test_set == c)].iloc[0]
T2 = {("neutral", "logit_l2"): (.58, .48, .66, .58, .56, .69), ("neutral", "rf"): (.65, .56, .73, .64, .57, .72),
      ("neutral", "xgb"): (.62, .53, .72, .61, .44, .73), ("neutral+sector", "logit_l2"): (.83, .76, .90, .61, .88, .93),
      ("neutral+sector", "rf"): (.85, .79, .90, .63, .84, .98), ("neutral+sector", "xgb"): (.84, .78, .90, .59, .81, .98),
      ("sector_only", "logit_l2"): (.85, .79, .91, .62, .95, .97), ("sector_only", "rf"): (.84, .78, .90, .63, .89, .97),
      ("sector_only", "xgb"): (.86, .79, .91, .63, .93, .98)}
for (f, m), (a, lo, hi, bgd, ind, pak) in T2.items():
    p = pooled(f, m); r2("T2", f"{f}/{m} pooled AUC", a, p.auc); r2("T2", f"{f}/{m} CI lo", lo, p.ci_lo); r2("T2", f"{f}/{m} CI hi", hi, p.ci_hi)
    for c, v in (("Bangladesh", bgd), ("India", ind), ("Pakistan", pak)): r2("T2", f"{f}/{m} LOCO {c}", v, loco(f, m, c).auc)
rf_lo = {c: loco("neutral", "rf", c).ci_lo for c in ("Bangladesh", "Pakistan")}
r2("5.3", "RF neutral LOCO lower bound, Bangladesh", 0.52, rf_lo["Bangladesh"]); r2("5.3", "RF neutral LOCO lower bound, Pakistan", 0.53, rf_lo["Pakistan"])
for m, n in (("logit_l2", 0), ("rf", 2), ("xgb", 1)):
    chk("5.3", f"neutral {m}: countries with LOCO lower bound > 0.50 (only RF has >= 2)",
        n, sum(loco("neutral", m, c).ci_lo > 0.5 for c in ("Bangladesh", "India", "Pakistan")))
A = M[(M["sample"] == "ALL ROWS (sensitivity)") & (M.analysis == "pooled + country control") & (M.test_set == "all") & (M.features == "neutral")].set_index("model").auc
for m, v in (("logit_l2", .65), ("rf", .70), ("xgb", .67)): r2("5.2", f"211-firm sensitivity neutral {m}", v, A[m])
H2 = pd.read_csv(R / "h2_paired_differences.csv"); H2 = H2[H2["sample"] == "FRAME (primary)"]
T3 = {("neutral", "rf"): (.07, -.02, .16), ("neutral", "xgb"): (.05, -.04, .13), ("neutral+sector", "rf"): (.02, -.02, .06),
      ("neutral+sector", "xgb"): (.01, -.02, .05), ("sector_only", "rf"): (-.01, -.04, .02), ("sector_only", "xgb"): (.00, -.02, .03)}
for (f, m), (a, lo, hi) in T3.items():
    r = H2[(H2.features == f) & (H2.comparison.str.startswith(m))].iloc[0]
    r2("T3", f"{f} {m}-logit", a, r.delta_auc); r2("T3", f"{f} {m}-logit CI lo", lo, r.ci_lo); r2("T3", f"{f} {m}-logit CI hi", hi, r.ci_hi)

# ------------------------------------------------------------------ Table 4, 5.5
FC = pd.read_csv(R / "founder_cv_performance.csv").set_index(["feature_set", "model"])
T4 = {"controls only": ((.65, .47, .80), (.70, .55, .85)), "founder only": ((.79, .65, .91), (.75, .58, .88)),
      "controls + founder": ((.79, .64, .91), (.82, .68, .93)), "fundamentals": ((.76, .60, .90), (.79, .64, .92)),
      "fundamentals + prestige": ((.78, .61, .91), (.80, .64, .93))}
for fs, (lg, rf) in T4.items():
    for m, (a, lo, hi) in (("logit_l2", lg), ("rf", rf)):
        r = FC.loc[(fs, m)]; r2("T4", f"{fs}/{m}", a, r.auc); r2("T4", f"{fs}/{m} CI lo", lo, r.ci_lo); r2("T4", f"{fs}/{m} CI hi", hi, r.ci_hi)
HP = pd.read_csv(R / "h1_h3_paired.csv")
for h, m, (a, lo, hi) in (("H1", "logit_l2", (.145, -.012, .302)), ("H1", "rf", (.112, -.005, .247)),
                          ("H3", "logit_l2", (.012, -.063, .091)), ("H3", "rf", (.002, -.046, .051))):
    r = HP[HP.hypothesis.str.startswith(h) & (HP.model == m)].iloc[0]
    chk("5.5", f"{h} {m} delta AUC", a, r.delta_auc, 0.0006); chk("5.5", f"{h} {m} CI lo", lo, r.ci_lo, 0.0006); chk("5.5", f"{h} {m} CI hi", hi, r.ci_hi, 0.0006)
OR = pd.read_csv(R / "founder_odds_ratios.csv").set_index("feature")
for f, (a, lo, hi) in {"edu_max_level": (2.19, 1.34, 3.80), "serial_entrepreneur": (1.44, .86, 2.32),
                       "industry_exp_max_years": (1.05, .62, 2.35), "any_stem_founder": (.32, .24, .47)}.items():
    r = OR.loc[f]; chk("5.5", f"OR {f}", a, r.odds_ratio_per_SD, .0051); chk("5.5", f"OR {f} CI lo", lo, r.ci_lo, .0051); chk("5.5", f"OR {f} CI hi", hi, r.ci_hi, .0051)

# ------------------------------------------------------------------ 5.6 SHAP, 5.7 simulation
def shap(f): return pd.read_csv(R / f, index_col=0).iloc[:, 0]
sn, ss, sb = shap("shap_pooled_neutral.csv"), shap("shap_pooled_neutral_sector.csv"), shap("shap_bangladesh_founder.csv")
for f, v in (("n_founders", .51), ("gdp_growth_at_founding", .40), ("inflation_at_founding", .20), ("age_years", .13)): r2("5.6", f"neutral SHAP {f}", v, sn[f])
r2("5.6", "neutral+sector SHAP subscription/SaaS", .58, ss["is_subscription_saas"]); r2("5.6", "neutral+sector SHAP IT sector", .55, ss["sector_group_IT/SaaS/AI/IoT"])
for f, v in (("n_founders", .40), ("gdp_growth_at_founding", .64), ("industry_exp_max_years", .65), ("edu_max_level", .26), ("serial_entrepreneur", .03)): r2("5.6", f"BD founder SHAP {f}", v, sb[f])
chk("5.6", "education rank in BD founder SHAP (of 7 features)", 4, list(sb.sort_values(ascending=False).index).index("edu_max_level") + 1)
chk("5.6", "serial entrepreneur rank (of 7)", 7, list(sb.sort_values(ascending=False).index).index("serial_entrepreneur") + 1)
S = pd.read_csv(R / "sim_summary.csv").set_index("investor")["mean"]
for k, v in (("black_box", .87), ("xai", .867), ("econometric", .80), ("heuristic", .797), ("random", .643)): chk("5.7", f"hit rate {k}", v, S[k], .0051)
SP = pd.read_csv(R / "sim_paired.csv").set_index("comparison")
for k, (a, lo, hi) in {"xai - black_box": (-.003, -.010, 0.0), "xai - heuristic": (.070, .020, .120), "xai - econometric": (.067, .027, .103),
                       "black_box - heuristic": (.073, .020, .127), "xai - random": (.223, .167, .280)}.items():
    r = SP.loc[k]; chk("5.7", f"{k}", a, r.mean_diff, .0006); chk("5.7", f"{k} CI lo", lo, r.ci_lo, .0006); chk("5.7", f"{k} CI hi", hi, r.ci_hi, .0006)

# ------------------------------------------------------------------ numbers introduced by items 1-3 and 7 (traceability)
def opt(f): 
    p = R / f; return pd.read_csv(p) if p.exists() else None
b = opt("source_baselines.csv")
if b is not None:
    r = b[b.model == "rule"].iloc[0]; r2("item1", "source flag fixed-rule AUC", .85, r.auc); chk("item1", "source rule AUC (3 dp)", .847, r.auc, .0006)
sd = opt("stem_descriptives.csv")
if sd is not None:
    v = sd.set_index("measure").value; chk("item2", "STEM alone fixed-rule AUC", .667, v.stem_rule_auc, .0006)
    chk("item2", "source rule AUC inside the 53 complete cases", .850, v.source_rule_auc, .0006); chk("item2", "Fisher exact p (STEM x funding)", .0038, v.fisher_p, .00006)
mp = opt("macro_pooled.csv")
if mp is not None:
    for m, v in (("logit_l2", .613), ("rf", .655), ("xgb", .636)):
        chk("item3", f"neutral no-macro pooled AUC {m}", v, mp[(mp.spec == "neutral, no macro") & (mp.model == m)].auc.iloc[0], .0006)
ha = opt("h4_activity.csv")
if ha is not None:
    chk("item7", "picks changed by SHAP filter (original, of 300)", 1, ha.total_picks_changed.iloc[0]); chk("item7", "picks changed (no macro)", 0, ha.total_picks_changed.iloc[1])

out = pd.DataFrame(rows); out.to_csv(R / "manuscript_number_check.csv", index=False)
bad = out[out.status == "FAIL"]
print(f"{len(out)} claims checked: {len(out) - len(bad)} pass, {len(bad)} fail")
if len(bad): print(bad.to_string(index=False)); sys.exit(1)
