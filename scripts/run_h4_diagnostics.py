"""H4 diagnostics (review item 7): is the SHAP-filtered selection policy actually testable?

Findings motivating this script (from the committed run): the filter in simulate._xai_filter allowed every held-out
candidate in a 3-repeat check; the 'missing-value indicator' half cannot fire (XGBoost handles NaN natively, so no
`*_missing` columns exist); the simulation feature set also contains sector/business-model features and STEM; and each fold
selects only k = 3 firms.

What this script does (sample, folds, seeds and models identical to simulate.run; only extra bookkeeping):
  1  fit once per fold, store per-candidate SHAP mass by feature group (prestige / NaN-attributed / macro+sector / other)
  2  reproduction check against committed hit rates
  3  filter ACTIVITY at the plan's tau = 0.40 (how many candidates are excluded; how often do picks change)
  4  tau sensitivity grid for three filter definitions (plan-as-coded, plan with NaN attribution implemented,
     exploratory 'artifact' filter on macro + sector/business-model mass) -- a grid, NOT a search for the best tau
  5  firm-level bootstrap for policy differences (folds are not independent; the fold-level CI is optimistic)

Outputs: results/tables/h4_*.csv. Env: XAI_SIM_REPEATS (default 20), XAI_BUDGET, XAI_SECTIONS.
Specs: 'original' (as committed) and 'no macro' (item 3).
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd, shap
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame
from xai_capital.models import make_models
from xai_capital.simulate import PRESTIGE, _heuristic_score

SIM_REP = int(os.environ.get("XAI_SIM_REPEATS", 20)); SEC = os.environ.get("XAI_SECTIONS", "1,3,4,5").split(",")
T0 = time.time(); BUDGET = float(os.environ.get("XAI_BUDGET", 235))
CACHE = Path(".cache_h4"); CACHE.mkdir(exist_ok=True)
TGT = C.TARGETS["primary"]; MACRO = ["gdp_growth_at_founding", "inflation_at_founding"]
NEUT_NM = [c for c in C.NEUTRAL if c not in MACRO]
TAUS = [0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 30)

fr = in_frame(load_startups())
bd = fr[(fr.country == "Bangladesh") & fr[C.FOUNDER].notna().all(axis=1)]
db, yb = labelled(bd, TGT); db = db.reset_index(drop=True); yb = yb.reset_index(drop=True)
SPECS = {"original": C.NEUTRAL + C.SECTOR_BIZ + C.FOUNDER, "no macro": NEUT_NM + C.SECTOR_BIZ + C.FOUNDER}
CAT = C.CAT_A

# ------------------------------------------------------------------------------------------ 1. per-candidate store
def collect(spec):
    cf = CACHE / f"cand_{spec.replace(' ', '_')}_R{SIM_REP}.csv"
    if cf.exists(): return pd.read_csv(cf)
    if time.time() - T0 > BUDGET: print("[budget reached] rerun to resume"); sys.exit(0)
    num = SPECS[spec]; X = db[num + CAT]; models = make_models(num, CAT)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=SIM_REP, random_state=C.SEED)
    rows, rng = [], np.random.default_rng(C.SEED)
    for fold, (tr, te) in enumerate(rskf.split(X, yb)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]; k = max(1, int(round(0.3 * len(te))))
        logit = clone(models["logit_l2"]).fit(Xtr, yb.iloc[tr]); xgb = clone(models["xgb"]).fit(Xtr, yb.iloc[tr])
        p_log, p_xgb = logit.predict_proba(Xte)[:, 1], xgb.predict_proba(Xte)[:, 1]
        pre, clf = xgb.named_steps["pre"], xgb.named_steps["clf"]
        names = [c.split("__", 1)[-1] for c in pre.get_feature_names_out()]
        sv = np.abs(np.asarray(shap.TreeExplainer(clf).shap_values(pre.transform(Xte))))
        tot = np.maximum(sv.sum(axis=1), 1e-12)
        is_pre = np.array([n in PRESTIGE for n in names]); is_miss_col = np.array([n.endswith("_missing") for n in names])
        is_art = np.array([(n in MACRO) or (n in C.SECTOR_BIZ) or n.startswith("sector_group_") for n in names])
        nan_mask = np.zeros_like(sv, dtype=bool)                       # feature is NaN for that candidate
        for j, n in enumerate(names):
            if n in Xte.columns: nan_mask[:, j] = Xte[n].isna().to_numpy()
        heur = _heuristic_score(Xte); rnd = rng.permutation(len(te))[:k]
        for r, i in enumerate(te):
            rows.append({"fold": fold, "repeat": fold // 5, "firm": int(i), "y": int(yb.iloc[i]), "k": k,
                         "p_xgb": p_xgb[r], "p_log": p_log[r], "heur": heur[r], "is_random_pick": int(r in rnd),
                         "share_prestige": sv[r, is_pre].sum() / tot[r],
                         "share_missing_col": sv[r, is_miss_col].sum() / tot[r],       # what the committed code looks for
                         "share_nan_attr": sv[r, nan_mask[r]].sum() / tot[r],           # SHAP mass on features NaN for this firm
                         "share_artifact": sv[r, is_art].sum() / tot[r]})
    df = pd.DataFrame(rows); df.to_csv(cf, index=False); return df

cands = {s: collect(s) for s in SPECS}
print(f"Simulation sample: n={len(db)} Bangladesh complete-case firms (funded {int(yb.sum())}); features (original): {SPECS['original']} + {CAT}")
nan_cols = db[SPECS["original"]].isna().sum(); print("NaN counts in simulation features:", nan_cols[nan_cols > 0].to_dict() or "none")

# ------------------------------------------------------------------------------------------ policy machinery
def picks_for(fd, policy, filt=None, tau=None):
    """indices (positions inside the fold frame) selected by a policy -- identical argsort calls to simulate.run."""
    k = int(fd.k.iloc[0])
    if policy == "random": return np.flatnonzero(fd.is_random_pick.to_numpy())[:k]
    if policy == "heuristic": return np.argsort(-fd.heur.to_numpy())[:k]
    if policy == "econometric": return np.argsort(-fd.p_log.to_numpy())[:k]
    if policy == "black_box": return np.argsort(-fd.p_xgb.to_numpy())[:k]
    share = fd[filt].to_numpy() if isinstance(filt, str) else sum(fd[f].to_numpy() for f in filt)
    return np.argsort(-np.where(share <= tau, fd.p_xgb.to_numpy(), -1.0))[:k]

def evaluate(c, policy, filt=None, tau=None):
    hr, sel = [], np.zeros(len(db))
    for _, fd in c.groupby("fold", sort=True):
        fd = fd.reset_index(drop=True); idx = picks_for(fd, policy, filt, tau)
        hr.append(fd.y.to_numpy()[idx].mean()); np.add.at(sel, fd.firm.to_numpy()[idx], 1)
    return np.array(hr), sel / c.repeat.nunique()          # per-fold hit rates; per-firm selection frequency

def boot_fold(a, b, rng, B=2000):
    d = a - b; return d.mean(), *np.percentile([rng.choice(d, len(d)).mean() for _ in range(B)], [2.5, 97.5])

def boot_firm(fa, fb, B=2000, seed=42):
    """bootstrap over FIRMS; selections are fixed out-of-fold choices, outcome resampled with the firm."""
    rng, y, n = np.random.default_rng(seed), yb.to_numpy(), len(yb)
    est = lambda f, i: (f[i] * y[i]).sum() / max(f[i].sum(), 1e-12)
    dif = [est(fa, i) - est(fb, i) for i in (rng.integers(0, n, n) for _ in range(B))]
    return est(fa, np.arange(n)) - est(fb, np.arange(n)), *np.percentile(dif, [2.5, 97.5])

# ------------------------------------------------------------------------------------------ 2/3. reproduction + activity
if "1" in SEC or "3" in SEC:
    rows = []
    for spec, c in cands.items():
        base = {p: evaluate(c, p)[0].mean() for p in ("random", "heuristic", "econometric", "black_box")}
        xai = evaluate(c, "xai", "share_prestige", 0.40)[0].mean()
        print(f"\n== [{spec}] reproduction (committed: original bb .870 / econ .800 / heur .797 / rand .643 / xai .867; no macro bb .803 xai .803) ==")
        print({k: round(v, 3) for k, v in base.items()}, "xai", round(xai, 3))
        s = c.share_prestige
        picks_changed = []
        for _, fd in c.groupby("fold"):
            fd = fd.reset_index(drop=True)
            picks_changed.append(len(set(picks_for(fd, "black_box")) - set(picks_for(fd, "xai", "share_prestige", 0.40))))
        picks_changed = np.array(picks_changed)
        act = {"spec": spec, "n_candidate_rows": len(c), "n_folds": c.fold.nunique(),
               "prestige_share_median": round(s.median(), 3), "p90": round(s.quantile(.9), 3), "p99": round(s.quantile(.99), 3),
               "max": round(s.max(), 3), "pct_candidates_above_tau0.40": round(100 * (s > 0.40).mean(), 2),
               "folds_with_any_pick_changed": int((picks_changed > 0).sum()), "total_picks_changed": int(picks_changed.sum()),
               "total_picks": int(c.groupby("fold").k.first().sum()),
               "max_share_missing_col (committed code)": round(c.share_missing_col.max(), 3),
               "max_share_nan_attr": round(c.share_nan_attr.max(), 3), "median_share_artifact(macro+sector)": round(c.share_artifact.median(), 3)}
        rows.append(act)
    activity = pd.DataFrame(rows); activity.to_csv(C.TABLES / "h4_activity.csv", index=False)
    print("\n== 3. Filter activity at the plan's tau = 0.40 ==\n", activity.T.to_string(header=False))

# ------------------------------------------------------------------------------------------ 4. tau grid, 3 filter definitions
if "4" in SEC:
    FILTERS = {"plan as coded (prestige + *_missing cols)": ["share_prestige", "share_missing_col"],
               "plan with NaN attribution (prestige + NaN-feature mass)": ["share_prestige", "share_nan_attr"],
               "EXPLORATORY artifact filter (macro + sector/biz-model mass)": "share_artifact"}
    rng, rows = np.random.default_rng(42), []
    for spec, c in cands.items():
        hr_bb, f_bb = evaluate(c, "black_box")
        for fname, flt in FILTERS.items():
            for tau in TAUS:
                hr_x, f_x = evaluate(c, "xai", flt, tau)
                share = (c[flt] if isinstance(flt, str) else c[flt].sum(axis=1))
                changed = 0
                for _, fd in c.groupby("fold"):
                    fd = fd.reset_index(drop=True)
                    changed += int(len(set(picks_for(fd, "black_box")) - set(picks_for(fd, "xai", flt, tau))) > 0)
                d, lo, hi = boot_fold(hr_x, hr_bb, rng); fd_, flo, fhi = boot_firm(f_x, f_bb)
                rows.append({"spec": spec, "filter": fname, "tau": tau, "pct_candidates_excluded": round(100 * (share > tau).mean(), 1),
                             "folds_picks_changed_pct": round(100 * changed / c.fold.nunique(), 1),
                             "hit_xai": round(hr_x.mean(), 3), "hit_black_box": round(hr_bb.mean(), 3),
                             "diff_foldlevel": round(d, 3), "fl_lo": round(lo, 3), "fl_hi": round(hi, 3),
                             "diff_firmlevel": round(fd_, 3), "firm_lo": round(flo, 3), "firm_hi": round(fhi, 3)})
    grid = pd.DataFrame(rows); grid.to_csv(C.TABLES / "h4_tau_grid.csv", index=False)
    print("\n== 4. tau grid: XAI-filtered minus unfiltered XGBoost (fold-level CI as in the paper; firm-level CI is the honest one) ==")
    print(grid.to_string(index=False))

# ------------------------------------------------------------------------------------------ 5. firm-level CIs for the headline contrasts
if "5" in SEC:
    rows = []
    for spec, c in cands.items():
        fq = {p: evaluate(c, p)[1] for p in ("random", "heuristic", "econometric", "black_box")}
        fq["xai (plan, tau=0.40)"] = evaluate(c, "xai", "share_prestige", 0.40)[1]
        for a, b in (("black_box", "heuristic"), ("black_box", "econometric"), ("black_box", "random"),
                     ("xai (plan, tau=0.40)", "black_box"), ("heuristic", "random")):
            d, lo, hi = boot_firm(fq[a], fq[b]); rows.append({"spec": spec, "comparison": f"{a} - {b}",
                                                            "diff": round(d, 3), "firm_lo": round(lo, 3), "firm_hi": round(hi, 3)})
        hr = {p: round(float((fq[p] * yb.to_numpy()).sum() / fq[p].sum()), 3) for p in fq}
        print(f"\n[{spec}] pooled firm-level hit rates:", hr)
    fl = pd.DataFrame(rows); fl.to_csv(C.TABLES / "h4_firmlevel.csv", index=False)
    print("\n== 5. Firm-level bootstrap (resampling the 53 firms) ==\n", fl.to_string(index=False))
