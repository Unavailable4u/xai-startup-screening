"""Descriptive survival: Kaplan-Meier by country + exploratory country-stratified Cox (<=2 covariates)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from xai_capital import config as C
from xai_capital.data import load_startups, in_frame

fr = in_frame(load_startups())
d = fr[fr.event_closed.notna() & fr.duration_months.notna() & fr.any_external_funding.notna()].copy()
print("n =", len(d), "| closures =", int(d.event_closed.sum())); print(d.groupby("country").event_closed.agg(["size", "sum"]).to_string())
ax = plt.subplot(111)
for c, g in d.groupby("country"):
    KaplanMeierFitter().fit(g.duration_months, g.event_closed, label=f"{c} (n={len(g)})").plot_survival_function(ax=ax, ci_show=False)
ax.set_xlabel("Months since founding"); ax.set_ylabel("Survival probability"); plt.tight_layout()
C.FIGS.mkdir(parents=True, exist_ok=True); plt.savefig(C.FIGS / "km_by_country.png", dpi=300)
lr = multivariate_logrank_test(d.duration_months, d.country, d.event_closed); print("log-rank p =", round(lr.p_value, 4))
cox = d[["duration_months", "event_closed", "country", "any_external_funding", "is_b2b"]].dropna()
cph = CoxPHFitter(penalizer=0.1).fit(cox, "duration_months", "event_col" if False else "event_closed", strata=["country"])
out = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3)
print("\nEXPLORATORY stratified Cox (immortal-time caveat: funding status is not a baseline covariate):\n", out.to_string())
out.to_csv(C.TABLES / "cox_exploratory.csv")
