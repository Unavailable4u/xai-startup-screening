import sys; sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from xai_capital import config as C
from xai_capital.data import load_startups

df = load_startups()
d = df[(df["event_closed"].notna()) & (df["duration_months"].notna())]
print("n =", len(d), "| closures =", int(d["event_closed"].sum()))
km = KaplanMeierFitter().fit(d["duration_months"], d["event_closed"], label="All startups")
ax = km.plot_survival_function()
ax.set_xlabel("Months since founding"); ax.set_ylabel("Survival probability")
C.FIGS.mkdir(parents=True, exist_ok=True)
plt.tight_layout(); plt.savefig(C.FIGS / "km_all.png", dpi=300)
print(km.survival_function_.tail(2))
