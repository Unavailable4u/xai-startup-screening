import sys; sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, shap
from xai_capital import config as C
from xai_capital.data import load_startups, labelled
from xai_capital.models import make_models
from xai_capital.explain import fit_and_shap, global_importance

df = load_startups()
num, cat = C.NUM_A + C.EXTRA_B, C.CAT_A
d, y = labelled(df, C.TARGETS["primary"])
Xt, S = fit_and_shap(make_models(num, cat)["xgb"], d[num + cat], y)
imp = global_importance(S)
C.TABLES.mkdir(parents=True, exist_ok=True); C.FIGS.mkdir(parents=True, exist_ok=True)
imp.to_csv(C.TABLES / "shap_global_importance.csv")
print(imp.head(10))
shap.summary_plot(S.values, Xt, show=False, max_display=12)
plt.tight_layout(); plt.savefig(C.FIGS / "shap_summary.png", dpi=300); plt.close()
