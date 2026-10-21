"""SHAP: (a) pooled frame, two feature groups; (b) Bangladesh complete cases incl. founder attributes (H3 rank)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, shap, pandas as pd
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame
from xai_capital.models import make_models
from xai_capital.explain import fit_and_shap, global_importance

T = C.TARGETS["primary"]; fr = in_frame(load_startups())
C.TABLES.mkdir(parents=True, exist_ok=True); C.FIGS.mkdir(parents=True, exist_ok=True)

def run(tag, data, num, cat):
    d, y = labelled(data, T)
    Xt, S = fit_and_shap(make_models(num, cat)["xgb"], d[num + cat], y)
    imp = global_importance(S); imp.to_csv(C.TABLES / f"shap_{tag}.csv"); print(f"\n[{tag}] n={len(d)} unfunded={int((y==0).sum())}\n", imp.head(8).round(3).to_string())
    shap.summary_plot(S.values, Xt, show=False, max_display=12); plt.tight_layout()
    plt.savefig(C.FIGS / f"shap_{tag}.png", dpi=300); plt.close()
    return imp

for grp in ("neutral", "neutral+sector"):
    num, cat = C.FEATURE_GROUPS[grp]; run(f"pooled_{grp.replace('+','_')}", fr, num + C.COUNTRY_DUMMIES, cat)
bd = fr[(fr.country == "Bangladesh") & fr[C.FOUNDER].notna().all(axis=1)]
imp = run("bangladesh_founder", bd, C.NEUTRAL + C.FOUNDER, [])
rank = {f: i + 1 for i, f in enumerate(imp.index)}; half = len(imp) / 2
print("\nH3 check: prestige rank (of %d):" % len(imp), {p: rank.get(p) for p in C.PRESTIGE}, "| bottom half =", [p for p in C.PRESTIGE if rank.get(p, 0) > half])
