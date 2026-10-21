"""Four-investor simulation (H4). Runs ONLY on Bangladesh founder-complete rows (prestige needs founder data).
Exploratory: ~50 rows. Hit rate = share of picks that obtained external funding."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame
from xai_capital.models import make_models
from xai_capital.simulate import run

REP = int(os.environ.get("XAI_REPEATS", 20)); T = C.TARGETS["primary"]
bd = in_frame(load_startups()); bd = bd[(bd.country == "Bangladesh") & bd[C.FOUNDER].notna().all(axis=1)]
num, cat = C.NEUTRAL + C.SECTOR_BIZ + C.FOUNDER, C.CAT_A
d, y = labelled(bd, T); d = d.reset_index(drop=True); y = y.reset_index(drop=True)
print(f"n={len(d)} unfunded={int((y==0).sum())} base funded rate={y.mean():.2f}")
res = run(make_models(num, cat), d[num + cat], y, outcome=y, k_frac=0.3, tau=0.4, n_repeats=REP)
res.to_csv(C.TABLES / "sim_raw.csv", index=False)
piv = res.pivot_table(index="fold", columns="investor", values="hit_rate")
summ = piv.agg(["mean", "std"]).T.round(3); print(summ.to_string())
rng = np.random.default_rng(42); out = []
for a, b in (("xai", "black_box"), ("xai", "heuristic"), ("xai", "econometric"), ("black_box", "heuristic"), ("xai", "random")):
    diff = (piv[a] - piv[b]).to_numpy(); bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    out.append({"comparison": f"{a} - {b}", "mean_diff": round(diff.mean(), 3), "ci_lo": round(np.percentile(bs, 2.5), 3), "ci_hi": round(np.percentile(bs, 97.5), 3)})
out = pd.DataFrame(out); print("\nPaired differences (folds are correlated across repeats; CIs are optimistic):\n", out.to_string(index=False))
out.to_csv(C.TABLES / "sim_paired.csv", index=False); summ.to_csv(C.TABLES / "sim_summary.csv")
