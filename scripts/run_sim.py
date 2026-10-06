import os, sys; sys.path.insert(0, "src")
from xai_capital import config as C
from xai_capital.data import load_startups, labelled
from xai_capital.models import make_models
from xai_capital.simulate import run

df = load_startups()
num, cat = C.NUM_A + C.EXTRA_B, C.CAT_A
d, y = labelled(df, C.TARGETS["primary"])
res = run(make_models(num, cat), d[num + cat], y, outcome=y, n_repeats=int(os.environ.get("XAI_REPEATS", 5)))
C.TABLES.mkdir(parents=True, exist_ok=True)
res.to_csv(C.TABLES / "sim_raw.csv", index=False)
print(res.groupby("investor")["hit_rate"].agg(["mean", "std"]).round(3))
