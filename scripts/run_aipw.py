import sys; sys.path.insert(0, "src")
from xai_capital import config as C
from xai_capital.data import load_startups
from xai_capital.causal import aipw_ate

df = load_startups(only_independent=True)
d = df[df["accelerator_participated"].notna() & df[C.TARGETS["primary"]].notna()]
res = aipw_ate(d[C.NUM_A], d["accelerator_participated"], d[C.TARGETS["primary"]], n_boot=100)
print(res)
