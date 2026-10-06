import os, sys; sys.path.insert(0, "src")
import pandas as pd
from xai_capital import config as C
from xai_capital.data import load_startups, labelled
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof, summarize

df = load_startups()
sets = {"A": (C.NUM_A, C.CAT_A),
        "B": (C.NUM_A + C.EXTRA_B, C.CAT_A),
        "C_diagnostic": (C.NUM_A + C.EXTRA_B + C.EXTRA_C, C.CAT_A)}
rows = []
for tname, target in C.TARGETS.items():
    d, y = labelled(df, target)
    for sname, (num, cat) in sets.items():
        X = d[num + cat]
        for mname, model in make_models(num, cat).items():
            oof = repeated_cv_oof(model, X, y, n_splits=5, n_repeats=int(os.environ.get("XAI_REPEATS", 20)))
            rows.append({"target": target, "featset": sname, "model": mname,
                         "n": len(y), **summarize(oof, y)})
res = pd.DataFrame(rows).round(3)
C.TABLES.mkdir(parents=True, exist_ok=True)
res.to_csv(C.TABLES / "pilot_cv_results.csv", index=False)
print(res.to_string())
