"""Data audit: run this first, and after EVERY data update."""
import sys; sys.path.insert(0, "src")
import pandas as pd
from xai_capital import config as C
from xai_capital.data import load_startups

df = load_startups()
print(f"rows={len(df)}  unique ids={df['startup_id'].nunique()}")
assert df["startup_id"].is_unique, "duplicate startup_id"
for name, t in C.TARGETS.items():
    vc = df[t].value_counts(dropna=False).to_dict()
    print(f"target[{name}] {t}: {vc}")
print("closures (event_closed==1):", int((df['event_closed'] == 1).sum()))
print("unknown survival status   :", int(df['event_closed'].isna().sum()))
print("entity types:", df["entity_type"].value_counts().to_dict())
if df["possible_duplicate_of"].notna().any():
    print("WARNING possible duplicates:",
          df.loc[df["possible_duplicate_of"].notna(), ["startup_id", "possible_duplicate_of"]].values.tolist())
feats = set(C.NUM_A + C.CAT_A + C.EXTRA_B + C.EXTRA_C)
leaks = feats & set(C.FORBIDDEN)
assert not leaks, f"Forbidden columns used as features: {leaks}"
missing = (df[sorted(feats)].isna().mean() * 100).round(0).sort_values(ascending=False)
print("\n% missing per feature:\n", missing.head(12).to_string())
