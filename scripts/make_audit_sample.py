"""Create the second-coder audit sheet: all unfunded, all closures, random 20% of other added rows."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from xai_capital.data import load_startups
from xai_capital import config as C

df = load_startups(); num = df.startup_id.str[6:].astype(int)
added = df[num >= 101]
must = df[(df.any_external_funding == 0) | (df.event_closed == 1)]
rest = added[~added.startup_id.isin(must.startup_id)]
rand = rest.sample(frac=0.2, random_state=42)
a = pd.concat([must, rand]).drop_duplicates("startup_id").sort_values("startup_id")
a["audit_reason"] = np.where(a.startup_id.isin(must.startup_id), "unfunded_or_closed", "random_20pct_of_added")
cols = ["startup_id", "startup_name", "country", "founding_year", "entity_type", "sector_group", "any_external_funding",
        "event_closed", "survival_status", "status_confidence", "source_citation", "status_note", "audit_reason"]
a = a[cols].copy()
for c in ["auditor", "date_audited", "source_opened (Y/N)", "funding_label_confirmed (Y/N/Unclear)",
          "status_confirmed (Y/N/Unclear)", "founding_year_confirmed (Y/N)", "in_frame_confirmed (Y/N)",
          "correction_needed (Y/N)", "corrected_value", "comments"]:
    a[c] = ""
out = C.ROOT / "docs" / "audit"; out.mkdir(parents=True, exist_ok=True)
a.to_csv(out / "audit_sample.csv", index=False)
print(f"{len(a)} rows -> docs/audit/audit_sample.csv", a.audit_reason.value_counts().to_dict())
