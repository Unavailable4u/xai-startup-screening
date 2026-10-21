"""After the second coder fills docs/audit/audit_sample.csv: error rate with Wilson 95% CI."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from xai_capital import config as C
a = pd.read_csv(C.ROOT / "docs" / "audit" / "audit_sample.csv")
done = a[a["correction_needed (Y/N)"].astype(str).str.upper().isin(["Y", "N"])]
n, k = len(done), int((done["correction_needed (Y/N)"].str.upper() == "Y").sum())
print(f"audited {n} of {len(a)} rows; corrections needed: {k}")
if n:
    p = k / n; z = 1.96; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    print(f"error rate {p:.1%} (Wilson 95% CI {c-h:.1%} to {c+h:.1%})")
    print("RULE: if the error rate exceeds 10%, audit ALL added rows." if p > 0.10 else "Below the 10% threshold.")
    print(done.groupby("audit_reason")["correction_needed (Y/N)"].apply(lambda s: (s.str.upper() == "Y").mean()).round(2).to_string())
