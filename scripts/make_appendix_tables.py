"""Appendix Tables 6, 8 and 9 (cited in Sec. 5.2/5.3 and the Table 2 note but missing from preprint v1).
Built only from results/tables/main_analysis.csv.  Writes results/appendix_tables.md.
  Table 6: leave-one-country-out AUC with 95% CI by held-out country (primary sample)
  Table 8: within-country cross-validated AUC with 95% CI (primary sample)
  Table 9: pooled AUC, primary sample (n=188) vs all rows with a funding label (n=211)
"""
import sys; sys.path.insert(0, "src")
import pandas as pd
from xai_capital import config as C
M = pd.read_csv(C.TABLES / "main_analysis.csv")
LAB = {"neutral": "Neutral", "neutral+sector": "Neutral + sector", "sector_only": "Sector only"}
MOD = {"logit_l2": "Logistic", "rf": "Random Forest", "xgb": "XGBoost"}
cell = lambda r: f"{r.auc:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
P = M[M["sample"] == "FRAME (primary)"]; S = M[M["sample"] == "ALL ROWS (sensitivity)"]
out = []
def table(title, note, rows, header):
    out.append(f"**{title}**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    out.extend(rows); out.append(f"\n*Note.* {note}\n")
for ana, title, note in (("leave-one-country-out", "Appendix Table 6. Leave-one-country-out AUC by held-out country [95% CI]",
                          "Models are trained on the other two countries. Intervals are percentile bootstrap intervals over the held-out firms only. Unfunded firms in the held-out country: Bangladesh 23, India 20, Pakistan 16."),
                         ("within-country", "Appendix Table 8. Within-country cross-validated AUC [95% CI]",
                          "5-fold cross-validation repeated 20 times within each country. Intervals are bootstrap intervals over firms; repeated folds are not independent.")):
    rows = []
    for f in LAB:
        for m in MOD:
            cells = [cell(P[(P.features == f) & (P.model == m) & (P.analysis == ana) & (P.test_set == c)].iloc[0]) for c in ("Bangladesh", "India", "Pakistan")]
            rows.append(f"| {LAB[f]} | {MOD[m]} | " + " | ".join(cells) + " |")
    table(title, note, rows, ["Features", "Model", "Bangladesh", "India", "Pakistan"])
rows = []
for f in LAB:
    for m in MOD:
        a = P[(P.features == f) & (P.model == m) & (P.analysis == "pooled + country control") & (P.test_set == "all")].iloc[0]
        b = S[(S.features == f) & (S.model == m) & (S.analysis == "pooled + country control") & (S.test_set == "all")].iloc[0]
        rows.append(f"| {LAB[f]} | {MOD[m]} | {cell(a)} | {cell(b)} |")
table("Appendix Table 9. Pooled AUC in the primary sample and in all firms with a funding label [95% CI]",
      "The primary sample has 188 firms (founded 2010-2023, independent). The sensitivity sample has all 211 firms with a binary funding label. Pooled estimates include country indicators.",
      rows, ["Features", "Model", "Primary sample (n = 188)", "All labelled firms (n = 211)"])
open(C.TABLES.parent / "appendix_tables.md", "w").write("\n".join(out)); print("\n".join(out[:14]))
