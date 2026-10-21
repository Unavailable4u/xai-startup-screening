import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd
from sklearn.base import clone
from xai_capital import config as C
from xai_capital.data import load_startups, labelled, in_frame, boot_auc
from xai_capital.models import make_models
from xai_capital.evaluate import repeated_cv_oof
T=C.TARGETS["primary"]; REP=3
fr=in_frame(load_startups()); d,y=labelled(fr,T); d=d.reset_index(drop=True); y=y.reset_index(drop=True)
rows=[]
def loco(name,num,cat):
    for held in ["Bangladesh","India","Pakistan"]:
        tr,te=d[d.country!=held],d[d.country==held]
        for mn,mod in make_models(num,cat).items():
            if mn not in("logit_l2","rf","xgb"): continue
            p=clone(mod).fit(tr[num+cat],y[tr.index]).predict_proba(te[num+cat])[:,1]
            a,lo,hi=boot_auc(y[te.index],p); rows.append((name,"LOCO "+held,mn,int((y[te.index]==0).sum()),round(a,3),round(lo,3),round(hi,3)))
def pooled(name,num,cat):
    strata=d.country+"_"+y.astype(str); n=num+C.COUNTRY_DUMMIES
    for mn,mod in make_models(n,cat).items():
        if mn not in("logit_l2","rf","xgb"): continue
        o=repeated_cv_oof(mod,d[n+cat],y,5,REP,strata=strata).mean(axis=0); a,lo,hi=boot_auc(y,o)
        rows.append((name,"pooled+country","%s"%mn,int((y==0).sum()),round(a,3),round(lo,3),round(hi,3)))
neutral=["age_years","product_stage","n_founders","gdp_growth_at_founding","inflation_at_founding"]
for nm,num,cat in (("CORE+sector",C.CORE,C.CAT_A),("CORE no sector/biz-model",neutral,[]),("sector+biz-model only",[c for c in C.CORE if c.startswith("is_")],C.CAT_A)):
    pooled(nm,num,cat); loco(nm,num,cat)
# H1/H3 within Bangladesh, complete cases on founder attributes (no missingness indicators needed)
F=["edu_max_level","any_stem_founder","serial_entrepreneur","industry_exp_max_years"]
bd=d[(d.country=="Bangladesh")&d[F].notna().all(axis=1)]; yb=y[bd.index]
print("BD complete-case founder rows:",len(bd),"unfunded",int((yb==0).sum()),"| all BD frame rows:",int((d.country=='Bangladesh').sum()))
for nm,num in (("founder attrs only (BD complete-case)",F),("founder attrs + age,stage (BD cc)",F+["age_years","product_stage"])):
    for mn,mod in make_models(num,[]).items():
        if mn not in("logit_l2","rf","xgb"): continue
        o=repeated_cv_oof(mod,bd[num].reset_index(drop=True),yb.reset_index(drop=True),5,REP).mean(axis=0); a,lo,hi=boot_auc(yb,o)
        rows.append((nm,"within BD",mn,int((yb==0).sum()),round(a,3),round(lo,3),round(hi,3)))
r=pd.DataFrame(rows,columns=["features","analysis","model","n_unfunded_test","auc","lo","hi"])
pd.set_option("display.width",220,"display.max_rows",200); print(r.to_string(index=False))
r.to_csv("results/tables/v3_checks.csv",index=False)
