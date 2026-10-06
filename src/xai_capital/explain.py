import numpy as np, pandas as pd, shap
from .config import SEED

def fit_and_shap(pipe, X, y):
    """Fit a tree pipeline on ALL labelled data (explanation only, not performance)."""
    pipe.fit(X, y)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pd.DataFrame(pre.transform(X), columns=[c.split("__", 1)[-1] for c in pre.get_feature_names_out()],
                      index=X.index)
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(Xt)
    if isinstance(sv, list):          # older shap returns [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        sv = sv[:, :, 1]
    return Xt, pd.DataFrame(sv, columns=Xt.columns, index=Xt.index)

def global_importance(shap_df):
    return shap_df.abs().mean().sort_values(ascending=False)
