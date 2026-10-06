"""Leakage-safe pipelines: imputation happens INSIDE each CV fold."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from .config import SEED

def _pre(num, cat, scale):
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat),
    ])

def make_models(num, cat):
    return {
        "logit_l2": Pipeline([("pre", _pre(num, cat, True)),
            ("clf", LogisticRegression(C=0.3, max_iter=5000,
                                       class_weight="balanced"))]),
        "logit_lasso": Pipeline([("pre", _pre(num, cat, True)),
            ("clf", LogisticRegression(l1_ratio=1.0, solver="liblinear", C=0.3,
                                       class_weight="balanced"))]),
        "rf": Pipeline([("pre", _pre(num, cat, False)),
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced_subsample",
                random_state=SEED, n_jobs=-1))]),
        # XGBoost handles NaN natively, so no imputer; shallow trees for N~86
        "xgb": Pipeline([("pre", ColumnTransformer([
                ("num", "passthrough", num),
                ("cat", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False), cat)])),
            ("clf", XGBClassifier(n_estimators=150, max_depth=2,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=2, reg_lambda=5.0, eval_metric="logloss",
                random_state=SEED, n_jobs=1))]),
    }
