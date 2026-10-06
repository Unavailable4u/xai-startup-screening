import pandas as pd
from .config import MODEL_READY

def load_startups(only_independent: bool = False) -> pd.DataFrame:
    df = pd.read_excel(MODEL_READY, sheet_name="Model_Ready_Startups")
    if only_independent:
        df = df[df["entity_type"].str.startswith("Independent")]
    return df.reset_index(drop=True)

def labelled(df: pd.DataFrame, target: str):
    d = df[df[target].notna()].copy()
    return d, d[target].astype(int)
