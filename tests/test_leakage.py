import sys; sys.path.insert(0, "src")
from xai_capital import config as C
from xai_capital.data import load_startups

def test_no_forbidden_features():
    feats = set(C.NUM_A + C.CAT_A + C.EXTRA_B + C.EXTRA_C)
    assert not (feats & set(C.FORBIDDEN))

def test_columns_exist():
    df = load_startups()
    need = set(C.NUM_A + C.CAT_A + C.EXTRA_B + C.EXTRA_C + list(C.TARGETS.values()))
    assert need <= set(df.columns), need - set(df.columns)

def test_ids_unique():
    assert load_startups()["startup_id"].is_unique
