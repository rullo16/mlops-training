import dagster as dg
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from ...partitions import monthly_partitions
from ..data.train_dataframe import TARGET_COL, FEATURE_COLS_CAT, FEATURE_COLS_NUM

def _to_records(df: pd.DataFrame)->list[dict]:
    feats = df[FEATURE_COLS_CAT + FEATURE_COLS_NUM].copy()
    for c in FEATURE_COLS_NUM: feats[c] = feats[c].astype(float).fillna(0.0)
    for c in FEATURE_COLS_CAT: feats[c] = feats[c].astype(str).fillna("UNK")

    return feats.to_dict(orient="records")


@dg.asset(partitions_def=monthly_partitions, group_name="features")
def dict_vectorizer(train_dataframe: pd.DataFrame)->DictVectorizer:
    dv = DictVectorizer()
    records = _to_records(train_dataframe)
    dv.fit(records)
    return dv
