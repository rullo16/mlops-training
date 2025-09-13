import dagster as dg
from dagster import AssetIn, asset
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from ...partitions import monthly_partitions
from .dict_vectorizer import _to_records

def _feature_names(dv: DictVectorizer):
    return dv.get_feature_names_out() if hasattr(dv, "get_feature_names_out") else dv.get_feature_names()

@dg.asset(partitions_def=monthly_partitions,
          group_name="features",
          ins={"train_dataframe": AssetIn()},
          deps=["dict_vectorizer"])
def X_train(train_dataframe: pd.DataFrame, dict_vectorizer: DictVectorizer)->pd.DataFrame:

    X = dict_vectorizer.transform(_to_records(train_dataframe))
    cols = _feature_names(dict_vectorizer)
    return pd.DataFrame.sparse.from_spmatrix(X, index=train_dataframe.index, columns=cols)
