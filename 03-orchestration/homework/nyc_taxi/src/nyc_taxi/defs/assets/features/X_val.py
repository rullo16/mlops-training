import dagster as dg
from dagster import asset, AssetIn, TimeWindowPartitionMapping
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from ...partitions import monthly_partitions
from .dict_vectorizer import _to_records
from .X_train import _feature_names


@dg.asset(
        partitions_def=monthly_partitions,
        group_name="features",
        ins={
            "validation_dataframe": AssetIn(
                partition_mapping=TimeWindowPartitionMapping(start_offset=1, end_offset=1)
            )
        },
        deps = ["dict_vectorizer"]
)
def X_val(validation_dataframe:pd.DataFrame, dict_vectorizer:DictVectorizer)->pd.DataFrame:
    X = dict_vectorizer.transform(_to_records(validation_dataframe))
    cols = _feature_names(dict_vectorizer)
    return pd.DataFrame.sparse.from_spmatrix(X, index=validation_dataframe.index, columns=cols)