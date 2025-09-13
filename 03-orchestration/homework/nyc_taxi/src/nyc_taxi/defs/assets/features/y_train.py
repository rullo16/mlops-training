import dagster as dg
from dagster import asset
import pandas as pd
from ...partitions import monthly_partitions
from ..data.train_dataframe import TARGET_COL


@dg.asset(partitions_def=monthly_partitions, group_name="features")
def y_train(train_dataframe: pd.DataFrame) -> pd.Series:
    return train_dataframe[TARGET_COL].astype(float)
