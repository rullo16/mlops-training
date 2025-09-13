import dagster as dg
from dagster import asset, AssetIn, TimeWindowPartitionMapping
import pandas as pd
from ...partitions import monthly_partitions
from ..data.train_dataframe import TARGET_COL

@dg.asset(
        partitions_def=monthly_partitions,
        group_name="features",
        ins={
            "validation_dataframe": AssetIn(
                partition_mapping=TimeWindowPartitionMapping(start_offset=1, end_offset=1)
            )
        },
)
def y_val(validation_dataframe:pd.DataFrame)->pd.Series:
    return validation_dataframe[TARGET_COL].astype(float)
