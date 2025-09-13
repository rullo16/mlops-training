import dagster as dg
from dagster import asset
import pandas as pd
from .train_dataframe import _read_dataframe, _ym_from_key
from ...partitions import monthly_partitions


@dg.asset(partitions_def=monthly_partitions, group_name="staging_data", io_manager_key="duckdb_io_manager",metadata={"partition_expr": "ds"})
def validation_dataframe(context) -> pd.DataFrame:
    y,m = _ym_from_key(context.partition_key)
    df = _read_dataframe()
    df["ds"] = pd.to_datetime(context.partition_key)
    return df
