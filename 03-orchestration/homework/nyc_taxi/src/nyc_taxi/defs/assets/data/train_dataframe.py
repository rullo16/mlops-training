import dagster as dg
from dagster import asset
from datetime import datetime
from ...partitions import monthly_partitions
import pandas as pd

TARGET_COL="duration"
FEATURE_COLS_CAT=["PU_DO"]
FEATURE_COLS_NUM=["trip_distance"]

def _ym_from_key(key:str)->str:
    dt = datetime.fromisoformat(key)
    return dt.year, dt.month

def _read_dataframe()->pd.DataFrame:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet"
    df = pd.read_parquet(url, engine="pyarrow")
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    df[TARGET_COL] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60.0
    df = df[(df[TARGET_COL] >= 1) & (df[TARGET_COL] <= 60)]
    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    return df[[*FEATURE_COLS_CAT, *FEATURE_COLS_NUM, TARGET_COL]].copy()


@dg.asset(group_name="staging_data", io_manager_key="duckdb_io_manager", metadata={"partition_expr":"ds"})
def train_dataframe(context) -> pd.DataFrame:
    df = _read_dataframe()
    df["ds"] = pd.to_datetime(context.partition_key)
    return df
