#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import argparse
import pickle
from typing import Tuple

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CATEGORICAL = ["PU_DO"]
NUMERICAL = ["trip_distance"]
TARGET = "duration"


def read_dataframe(year: int, month: int) -> pd.DataFrame:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url, engine="pyarrow")

    # Ensure datetimes (some parquet readers can already be correct, this is defensive)
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])

    df[TARGET] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60.0
    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]

    # Base categorical fields
    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)

    # Combined categorical
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    # Defensive: keep only what we need + target
    cols = CATEGORICAL + NUMERICAL + [TARGET]
    df = df[cols].copy()

    # Handle NaNs (important for vectorizer/XGB)
    df["PU_DO"] = df["PU_DO"].fillna("UNK")
    df["trip_distance"] = df["trip_distance"].fillna(0.0)

    return df


def feature_X(df: pd.DataFrame, dv: DictVectorizer | None = None) -> Tuple[xgb.DMatrix, DictVectorizer]:
    dicts = df[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    # Give xgboost feature names for better traceability
    feature_names = dv.get_feature_names_out().tolist()
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    return dmatrix, dv


def train_model(train: xgb.DMatrix, valid: xgb.DMatrix, y_val, dv: DictVectorizer) -> str:
    params = {
        "learning_rate": 0.06431075452279457,
        "max_depth": 45,                    # very large; consider 6–12 for generalization
        "min_child_weight": 3.9809358849019483,
        "objective": "reg:squarederror",
        "reg_alpha": 0.021831790170560292,
        "reg_lambda": 0.008839071537844754,
        "seed": 42,
        "eval_metric": "rmse",              # make early stopping explicit
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Use best_iteration for evaluation
        y_pred = booster.predict(valid, iteration_range=(0, booster.best_iteration + 1))
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("best_iteration", int(booster.best_iteration))

        # Save and log preprocessor
        preproc_path = MODELS_DIR / "preprocessor.b"
        with open(preproc_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(str(preproc_path), name="preprocessor")

        # Log model
        mlflow.xgboost.log_model(booster, artifact_path="models")

        return run.info.run_id


def run(year: int, month: int):
    df_train = read_dataframe(year=year, month=month)

    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    df_val = read_dataframe(year=next_year, month=next_month)

    # Targets
    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values

    # Features
    X_train, dv = feature_X(df_train)
    X_val, _ = feature_X(df_val, dv)

    # Attach labels (kept separate for RMSE later)
    X_train.set_label(y_train)
    X_val.set_label(y_val)

    run_id = train_model(X_train, X_val, y_val, dv)
    print(f"Model trained and logged in run {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to predict taxi trip duration.")
    parser.add_argument("--year", type=int, required=True, help="Year for training data.")
    parser.add_argument("--month", type=int, required=True, help="Month for training data.")
    args = parser.parse_args()
    run(year=args.year, month=args.month)
