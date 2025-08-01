#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    df = pd.read_parquet(f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet", engine="pyarrow")
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df

def feature_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        params = {
            'learning_rate': 0.06431075452279457,
            'max_depth': 45,
            'min_child_weight': 3.9809358849019483,
            'objective': 'reg:squarederror',
            'reg_alpha': 0.021831790170560292,
            'reg_lambda': 0.008839071537844754,
            'seed': 42
        }

        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round = 1000,
            evals=[(valid,'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", name="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models")

    return run.info.run_id


def run(year, month):
    df_train = read_dataframe(year=year, month=month)
    
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    df_val = read_dataframe(year = next_year, month= next_month)

    X_train, dv = feature_X(df_train)
    X_val, _ = feature_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"Model trained and logged in run {run_id}")

if __name__ == "__main__":

    # use argparse to get year and month from command line

    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type = int, required=True, help='Year of the training data to train the model on.')
    parser.add_argument('--month', type = int, required=True, help='Month of the training data to train the model on.')
    args = parser.parse_args()


    run(year=args.year, month=args.month)
