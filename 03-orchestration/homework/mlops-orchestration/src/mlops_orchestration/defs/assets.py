import dagster as dg
from dagster import asset, Config, AssetExecutionContext


from .resources import mlflow_tracking


import pandas as pd
import xgboost as xgb
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow




class TrainConfig(Config):
    year: int = 2023
    month: int = 1

def read_dataframe(year: int, month:int)->pd.DataFrame:
    df = pd.read_parquet(f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet", engine="pyarrow")
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df

@dg.asset(group_name="staging_data")
def train_dataframe(config: TrainConfig)->pd.DataFrame:
    return read_dataframe(year=config.year, month=config.month)

@dg.asset(group_name="staging_data")
def validation_dataframe(config:TrainConfig)->pd.DataFrame:
    next_month = config.month + 1 if config.month < 12 else 1
    next_year = config.year if config.month < 12 else config.year + 1
    return read_dataframe(year=next_year, month=next_month)

@dg.asset(group_name="features")
def dict_vectorizer(train_dataframe: pd.DataFrame)->DictVectorizer:
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = train_dataframe[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=True)
    dv.fit(dicts)
    
    return dv


@dg.asset(deps=[train_dataframe,dict_vectorizer],group_name="features")
def X_train(train_dataframe: pd.DataFrame, dict_vectorizer: DictVectorizer):
    # categorical = ['PU_DO']
    # numerical = ['trip_distance']
    # dicts = train_dataframe[categorical + numerical].to_dict(orient='records')
    return dict_vectorizer.transform(train_dataframe)

@dg.asset(deps= [validation_dataframe, dict_vectorizer],group_name="features")
def X_val(validation_dataframe: pd.DataFrame, dict_vectorizer: DictVectorizer):
    # categorical = ['PU_DO']
    # numerical = ['trip_distance']
    # dicts = validation_dataframe[categorical + numerical].to_dict(orient='records')
    return dict_vectorizer.transform(validation_dataframe)

@dg.asset(deps=[train_dataframe],group_name="features")
def y_train(train_dataframe: pd.DataFrame):
    return train_dataframe['duration'].values

@dg.asset(deps=[validation_dataframe],group_name="features")
def y_val(validation_dataframe: pd.DataFrame):
    return validation_dataframe['duration'].values

@dg.asset(deps=[X_train, y_train, X_val, y_val, dict_vectorizer], group_name="model")
def trained_model(
    context: AssetExecutionContext,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dict_vectorizer: DictVectorizer
)->str:
    context.log.info("Setting MLflow experiment")
    mlflow.set_experiment("nyc-taxi-duration-prediction")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        context.log.info(f"Training model with run_id: {run_id}")

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
        context.log.info(f"Training parameters: {params}")

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        context.log.info("Model training complete.")

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        context.log.info(f"Validation RMSE: {rmse}")

        with open("preprocessor.pkl", "wb") as f_out:
            pickle.dump(dict_vectorizer, f_out)
        mlflow.log_artifact("preprocessor.pkl", artifact_path="preprocessor")
        context.log.info("Preprocessor saved and logged.")

        mlflow.xgboost.log_model(booster, artifact_path="models")
        context.log.info("Model logged to MLflow.")
    return run_id



@dg.asset
def assets(context: dg.AssetExecutionContext) -> dg.MaterializeResult: ...
