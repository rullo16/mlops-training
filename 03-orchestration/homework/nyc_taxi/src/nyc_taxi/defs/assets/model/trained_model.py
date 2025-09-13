import dagster as dg
from dagster import asset, AssetExecutionContext, AssetIn
import pandas as pd
import xgboost as xgb
import pickle, mlflow
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from ...partitions import monthly_partitions


PARAMS = {
    "learning_rate": 0.06431075452279457,
    "max_depth": 12,
    "min_child_weight": 4.0,
    "objective": "reg:squarederror",
    "reg_alpha": 0.021831790170560292,
    "reg_lambda": 0.008839071537844754,
    "seed": 42,
    "eval_metric": "rmse",
}

@dg.asset(
        partitions_def=monthly_partitions,
        group_name="model",
        ins={
            "X_train": AssetIn(),
            "X_val": AssetIn(),
            "y_train": AssetIn(),
            "y_val": AssetIn(),
            "dict_vectorizer": AssetIn(),
        },
        required_resource_keys={"mlflow_tracking"},)
def trained_model(
    context: AssetExecutionContext,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    dict_vectorizer: DictVectorizer,
)->str:
    _ = context.resources.mlflow_tracking

    X_tr = X_train.sparse.to_coo().tocsr() if hasattr(X_train, "sparse") else X_train.values
    X_vl = X_val.sparse.to_coo().tocsr() if hasattr(X_val, "sparse") else X_val.values
    y_tr, y_vl = y_train.to_numpy(), y_val.to_numpy()
    feat_names = list(X_train.columns)

    pk = context.partition_key if context.has_partition_key else None
    tags = {
        "dagster_run_id": context.run_id,
        "asset_key": "/".join(context.asset_key.path),
        "partition_key": pk or "",
    }

    with mlflow.start_run(run_name=f"train_xgb:{context.run_id}", tags=tags) as run:
        mlflow.log_params(PARAMS)
        mlflow.log_params({
            "train_rows": X_tr.shape[0],
            "val_rows": X_vl.shape[0],
            "n_features": X_tr.shape[1],
        })

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        dvalid = xgb.DMatrix(X_vl, label=y_vl, feature_names=feat_names)

        booster = xgb.train(
            params=PARAMS,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_pred = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
        rmse = root_mean_squared_error(y_vl, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("best_iteration", int(booster.best_iteration))

        with open("preprocessor.pkl", "wb") as f_out:
            pickle.dump(dict_vectorizer, f_out)
        mlflow.log_artifact("preprocessor.pkl", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, name="model")
        context.log.info(f"Validation RMSE: {rmse:.4f}")
        return run.info.run_id