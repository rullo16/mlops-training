import dagster as dg
from dagster import Definitions

from .defs.resources import duckdb_io_manager, dbt_resource, mlflow_tracking

from .defs.assets.data.train_dataframe import train_dataframe
from .defs.assets.data.validation_dataframe import validation_dataframe
from .defs.assets.features.dict_vectorizer import dict_vectorizer
from .defs.assets.features.X_train import X_train
from .defs.assets.features.X_val import X_val
from .defs.assets.features.y_train import y_train
from .defs.assets.features.y_val import y_val
from .defs.assets.model.trained_model import trained_model
from .defs.assets.reports.monthly_report import monthly_report


from .defs.integrations.dbt_assets import dbt_models

defs = Definitions(
    assets=[
        train_dataframe,
        validation_dataframe,
        dict_vectorizer,
        X_train,
        X_val,
        y_train,
        y_val,
        trained_model,
        monthly_report,
        dbt_models,
    ],
    resources={
        "duckdb_io_manager": duckdb_io_manager,
        "dbt": dbt_resource,
        "mlflow_tracking": mlflow_tracking,
    },
)
