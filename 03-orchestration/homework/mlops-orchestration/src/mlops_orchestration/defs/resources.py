import dagster as dg
from dagster_mlflow import mlflow_tracking

@dg.definitions
def resources() -> dg.Definitions:
    return dg.Definitions(resources={
        "mlflow_tracking": mlflow_tracking.configured(
            tracking_uri="http://localhost:5000",
            experiment_name="nyc-taxi-experiment"
        )
    })


@dg.definitions
def resources() -> dg.Definitions:
    return dg.Definitions(resources={})
