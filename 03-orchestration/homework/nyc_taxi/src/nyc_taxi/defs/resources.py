import dagster as dg
from dagster_duckdb_pandas import DuckDBPandasIOManager
from dagster import ConfigurableResource, InitResourceContext
from dagster_dbt import DbtCliResource
import mlflow
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
DUCKDB_ABS = str(DATA_DIR / "nyc.duckdb")




duckdb_io_manager = DuckDBPandasIOManager(
    database=DUCKDB_ABS,
    schema="raw",
)

# dbt_resource = DbtCliResource(
#     project_dir=str(REPO_ROOT / "dbt"),
#     profiles_dir=str(REPO_ROOT / "dbt"),
#     dbt_executable="dbt",
#     env={"DUCKDB_PATH": DUCKDB_ABS},
# )

class MLFlowTracking(ConfigurableResource):
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment: str = "nyc-taxi-duration-prediction"

    def setup_for_execution(self, context: InitResourceContext):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        context.log.info(
            f"MLflow configured: uri={self.tracking_uri}, experiment={self.experiment}"
        )

mlflow_tracking = MLFlowTracking()
dbt_resource = DbtCliResource(project_dir="dbt")
