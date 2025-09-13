import dagster as dg
from dagster import AssetExecutionContext
from dagster_dbt import dbt_assets, DbtCliResource, DbtProject
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[4]
DBT_DIR=REPO_ROOT/"dbt"

project = DbtProject(project_dir=str(DBT_DIR), profiles_dir=str(DBT_DIR))
project.prepare_if_dev()

@dbt_assets(
    manifest=project.manifest_path,
    project=project,
    name="dbt_models",
)
def dbt_models(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()
