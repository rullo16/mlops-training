from dagster import asset, AssetIn, AssetExecutionContext, MetadataValue
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

@asset(
    group_name="reports",
    ins={"trained_model": AssetIn()},  # <-- SAME MONTH, no TimeWindowPartitionMapping
)
def monthly_report(context: AssetExecutionContext, trained_model: str) -> str:
    """
    Example: create a small plot or CSV summary for the current partition and attach it as metadata.
    Replace the body with whatever your real report builds.
    """
    # ----- build your figure / file(s) -----
    # Example plot
    fig, ax = plt.subplots()
    ax.set_title(f"Validation RMSE for {context.partition_key or 'N/A'}")
    ax.plot([0, 1, 2], [1.2, 0.9, 0.8])
    plot_path = REPORTS_DIR / f"report_{(context.partition_key or 'none').replace('-', '')}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    # Optionally write a small CSV/JSON summary
    # summary_path = REPORTS_DIR / f"summary_{(context.partition_key or 'none').replace('-', '')}.csv"
    # pd.DataFrame({"metric": ["rmse"], "value": [0.8]}).to_csv(summary_path, index=False)

    # ----- attach metadata so it shows up in Dagster UI -----
    context.add_output_metadata({"trained_model_run": MetadataValue.text(trained_model)})

    # You can return something if your report is a materialized file path or dict
    return str(plot_path)

