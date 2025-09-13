import dagster as dg
from dagster import MonthlyPartitionsDefinition

monthly_partitions = MonthlyPartitionsDefinition(start_date="2023-01-01")
