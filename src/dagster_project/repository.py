from dagster import repository
from src.dagster_project.assets import raw_data, trained_model, saved_model
from src.dagster_project.pipeline import (
    ml_pipeline,
    model_evaluation_pipeline,
)  # Import jobs


@repository
def my_repository():
    return [
        raw_data,
        trained_model,
        saved_model,
        ml_pipeline,
        model_evaluation_pipeline,
    ]  # Add jobs
