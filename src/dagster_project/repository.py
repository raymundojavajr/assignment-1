from dagster import repository
from src.dagster_project.pipeline import ml_pipeline, model_evaluation_pipeline

@repository
def my_repository():
    return [ml_pipeline, model_evaluation_pipeline]
