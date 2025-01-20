from dagster import repository
from src.dagster_project.pipeline import my_pipeline

@repository
def my_repository():
    return [my_pipeline]
