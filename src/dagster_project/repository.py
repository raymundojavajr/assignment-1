from dagster import repository
from dagster_project.assets import raw_data, trained_model, saved_model

@repository
def my_repository():
    return [raw_data, trained_model, saved_model]
