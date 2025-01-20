from dagster import op, job
from src.data_loading import load_data
from src.models import train_model
from src.utils.save_utils import save_model, load_model


@op
def load_data_op():
    file_path = "data/match_maker.xlsx"
    return load_data(file_path)


@op
def train_model_op(data):
    x_data = data.drop("target", axis=1)
    y_data = data["target"]
    model = train_model(x_data, y_data)
    return model


@op
def save_model_op(model):
    file_path = "models/model.pkl"
    save_model(model, file_path)


@op
def load_model_op():
    file_path = "models/model.pkl"
    return load_model(file_path)


@job
def ml_pipeline():
    data = load_data_op()
    model = train_model_op(data)
    save_model_op(model)


@job
def model_evaluation_pipeline():
    load_model_op()
