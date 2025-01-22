from dagster import asset
from src.data_loading import load_data
from src.models import train_model as train_model_fn
from src.utils.save_utils import save_model

@asset
def raw_data():
    return load_data("data/match_maker.xlsx")

@asset
def trained_model(raw_data):
    x_data = raw_data.drop("target", axis=1)    
    y_data = raw_data["target"] 
    model = train_model_fn(x_data, y_data) 
    return model

@asset
def saved_model(trained_model):
    save_model(trained_model, "model.pkl")
    return "model.pkl"