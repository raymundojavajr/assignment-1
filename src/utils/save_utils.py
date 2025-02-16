"""
Utility module for saving and loading models.
"""

import pickle
import joblib


def save_model(model, file_path):
    """Save the model to a file."""
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"The model file '{file_path}' does not exist."
        ) from exc


def load_model(file_path="models/logistic_regression_model.joblib"):
    """Load a model from the specified file path."""
    try:
        model = joblib.load(file_path)
        print(f"Model loaded successfully from '{file_path}'")
        return model
    except FileNotFoundError:
        print(f"Error: The model file '{file_path}' does not exist.")
        raise FileNotFoundError(f"The model file '{file_path}' does not exist.")
