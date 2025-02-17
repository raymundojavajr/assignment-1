import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set MLflow tracking URI
MLFLOW_URI = "http://127.0.0.1:5000"  # Change to "http://mlflow:5000" if running in Docker
mlflow.set_tracking_uri(MLFLOW_URI)

# Define experiment name
EXPERIMENT_NAME = "mlflow_experiment"

# Ensure experiment exists before setting it
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

def train_model(input_data, target):
    """
    Trains a logistic regression model, logs metrics & parameters to MLflow, and returns it.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, target, test_size=0.25, random_state=42
    )

    model = LogisticRegression(max_iter=10000)

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # Log hyperparameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 10000)
        mlflow.log_param("test_size", 0.25)

        model.fit(x_train, y_train)

        # Predict and log accuracy
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)

        # Input example (ensure conversion to proper format)
        input_example = np.array(x_train.iloc[0].values).reshape(1, -1)

        # Save and log model
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)

        print(f"Model trained & logged. Accuracy: {accuracy:.4f}")
        print(f"View run at: http://127.0.0.1:5000/#/experiments/1/runs/{run.info.run_id}")

    return model
