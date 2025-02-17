from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import uvicorn
import logging

# Initialize FastAPI
app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.INFO)

# Define input schema
class PredictionInput(BaseModel):
    features: list

# Load model from MLflow
try:
    model_name = "my_model"
    model_version = 1  # Change if needed
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
    logging.info("Model loaded successfully from MLflow.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    model = None

@app.get("/")
def home():
    return {"message": "MLflow FastAPI is running!"}

@app.post("/predict/")
def predict(data: PredictionInput):
    """
    Receive input data as JSON and return predictions.
    Example input: {"features": [[5.1, 3.5, 1.4, 0.2]]}
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available.")

    try:
        predictions = model.predict(data.features).tolist()
        return {"predictions": predictions}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
