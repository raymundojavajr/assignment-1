import os
import sys
import logging
from datetime import datetime

"""
Main script for training a machine learning model.

This script performs the following steps:
1. Sets up logging to capture the training process.
2. Loads data from an Excel file.
3. Prepares the data for training by separating features and target variable.
4. Trains a logistic regression model.
5. Saves the trained model to a specified directory.

Modules:
    os: Provides a way of using operating system dependent functionality.
    sys: Provides access to some variables used or maintained by the interpreter.
    logging: Provides a way to configure logging.
    datetime: Supplies classes for manipulating dates and times.
    src.models: Contains the function to train the model.
    src.data_loading: Contains the function to load data.
    src.utils.save_utils: Contains the function to save the model.

Functions:
    None

Exceptions:
    Logs any exceptions that occur during the training process.

Logging:
    Logs the start and end of the training process, data loading details, and model saving status.
"""

# Add the project root to Python's path to recognize src as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import train_model  # Train the model
from src.data_loading import load_data  # Load the data
from src.utils.save_utils import save_model  # Save the trained model

# Create logs directory if it does not exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Log file path
log_file = f'logs/training_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    logging.info("Training process started")

    # Load data
    file_path = 'data/match_maker.xlsx'
    data = load_data(file_path)
    logging.info(f'Data loaded from {file_path}, shape: {data.shape}')

    # Prepare features (X) and target (y)
    X = data.drop('target', axis=1)  # Ensure 'target' column exists
    y = data['target']  # Ensure this is the correct column for target

    # Train the model
    model = train_model(X, y)

    # Model save path
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'final_logistic_regression_model.joblib')
    save_model(model, model_path)

    logging.info(f"Model saved successfully to '{model_path}'")
    logging.info("Training process finished")

except Exception as e:
    logging.error(f"An error occurred: {e}")