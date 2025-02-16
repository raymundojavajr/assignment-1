"""
Main module for running the ML pipeline.
"""

import sys
import os
import logging
import datetime

from models import train_model
from data_loading import load_data
from utils.save_utils import save_model

# Add the src directory to the Python path BEFORE importing modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Main function to run the ML pipeline."""
    # Generate a dynamic log filename using the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join("logs", f"training_{current_date}.log")
    
    # Configure logging to file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_filename,
        filemode="a"  # Append mode
    )
    
    # Add a stream handler to output logs to the terminal as well
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info("Starting the ML pipeline")

    try:
        file_path = "data/match_maker.xlsx"
        data = load_data(file_path)
        logging.info("Data loaded successfully")

        x_data = data.drop("target", axis=1)
        y_data = data["target"]

        model = train_model(x_data, y_data)
        save_model(model, "model.pkl")
        logging.info("Model saved successfully")

    except Exception as ex:
        logging.error("An error occurred: %s", ex)
        sys.exit(1)


if __name__ == "__main__":
    main()
