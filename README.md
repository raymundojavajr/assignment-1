# **ML Project: Testing the Organization of MLOps**

This repository contains the implementation of a machine learning project as part of **Assignment 1 - MLOps**. The goal is to demonstrate best practices in structuring and organizing an ML project following MLOps principles. This includes proper separation of concerns, logging, saving/loading models, and maintaining a clear project structure.

## **Project Overview**

This project focuses on **logistic regression-based classification**, covering key MLOps aspects such as:

- **Model Training**: Implementing logistic regression for data classification.
- **Model Saving & Loading**: Using `joblib` to persist models for later use.
- **Logging**: Capturing model training logs for better traceability.
- **Project Structure**: Ensuring modularity, maintainability, and scalability.

## **Repository Structure**
```
├── data/                   # Dataset storage (ignored in Git)
├── models/                 # Trained model storage (ignored in Git)
├── notebooks/              # Jupyter notebooks for EDA & experiments
├── src/                    # Source code for model training & utilities
│   ├── train.py            # Training script
│   ├── model.py            # Model implementation
│   ├── utils.py            # Helper functions
├── .gitignore              # Ignoring unnecessary files (cache, logs, models, etc.)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

## **Setup & Installation**
To set up the project, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/raymundojavajr/assignment-1.git
   cd assignment-1
   ```

2. **Create a virtual environment**:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## **Usage**
To train the model:
```sh
python src/train.py
```

## **Logging & Debugging**
All logs are stored in `logs/`, and training runs are tracked to monitor model performance.