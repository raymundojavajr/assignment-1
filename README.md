Here's an updated version of your README that reflects your current project structure and usage:

---

# ML Ops Project: Testing the Organization of MLOps

This repository implements a machine learning project following MLOps principles. It demonstrates best practices in organizing an ML project, including modularity, logging, model persistence, and pipeline orchestration using Dagster.

## Project Overview

The project focuses on **logistic regression-based classification** and covers key MLOps aspects such as:

- **Model Training**: Training a logistic regression model.
- **Model Saving & Loading**: Persisting the trained model for later use.
- **Logging**: Capturing detailed logs during training and pipeline execution.
- **Pipeline Orchestration**: Using Dagster to manage and monitor the ML pipeline.
- **Code Quality**: Enforcing code quality with pre-commit hooks using Ruff.

## Repository Structure

```
├── data/                    # Dataset storage (e.g., Excel files)
├── logs/                    # Training logs (generated during execution)
├── src/                     # Source code for the ML pipeline
│   ├── main.py              # Main module for running the ML pipeline
│   ├── models.py            # Model implementation and training logic
│   ├── data_loading.py      # Module for loading data from files
│   ├── dagster_project/     # Dagster repository and pipeline definitions
│   │   ├── __init__.py      # Package initialization for Dagster assets
│   │   ├── pipeline.py      # Pipeline/job definitions
│   │   └── repository.py    # Dagster repository definition
│   └── utils/               # Utility modules
│       └── save_utils.py    # Functions for saving and loading models
├── .gitignore               # Files/directories ignored by Git
├── .pre-commit-config.yaml  # Pre-commit hook configuration for code quality (Ruff)
├── .python-version          # Specifies the Python version used in this project
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project configuration for tools (e.g., build, linting)
├── uv.lock                  # Lock file for the uv environment (if applicable)
└── README.md                # This project documentation
```

## Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/raymundojavajr/ml-ops.git
   cd ml-ops
   ```

2. **Create a virtual environment:**
   ```sh
   python -m venv .venv
   ```
   Activate it:
   - On macOS/Linux:
     ```sh
     source .venv/bin/activate
     ```
   - On Windows:
     ```sh
     .venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Install Pre-commit Hooks (Optional):**
   To enforce code quality with Ruff:
   ```sh
   pre-commit install
   ```

## Usage

- **Run the ML pipeline:**
  ```sh
  python src/main.py
  ```
  This command will load your data, train a logistic regression model, and save the model while logging the process.

- **Monitor Pipelines with Dagster:**
  Launch the Dagit UI to monitor your pipeline:
  ```sh
  dagit -f src/dagster_project/repository.py
  ```
  Then open [http://localhost:3000](http://localhost:3000) in your browser.

## Logging & Debugging

Logs are automatically generated in the `logs/` directory with filenames that include the current date (e.g., `training_2025-02-16.log`). This allows you to track each training run and debug issues efficiently.