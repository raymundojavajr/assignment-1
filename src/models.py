"""
Module for training a logistic regression model.
"""

import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model(input_data, target):
    """
    Trains a logistic regression model and returns it.

    Parameters:
    input_data (array-like): Input data.
    target (array-like): Target values.

    Returns:
    LogisticRegression: Trained model.
    """
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, target, test_size=0.25, random_state=42
    )

    # Initialize and train the model
    model = LogisticRegression(max_iter=10000)
    model.fit(x_train, y_train)

    # Predict and log accuracy
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Model accuracy: %s", accuracy)

    return model
