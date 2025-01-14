import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(X, y):
    """
    Trains a logistic regression model on the provided data and returns the trained model.
    Parameters:
    X (array-like or sparse matrix): The input data to train the model.
    y (array-like): The target values (class labels) corresponding to the input data.
    Returns:
    model (LogisticRegression): The trained logistic regression model.
    The function performs the following steps:
    1. Splits the data into training and testing sets.
    2. Initializes and trains a logistic regression model.
    3. Calculates and prints the accuracy on both the training and testing sets.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Initialize and train the model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Logging the results
    logging.info(f"Train accuracy: {train_accuracy:.2f}")
    logging.info(f"Test accuracy: {test_accuracy:.2f}")
    
    print(f"Train accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")
    
    return model