import joblib

def save_model(model, file_path='models/logistic_regression_model.joblib'):
    """
    Save the trained model to the specified file path.
    
    Parameters:
    model: The trained machine learning model.
    file_path: The location to save the model (default: 'models/logistic_regression_model.joblib').
    """
    joblib.dump(model, file_path)
    print(f"Model saved successfully to '{file_path}'")

def load_model(file_path='models/logistic_regression_model.joblib'):
    """
    Load a machine learning model from the specified file path.
    
    Parameters:
    file_path: The path to the saved model file (default: 'models/logistic_regression_model.joblib').
    
    Returns:
    model: The loaded machine learning model.
    
    Raises:
    FileNotFoundError: If the model file does not exist.
    """
    try:
        model = joblib.load(file_path)
        print(f"Model loaded successfully from '{file_path}'")
        return model
    except FileNotFoundError:
        print(f"Error: The model file '{file_path}' does not exist.")
        raise FileNotFoundError(f"The model file '{file_path}' does not exist.")