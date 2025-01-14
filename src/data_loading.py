import pandas as pd

def load_data(file_path):
    """
    Load data from an Excel file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the Excel file to be loaded.

    Returns:
    pandas.DataFrame: The data loaded from the Excel file.
    """
    df = pd.read_excel(file_path)
    return df

if __name__ == "__main__":
    file_path = "data/match_maker.xlsx"
    data = load_data(file_path)
    print("Data loaded successfully!")
    print(data.head())
