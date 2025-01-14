"""
Module for loading data from an Excel file.
"""

import pandas as pd


def load_data(file_path):
    """Load data from an Excel file."""
    data_frame = pd.read_excel(file_path)
    return data_frame


if __name__ == "__main__":
    FILE_PATH = "data/match_maker.xlsx"
    data = load_data(FILE_PATH)
    print("Data loaded successfully!")
    print(data.head())
