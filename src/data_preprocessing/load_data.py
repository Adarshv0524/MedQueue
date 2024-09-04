import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing raw data.

    Returns:
        pd.DataFrame: The loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Load raw data
    df = load_data('../../data/raw/patient_data.csv')
