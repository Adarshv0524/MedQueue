import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path):
    """
    Load the preprocessed data from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

def encode_features(df, categorical_columns):
    """
    Encode categorical features using LabelEncoder.
    
    Parameters:
        df (pd.DataFrame): The data with features.
        categorical_columns (list): List of columns to encode.
    
    Returns:
        pd.DataFrame: The data with encoded features.
    """
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def select_features(df, target_column, num_features=10):
    """
    Select the most relevant features based on statistical tests.
    
    Parameters:
        df (pd.DataFrame): The data with features.
        target_column (str): The name of the target column.
        num_features (int): The number of top features to select.
    
    Returns:
        pd.DataFrame: The data with selected features.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Apply feature selection
    selector = SelectKBest(score_func=f_classif, k=min(num_features, len(X.columns)))
    X_new = selector.fit_transform(X, y)
    
    # Get the selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Return DataFrame with selected features
    df_selected = df[selected_features + [target_column]]
    return df_selected, selected_features

def save_selected_features(df, file_path):
    """
    Save the data with selected features to a CSV file.
    
    Parameters:
        df (pd.DataFrame): The data with selected features.
        file_path (str): The path where the data will be saved.
    """
    df.to_csv(file_path, index=False)
    print(f"Selected features data saved to {file_path}")

if __name__ == "__main__":
    # Define the paths
    preprocessed_data_path = os.path.join('..', '..', 'data', 'processed', 'preprocessed_medical_data.csv')
    selected_features_data_path = os.path.join('..', '..', 'data', 'processed', 'selected_features_medical_data.csv')
    
    # Load the data
    df = load_data(preprocessed_data_path)
    
    # Define the target column
    target_column = 'Urgency_Score'
    
    # Identify categorical columns (object dtype)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical features
    df = encode_features(df, categorical_columns)
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                print(f"Warning: Could not convert column '{col}' to numeric. Dropping this column.")
                df = df.drop(columns=[col])
    
    # Select features
    df_selected, selected_features = select_features(df, target_column, num_features=10)
    
    # Save the selected features
    save_selected_features(df_selected, selected_features_data_path)
    
    print(f"Selected Features: {selected_features}")