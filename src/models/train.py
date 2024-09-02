import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_data(file_path):
    """
    Load the selected features data from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

def train_model(df, target_column):
    """
    Train a Decision Tree regression model.
    
    Parameters:
        df (pd.DataFrame): The data with features.
        target_column (str): The name of the target column.
    
    Returns:
        DecisionTreeRegressor: The trained model.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Decision Tree model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    return model

def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Parameters:
        model (DecisionTreeRegressor): The trained model.
        file_path (str): The path where the model will be saved.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

if __name__ == "__main__":
    # Define the paths
    selected_features_data_path = '../../data/processed/selected_features_medical_data.csv'
    model_save_path = '../../data/models/decision_tree_model.pkl'
    
    # Load the data
    df = load_data(selected_features_data_path)
    
    # Define the target column
    target_column = 'Urgency_Score'
    
    # Train the model
    model = train_model(df, target_column)
    
    # Save the trained model
    save_model(model, model_save_path)