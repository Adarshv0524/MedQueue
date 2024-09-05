import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import os

def preprocess_data(df):
    """
    Preprocess the raw data including normalization, feature engineering,
    and urgency score creation.
    
    Parameters:
        df (pd.DataFrame): The raw data.
    
    Returns:
        pd.DataFrame: The preprocessed data.
    """
    # Ensure the Blood Pressure column is a string and strip any extra spaces
    df['Blood Pressure'] = df['Blood Pressure'].astype(str).str.strip()

    # Split the Blood Pressure column into Systolic and Diastolic BP
    df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)

    # Convert Systolic and Diastolic BP to numeric
    df['Systolic BP'] = pd.to_numeric(df['Systolic BP'], errors='coerce')
    df['Diastolic BP'] = pd.to_numeric(df['Diastolic BP'], errors='coerce')

    # Handling missing values with SimpleImputer
    numerical_columns = ['Heart Rate', 'Temperature' , 'Respiratory Rate', 'Systolic BP', 'Diastolic BP']
    categorical_columns = ['Gender', 'Symptoms', 'Medical History', 'Diagnosis']

    # Impute missing values for numerical columns
    imputer_num = SimpleImputer(strategy='median')
    df[numerical_columns] = imputer_num.fit_transform(df[numerical_columns])
    
    # Impute missing values for categorical columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])
    
    # Feature Engineering: Adding interaction terms
    df['BP_Ratio'] = df['Systolic BP'] / (df['Diastolic BP'] + 1)
    df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, np.inf], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    
    # Create Urgency Score based on more features
    df['Urgency_Score'] = (
        df['Systolic BP'] * 0.3 +
        df['Diastolic BP'] * 0.2 +
        df['Heart Rate'] * 0.3 +
        df['Temperature'] * 0.1 +
        df['Respiratory Rate'] * 0.1
    )
    
    # Normalize numerical features
    scaler = StandardScaler()
    df[numerical_columns + ['Urgency_Score']] = scaler.fit_transform(df[numerical_columns + ['Urgency_Score']])
    
    # Encode categorical features with OneHotEncoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cats = encoder.fit_transform(df[categorical_columns])
    
    # Create a new DataFrame for the encoded features
    df_encoded = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Combine the encoded features with the original DataFrame
    df = pd.concat([df.drop(columns=categorical_columns), df_encoded], axis=1)
    
    return df

def save_preprocessed_data(df, file_path):
    """
    Save the preprocessed data to a CSV file.
    
    Parameters:
        df (pd.DataFrame): The preprocessed data.
        file_path (str): The path where the preprocessed data will be saved.
    """
    df.to_csv(file_path, index=False)
    print(f"Preprocessed data saved to {file_path}")

if __name__ == "__main__":
    # Define the correct path to the raw data file
    raw_data_path = os.path.join('..', '..', 'data', 'raw', 'patient_data.csv')
    
    try:
        # Load raw data
        df = pd.read_csv(raw_data_path)
        
        # Preprocess the data
        df_preprocessed = preprocess_data(df)
        
        # Define the path for saving preprocessed data
        processed_data_path = os.path.join('..', '..', 'data', 'processed', 'preprocessed_medical_data.csv')
        
        # Save the preprocessed data
        save_preprocessed_data(df_preprocessed, processed_data_path)
        
    except FileNotFoundError:
        print(f"Error: The file {raw_data_path} was not found. Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Display the first few rows of the split columns
    print(df[['Systolic BP', 'Diastolic BP']].head())

    import matplotlib.pyplot as plt
    import seaborn as sns
    # Plot the distribution of Systolic BP
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Systolic BP'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Systolic Blood Pressure')
    plt.xlabel('Systolic BP')
    plt.ylabel('Frequency')
    plt.show()

    # Plot the distribution of Diastolic BP
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Diastolic BP'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Diastolic Blood Pressure')
    plt.xlabel('Diastolic BP')
    plt.ylabel('Frequency')
    plt.show()