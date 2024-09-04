# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# from datetime import datetime

# def preprocess_data(df):
#     """
#     Preprocess the raw medical data including normalization and feature engineering.
#     Parameters:
#         df (pd.DataFrame): The raw data.
#     Returns:
#         pd.DataFrame: The preprocessed data.
#     """
#     # Drop rows with missing values in essential columns
#     df = df.dropna(subset=['Age', 'Gender', 'Blood Pressure', 'Heart Rate', 'Temperature', 'Respiratory Rate', 'Symptoms', 'Medical History', 'Diagnosis', 'Admission Date'])
    
#     # Feature Engineering
#     df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, np.inf], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    
#     # Split Blood Pressure into Systolic and Diastolic
#     df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
#     df = df.drop('Blood Pressure', axis=1)
    
#     # Normalize numerical features
#     scaler = StandardScaler()
#     df[['Systolic', 'Diastolic', 'Heart Rate', 'Temperature', 'Respiratory Rate']] = scaler.fit_transform(df[['Systolic', 'Diastolic', 'Heart Rate', 'Temperature', 'Respiratory Rate']])
    
#     # Encode categorical features
#     df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
    
#     # Multi-label encoding for Symptoms and Medical History
#     mlb = MultiLabelBinarizer()
#     symptoms_encoded = pd.DataFrame(mlb.fit_transform(df['Symptoms'].str.split(', ')),
#                                     columns=mlb.classes_,
#                                     index=df.index)
#     medical_history_encoded = pd.DataFrame(mlb.fit_transform(df['Medical History'].str.split(', ')),
#                                            columns=mlb.classes_,
#                                            index=df.index)
    
#     df = pd.concat([df, symptoms_encoded.add_prefix('Symptom_'), medical_history_encoded.add_prefix('History_')], axis=1)
#     df = df.drop(['Symptoms', 'Medical History'], axis=1)
    
#     # Encode Diagnosis
#     df['Diagnosis'] = pd.Categorical(df['Diagnosis']).codes
    
#     # Process Admission Date
#     df['Admission Date'] = pd.to_datetime(df['Admission Date'])
#     df['Admission_Year'] = df['Admission Date'].dt.year
#     df['Admission_Month'] = df['Admission Date'].dt.month
#     df['Admission_DayOfWeek'] = df['Admission Date'].dt.dayofweek
#     df = df.drop('Admission Date', axis=1)
    
#     return df

# def save_preprocessed_data(df, file_path):
#     """
#     Save the preprocessed data to a CSV file.
#     Parameters:
#         df (pd.DataFrame): The preprocessed data.
#         file_path (str): The path where the preprocessed data will be saved.
#     """
#     df.to_csv(file_path, index=False)
#     print(f"Preprocessed data saved to {file_path}")

# if __name__ == "__main__":
#     # Load raw data
#     df = pd.read_csv('MedQueue/data/patient_data.csv')
    
#     # Preprocess the data
#     df_preprocessed = preprocess_data(df)
    
#     # Save the preprocessed data
#     save_preprocessed_data(df_preprocessed, 'MedQueue/data/preprocessed_medical_data.csv')














import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    
    # Handling missing values
    # Drop the missing values columns

    numerical_columns = ['Heart Rate', 'Temperature', 'Respiratory Rate']
    categorical_columns = ['Symptoms', 'Medical History', 'Diagnosis']

    df = df.dropna(subset = numerical_columns + categorical_columns)

    # Convert 'Admission Date' to datetime and extract features
    df['Admission Date'] = pd.to_datetime(df['Admission Date'])
    df['Admission Year'] = df['Admission Date'].dt.year
    df['Admission Month'] = df['Admission Date'].dt.month
    df['Admission Day'] = df['Admission Date'].dt.day
    
    # Feature Engineering
    df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, np.inf], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    
    # Handle Blood Pressure (New)
    df['Systolic BP'], df['Diastolic BP'] = zip(*df['Blood Pressure'].str.split('/').tolist())
    df['Systolic BP'] = pd.to_numeric(df['Systolic BP'], errors='coerce')
    df['Diastolic BP'] = pd.to_numeric(df['Diastolic BP'], errors='coerce')
    
    # Fill NaN values in BP columns with median (New)
    df['Systolic BP'] = df['Systolic BP'].fillna(df['Systolic BP'].median())
    df['Diastolic BP'] = df['Diastolic BP'].fillna(df['Diastolic BP'].median())
    
    # Create Urgency Score based on multiple factors (Updated)
    df['Urgency_Score'] = (
        df['Systolic BP'] * 0.3 +
        df['Diastolic BP'] * 0.2 +
        df['Heart Rate'] * 0.3 +
        df['Temperature'] * 0.2
    )
    
    # Normalize numerical features (Updated)
    scaler = StandardScaler()
    numerical_columns = ['Systolic BP', 'Diastolic BP', 'Heart Rate', 'Temperature', 'Respiratory Rate', 'Urgency_Score']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Encode categorical features (Updated)
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Symptoms', 'Medical History', 'Diagnosis', 'Age_Bin']
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
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
    raw_data_path = os.path.join('..', '..', 'data','raw' , 'patient_data.csv')
    
    try:
        # Load raw data
        df = pd.read_csv(raw_data_path)
        
        # Preprocess the data
        df_preprocessed = preprocess_data(df)
        
        # Define the path for saving preprocessed data
        processed_data_path = os.path.join('..', '..', 'data','processed', 'preprocessed_medical_data.csv')
        
        # Save the preprocessed data
        save_preprocessed_data(df_preprocessed, processed_data_path)
        
    except FileNotFoundError:
        print(f"Error: The file {raw_data_path} was not found. Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")