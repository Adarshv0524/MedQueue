# evaluate.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score

# Load the test dataset
data = pd.read_csv('../../data/processed/selected_features_medical_data.csv')

# Define feature columns and target column
feature_columns = ['Age', 'Gender', 'Blood Pressure', 'Heart Rate', 'Respiratory Rate', 
                   'Medical History', 'Age_Bin', 'Systolic BP', 'Urgency_Score']
target_column = 'Diagnosis'

X = data[feature_columns]
y = data[target_column]

# Load the trained model and feature columns
model = joblib.load('../../data/models/decision_tree_model.pkl')

# Predict using the loaded model
y_pred = model.predict(X)

# Calculate and print performance metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'F1-Score: {f1}')
