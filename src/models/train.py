import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , f1_score
from sklearn.model_selection import train_test_split
import joblib
import logging

# Configure logging
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Load the selected feature dataset
data = pd.read_csv('../../data/processed/selected_features_medical_data.csv')

# Define feature columns and target column
feature_columns = ['Age', 'Gender', 'Blood Pressure', 'Heart Rate', 'Respiratory Rate', 
                   'Medical History', 'Age_Bin', 'Systolic BP', 'Urgency_Score']
target_column = 'Diagnosis'

X = data[feature_columns]
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Log the performance
logging.info(f'Model Accuracy: {accuracy}')
logging.info(f'Model F1-Score: {f1}')

# Save the trained model and the feature columns
joblib.dump(model, '../../data/models/decision_tree_model.pkl')
joblib.dump(feature_columns, '../../data/models/feature_columns.pkl')

logging.info('Model training completed and model saved successfully.')