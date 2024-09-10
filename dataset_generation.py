import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Random seed for reproducibility
random.seed(41)

# Function to generate a realistic random date within the past year
def generate_random_date():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')

# Lists of realistic symptoms, medical history, and diagnoses
symptoms = ['Headache', 'Cough', 'Fever', 'Shortness of Breath', 'Fatigue', 'Chest Pain', 'Nausea', 'None']
medical_history = ['Diabetes', 'Hypertension', 'Asthma', 'COVID-19', 'Heart Disease', 'Stroke', 'None']
diagnoses = ['Asthma', 'Diabetes', 'Hypertension', 'COVID-19', 'Flu', 'Stroke', 'Healthy']

# Generate realistic blood pressure values
def generate_bp(age, history):
    # Base ranges for normal and hypertensive individuals
    if 'Hypertension' in history or age > 60:
        base_systolic = random.randint(130, 160)
        base_diastolic = random.randint(80, 100)
    else:
        base_systolic = random.randint(110, 130)
        base_diastolic = random.randint(70, 85)
    
    # Adjust systolic based on diastolic to maintain realistic ratios
    diastolic = base_diastolic + random.randint(-5, 5)
    systolic = diastolic + random.randint(30, 50)
    
    # Ensure systolic is always higher than diastolic
    if systolic <= diastolic:
        systolic = diastolic + random.randint(30, 50)
    
    return systolic, diastolic

# Generate realistic heart rate
def generate_heart_rate(temperature, symptoms, history):
    if temperature is None:
        return None
    if 'Fever' in symptoms or temperature > 37.5:
        return random.randint(90, 160)  # Wider range for fever
    elif 'Hypothermia' in symptoms or temperature < 35.0:
        return random.randint(40, 70)  # Adjusted range for hypothermia
    elif 'Heart Disease' in history:
        return random.randint(60, 140)  # Wider range for heart disease
    else:
        return random.randint(60, 100)  # Normal range

# Generate realistic temperature
def generate_temperature(symptoms, diagnosis):
    if diagnosis == 'COVID-19':
        return round(random.uniform(37.5, 40.0), 1)  # Fever range
    elif 'Fever' in symptoms:
        return round(random.uniform(37.5, 39.0), 1)  # Fever range
    elif 'Hypothermia' in symptoms:
        return round(random.uniform(33.0, 35.0), 1)  # Hypothermia range
    else:
        return round(random.uniform(36.5, 37.5), 1)  # Normal range

# Generate realistic oxygen saturation levels
def generate_oxygen_saturation(symptoms, history, diagnosis):
    # Base normal oxygen saturation level
    normal_oxygen = random.randint(95, 100)

    # Adjust saturation based on symptoms and history
    if 'COVID-19' in history or diagnosis == 'COVID-19':
        return random.randint(88, 94)  # Mildly lower oxygen levels for COVID-19
    elif 'Asthma' in history or diagnosis == 'Asthma' or 'Shortness of Breath' in symptoms:
        return random.randint(85, 94)  # Reduced oxygen for respiratory issues
    elif 'Heart Disease' in history or diagnosis == 'Heart Disease':
        return random.randint(90, 95)  # Mild reduction for heart-related issues
    else:
        return normal_oxygen  # Normal oxygen levels for healthy individuals




# Generate realistic respiratory rate
def generate_respiratory_rate(symptoms):
    if 'Shortness of Breath' in symptoms:
        return random.randint(18, 30)  # Elevated RR for breathing issues
    elif 'Fatigue' in symptoms:
        return random.randint(10, 16)  # Lower RR for fatigue
    else:
        return random.randint(12, 20)  # Normal RR

# Add realistic outliers
def add_outliers(value, feature):
    if feature == 'Blood Pressure':
        if random.random() < 0.05:  # 5% chance to add outlier
            return (random.randint(180, 250), random.randint(120, 160))
    elif feature == 'Heart Rate':
        if random.random() < 0.05:
            return random.randint(30, 200)
    elif feature == 'Temperature':
        if random.random() < 0.05:
            return round(random.uniform(31.0, 43.0), 1)
    elif feature == 'Respiratory Rate':
        if random.random() < 0.05:
            return random.randint(5, 40)
    elif feature == 'Oxygen Saturation':
        if random.random() < 0.05:
            return random.randint(60, 85)  # Severe outlier in SpO2
        
    return value


# Generate dataset
data = []
for i in range(1, 11):
    age = random.randint(1, 90)
    gender = random.choice(['Male', 'Female', 'Other'])
    symptom = random.choice(symptoms)
    history = random.choice(medical_history)
    diagnosis = random.choice(diagnoses)
    admission_date = generate_random_date()

    # Generate BP values
    systolic_bp, diastolic_bp = generate_bp(age, history)
    systolic_bp, diastolic_bp = add_outliers((systolic_bp, diastolic_bp), 'Blood Pressure')
    
    # Generate temperature and associated health metrics
    temperature = generate_temperature(symptom, diagnosis)
    temperature = add_outliers(temperature, 'Temperature')
    heart_rate = generate_heart_rate(temperature, symptom, history)
    heart_rate = add_outliers(heart_rate, 'Heart Rate')
    respiratory_rate = generate_respiratory_rate(symptom)
    respiratory_rate = add_outliers(respiratory_rate, 'Respiratory Rate')

    # Generate oxygen saturation
    oxygen_saturation = generate_oxygen_saturation(symptom, history, diagnosis)


    # Simulate a small amount of missing data
    if random.random() < 0.05:  # 5% chance to add null values
        temperature = None
    if random.random() < 0.05:
        heart_rate = None

    data.append([i, age, gender, f'{systolic_bp}/{diastolic_bp}' if systolic_bp and diastolic_bp else None, heart_rate, temperature,oxygen_saturation, respiratory_rate, symptom, history, diagnosis, admission_date])

# Convert to DataFrame
columns = ['Patient ID', 'Age', 'Gender', 'Blood Pressure', 'Heart Rate', 'Temperature', 'Oxygen Saturation', 'Respiratory Rate', 'Symptoms', 'Medical History', 'Diagnosis', 'Admission Date']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('data/raw/data.csv', index=False)

