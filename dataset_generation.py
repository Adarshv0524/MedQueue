import csv
import random
from faker import Faker

fake = Faker()

def generate_patient_data(num_rows):
    data = []
    for i in range(1, num_rows + 1):  # Start from 1 to num_rows
        patient_id = i  # Use a serial number for patient ID
        
        # Introduce some outliers in age
        if random.random() < 0.01:  # 1% chance to be an outlier
            age = random.randint(101, 120)  # Outlier age
        else:
            age = random.randint(0, 100)
        
        gender = random.choice(['Male', 'Female', 'Other'])
        
        # Introduce some missing values in blood pressure and some outliers
        if random.random() < 0.02:  # 2% chance to be missing
            blood_pressure = None
        elif random.random() < 0.01:  # 1% chance to be an outlier
            blood_pressure = f"{random.randint(180, 200)}/{random.randint(120, 140)}"
        else:
            blood_pressure = f"{random.randint(90, 140)}/{random.randint(60, 90)}"
        
        heart_rate = random.randint(60, 100)
        
        # Introduce temperature outliers and missing values
        if random.random() < 0.02:  # 2% chance to be an outlier
            temperature = round(random.uniform(30.0, 45.0), 1)
        elif random.random() < 0.01:  # 1% chance to be missing
            temperature = None
        else:
            temperature = round(random.uniform(36.0, 39.0), 1)
        
        respiratory_rate = random.randint(12, 20)
        
        # Introduce some missing values in symptoms
        if random.random() < 0.01:  # 1% chance to be missing
            symptoms = None
        else:
            symptoms = random.choice(['Cough', 'Fever', 'Headache', 'Fatigue', 'None'])
        
        medical_history = random.choice(['Diabetes', 'Hypertension', 'Asthma', 'None'])
        
        # Introduce outliers in diagnosis by repeating a rare condition
        if random.random() < 0.05:  # 5% chance to be an outlier
            diagnosis = 'Rare Condition'
        else:
            diagnosis = random.choice([
                'Flu', 'Common Cold', 'COVID-19', 'Healthy', 'Pneumonia', 'Bronchitis', 
                'Asthma', 'Hypertension', 'Diabetes', 'Heart Disease', 'Kidney Disease', 
                'Liver Disease', 'Cancer', 'Stroke', 'Migraine', 'Arthritis', 'Allergies'
            ])
        
        admission_date = fake.date_this_year()
        
        data.append([
            patient_id, age, gender, blood_pressure, heart_rate, temperature, 
            respiratory_rate, symptoms, medical_history, diagnosis, admission_date
        ])
    
    return data

def write_to_csv(filename, data):
    header = [
        "Patient ID", "Age", "Gender", "Blood Pressure", "Heart Rate", "Temperature", 
        "Respiratory Rate", "Symptoms", "Medical History", "Diagnosis", "Admission Date"
    ]
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

# Generate data and write to CSV
num_rows = 2000
data = generate_patient_data(num_rows)
write_to_csv('patient_data.csv', data)

print("Dataset generated and saved to 'patient_data.csv'")
