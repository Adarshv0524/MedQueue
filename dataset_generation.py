import csv
import random
from faker import Faker

fake = Faker()

def generate_patient_data(num_rows):
    data = []
    for _ in range(num_rows):
        patient_id = fake.unique.uuid4()
        age = random.randint(0, 100)
        gender = random.choice(['Male', 'Female', 'Other'])
        blood_pressure = f"{random.randint(90, 140)}/{random.randint(60, 90)}"
        heart_rate = random.randint(60, 100)
        temperature = round(random.uniform(36.0, 39.0), 1)
        respiratory_rate = random.randint(12, 20)
        symptoms = random.choice(['Cough', 'Fever', 'Headache', 'Fatigue', 'None'])
        medical_history = random.choice(['Diabetes', 'Hypertension', 'Asthma', 'None'])
        diagnosis = random.choice([
            'Flu', 'Common Cold', 'COVID-19', 'Healthy', 'Pneumonia', 'Bronchitis', 
            'Asthma', 'Hypertension', 'Diabetes', 'Heart Disease', 'Kidney Disease', 
            'Liver Disease', 'Cancer', 'Stroke', 'Migraine', 'Arthritis', 'Allergies'
        ])
        admission_date = fake.date_this_year()
        
        data.append([patient_id, age, gender, blood_pressure, heart_rate, temperature, respiratory_rate, symptoms, medical_history, diagnosis, admission_date])
    
    return data

def write_to_csv(filename, data):
    header = ["Patient ID", "Age", "Gender", "Blood Pressure", "Heart Rate", "Temperature", "Respiratory Rate", "Symptoms", "Medical History", "Diagnosis", "Admission Date"]
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

# Generate data and write to CSV
num_rows = 2000
data = generate_patient_data(num_rows)
write_to_csv('patient_data.csv', data)

print("Dataset generated and saved to 'patient_data.csv'")
