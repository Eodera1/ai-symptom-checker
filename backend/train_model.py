import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate a larger synthetic dataset (100 samples, 10 symptoms)
np.random.seed(42)
num_samples = 100
data = np.random.randint(0, 2, size=(num_samples, 10))  # Binary symptoms
conditions = ['Flu', 'Cold', 'Migraine', 'Allergies', 'Gastroenteritis']
y = np.random.choice(conditions, size=num_samples)

# Adjust data to reflect realistic symptom patterns
for i in range(num_samples):
    if y[i] == 'Flu':
        data[i, [0, 1, 8]] = 1  # Fever, Cough, Chills
    elif y[i] == 'Cold':
        data[i, [1, 5, 6]] = 1  # Cough, Sore Throat, Shortness of Breath
    elif y[i] == 'Migraine':
        data[i, [3, 2, 9]] = 1  # Headache, Fatigue, Loss of Taste
    elif y[i] == 'Allergies':
        data[i, [4, 5]] = 1     # Nausea, Sore Throat
    elif y[i] == 'Gastroenteritis':
        data[i, [4, 7, 8]] = 1  # Nausea, Muscle Pain, Chills

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data, y)

# Save the model
joblib.dump(model, 'model.pkl')