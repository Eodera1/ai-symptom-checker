import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Synthetic data (10 symptoms matching the database)
data = np.array([
    [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # Flu
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],  # Cold
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],  # Migraine
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # Allergies
    [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]   # Gastroenteritis
])
y = ['Flu', 'Cold', 'Migraine', 'Allergies', 'Gastroenteritis']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data, y)

# Save the model
joblib.dump(model, 'model.pkl')