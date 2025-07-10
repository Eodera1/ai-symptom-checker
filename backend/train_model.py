import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("symptom_disease.csv")
X = df.drop("disease", axis=1)
y = df["disease"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved.")