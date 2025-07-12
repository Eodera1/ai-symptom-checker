from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from models import db, PatientSymptom  # Import SQLAlchemy model

app = Flask(__name__)
CORS(app)

# ✅ Correctly encoded password: "Password@123" → "Password%40123"
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://erickdev:Password%40123@localhost/ai_symptom_checker'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with Flask app
db.init_app(app)

# Create all tables
with app.app_context():
    db.drop_all()
    db.create_all()

# Load trained ML model
model = joblib.load("model.pkl")
symptoms = ["fever", "cough", "fatigue", "headache", "nausea"]

@app.route("/")
def home():
    return "Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_vector = [1 if s in data["symptoms"] else 0 for s in symptoms]
    prediction = model.predict([input_vector])[0]

    # Save prediction to database
    patient = PatientSymptom(
        fever="fever" in data["symptoms"],
        cough="cough" in data["symptoms"],
        fatigue="fatigue" in data["symptoms"],
        headache="headache" in data["symptoms"],
        nausea="nausea" in data["symptoms"],
        prediction=prediction
    )
    db.session.add(patient)
    db.session.commit()

    # Return response with prediction and timestamp
    return jsonify({
        "prediction": prediction,
        "timestamp": patient.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    })


if __name__ == "__main__":
    app.run(debug=True)
