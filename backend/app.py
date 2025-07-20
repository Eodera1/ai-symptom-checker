from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from cryptography.fernet import Fernet
import joblib
import numpy as np
import logging
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    'mysql+pymysql://erickdev:Password%40123@localhost/ai_symptom_checker'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Encryption setup
ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY', Fernet.generate_key())
cipher = Fernet(ENCRYPTION_KEY)

# Database models
class Symptom(db.Model):
    __tablename__ = 'symptoms'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)

class Condition(db.Model):
    __tablename__ = 'conditions'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    severity = db.Column(db.Enum('Mild', 'Moderate', 'Severe'), nullable=False)

class PatientSymptom(db.Model):
    __tablename__ = 'patient_symptoms'
    id = db.Column(db.Integer, primary_key=True)
    encrypted_symptoms = db.Column(db.LargeBinary, nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

class PatientSymptomMapping(db.Model):
    __tablename__ = 'patient_symptom_mapping'
    id = db.Column(db.Integer, primary_key=True)
    patient_symptom_id = db.Column(db.Integer, db.ForeignKey('patient_symptoms.id'), nullable=False)
    symptom_id = db.Column(db.Integer, db.ForeignKey('symptoms.id'), nullable=False)
    severity = db.Column(db.Enum('Mild', 'Moderate', 'Severe'), nullable=False)

# Load ML model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise

# Initialize database
with app.app_context():
    db.create_all()
    # Seed symptoms if empty
    if not Symptom.query.first():
        initial_symptoms = [
            {'name': 'Fever', 'description': 'Elevated body temperature above 38°C'},
            {'name': 'Cough', 'description': 'Persistent or occasional coughing'},
            {'name': 'Fatigue', 'description': 'Extreme tiredness or lack of energy'},
            {'name': 'Headache', 'description': 'Pain in the head or neck'},
            {'name': 'Nausea', 'description': 'Feeling of sickness or urge to vomit'},
            {'name': 'Sore Throat', 'description': 'Pain or irritation in the throat'},
            {'name': 'Shortness of Breath', 'description': 'Difficulty breathing'},
            {'name': 'Muscle Pain', 'description': 'Aches or soreness in muscles'},
            {'name': 'Chills', 'description': 'Feeling cold with shivering'},
            {'name': 'Loss of Taste', 'description': 'Inability to taste flavors'}
        ]
        for s in initial_symptoms:
            db.session.add(Symptom(**s))
        db.session.commit()
        logging.info("Initialized symptoms in database")

@app.route("/api/", methods=["GET"])
def home():
    return jsonify({"message": "AI Symptom Checker API is running!", "version": "1.0.0"})

@app.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    try:
        symptoms = Symptom.query.all()
        return jsonify([
            {"id": s.id, "name": s.name, "description": s.description}
            for s in symptoms
        ])
    except Exception as e:
        logging.error(f"Error fetching symptoms: {str(e)}")
        return jsonify({"error": "Failed to fetch symptoms"}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data or 'severities' not in data:
            logging.warning("Invalid input data")
            return jsonify({"error": "Missing symptoms or severities"}), 400

        symptom_ids = data["symptoms"]  # List of symptom IDs
        severities = data["severities"]  # List of severities (Mild, Moderate, Severe)

        # Validate input
        if len(symptom_ids) == 0:
            logging.warning("No symptoms provided")
            return jsonify({"error": "At least one symptom is required"}), 400
        if len(symptom_ids) != len(severities):
            logging.warning("Mismatched symptoms and severities")
            return jsonify({"error": "Each symptom must have a severity"}), 400

        # Create input vector as a NumPy array
        symptom_count = Symptom.query.count()
        input_vector = np.zeros(symptom_count)
        for sid in symptom_ids:
            if 1 <= sid <= symptom_count:
                input_vector[sid - 1] = 1
            else:
                logging.warning(f"Invalid symptom ID: {sid}")
                return jsonify({"error": f"Invalid symptom ID: {sid}"}), 400

        # Predict
        prediction_proba = model.predict_proba([input_vector])[0]
        predictions = [
            {"condition": model.classes_[i], "probability": float(prob)}
            for i, prob in enumerate(prediction_proba)
        ]
        top_prediction = max(predictions, key=lambda x: x["probability"])["condition"]

        # Save to database
        encrypted_symptoms = cipher.encrypt(str(symptom_ids).encode())
        patient = PatientSymptom(
            encrypted_symptoms=encrypted_symptoms,
            prediction=top_prediction
        )
        db.session.add(patient)
        db.session.flush()  # Get patient ID before commit

        # Save symptom mappings
        for sid, severity in zip(symptom_ids, severities):
            if severity not in ['Mild', 'Moderate', 'Severe']:
                logging.warning(f"Invalid severity: {severity}")
                return jsonify({"error": f"Invalid severity: {severity}"}), 400
            mapping = PatientSymptomMapping(
                patient_symptom_id=patient.id,
                symptom_id=sid,
                severity=severity
            )
            db.session.add(mapping)

        db.session.commit()
        logging.info(f"Saved prediction for patient_symptom_id: {patient.id}")

        return jsonify({
            "predictions": predictions,
            "timestamp": patient.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        db.session.rollback()
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/api/analytics", methods=["GET"])
def analytics():
    try:
        common_symptoms = db.session.query(
            Symptom.name,
            db.func.count(PatientSymptomMapping.id).label('count')
        ).join(PatientSymptomMapping).group_by(Symptom.name).all()
        return jsonify([
            {"symptom": name, "count": count}
            for name, count in common_symptoms
        ])
    except Exception as e:
        logging.error(f"Analytics error: {str(e)}")
        return jsonify({"error": "Failed to fetch analytics"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)