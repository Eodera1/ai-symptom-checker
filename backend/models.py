from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class PatientSymptom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fever = db.Column(db.Boolean)
    cough = db.Column(db.Boolean)
    fatigue = db.Column(db.Boolean)
    headache = db.Column(db.Boolean)
    nausea = db.Column(db.Boolean)
    prediction = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # 🕒 Auto timestamp
