# service/main.py
from fastapi import FastAPI
from pathlib import Path
import joblib
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict

from .schemas import TransactionInput, PredictionOutput, FeedbackInput

app = FastAPI(title="TransactAI Service")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
clf = joblib.load(MODELS_DIR / "logistic_regression.joblib")

with open(DATA_DIR / "taxonomy_config.yaml") as f:
    TAXONOMY: Dict = yaml.safe_load(f)

def predict_internal(text: str):
    clean = " ".join(text.lower().split())
    vec = tfidf.transform([clean])
    proba = clf.predict_proba(vec)[0]
    best_idx = proba.argmax()
    category = clf.classes_[best_idx]
    confidence = float(proba[best_idx])

    # explainability: top n features
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[best_idx]
    # contribution = feature_value * weight – approx
    feature_values = vec.toarray()[0]
    contrib = feature_values * coefs
    top_idx = contrib.argsort()[-5:][::-1]
    top_features = [feature_names[i] for i in top_idx if feature_values[i] > 0]

    return category, confidence, top_features

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TransactionInput):
    category, confidence, top_features = predict_internal(input_data.description)
    return PredictionOutput(
        category=category,
        confidence=confidence,
        top_features=top_features,
    )

@app.post("/feedback")
def feedback(data: FeedbackInput):
    feedback_path = DATA_DIR / "feedback.csv"
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "raw_description": data.description,
        "predicted_category": data.predicted_category,
        "predicted_confidence": data.predicted_confidence,
        "corrected_category": data.corrected_category,
        "user_comment": data.user_comment or "",
    }
    if feedback_path.exists():
        df = pd.read_csv(feedback_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(feedback_path, index=False)
    return {"status": "ok"}

@app.get("/categories")
def categories():
    return {"categories": TAXONOMY.get("categories", [])}
