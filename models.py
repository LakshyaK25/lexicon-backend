import requests
import os
import joblib
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# RoBERTa via HuggingFace API
HF_MODEL = "Kanjani25/roberta-fakenews" 
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL  = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# SVM + Logistic paths (loaded locally — files are tiny)
SVM_PATH      = "svm_model.joblib"
LOGISTIC_PATH = "logistic_model.joblib"
TFIDF_PATH    = "tfidf_vectorizer.joblib"

# Global holders
svm_model        = None
logistic_model   = None
tfidf_vectorizer = None


def load_models():
    global svm_model, logistic_model, tfidf_vectorizer
    status = {"roberta": False, "svm": False, "logistic": False}

    # Load SVM + TF-IDF
    try:
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        svm_model        = joblib.load(SVM_PATH)
        status["svm"]    = True
        print("✅ SVM loaded")
    except Exception as e:
        print(f"⚠️  SVM not loaded: {e}")

    # Load Logistic Regression
    try:
        logistic_model      = joblib.load(LOGISTIC_PATH)
        status["logistic"]  = True
        print("✅ Logistic Regression loaded")
    except Exception as e:
        print(f"⚠️  Logistic not loaded: {e}")

    # RoBERTa — via HF API (no local loading needed)
    if HF_TOKEN:
        status["roberta"] = True
        print("✅ RoBERTa API ready")
    else:
        print("⚠️  HF_TOKEN not set — RoBERTa unavailable")

    return status


def predict_roberta(text: str) -> dict:
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        raise ValueError(f"HF API error: {response.text}")

    result = response.json()
    top    = max(result[0], key=lambda x: x["score"])
    label  = top["label"]
    score  = top["score"]

    return {
        "verdict":    "FAKE" if label == "LABEL_1" else "REAL",
        "confidence": round(score, 4),
        "model":      "RoBERTa Base"
    }


def predict_svm(text: str) -> dict:
    if svm_model is None or tfidf_vectorizer is None:
        raise ValueError("SVM model not loaded")

    vec        = tfidf_vectorizer.transform([text])
    pred_label = int(svm_model.predict(vec)[0])

    try:
        score      = svm_model.decision_function(vec)[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))
    except Exception:
        confidence = 1.0

    return {
        "verdict":    "FAKE" if pred_label == 1 else "REAL",
        "confidence": round(confidence, 4),
        "model":      "SVM + TF-IDF"
    }


def predict_logistic(text: str) -> dict:
    if logistic_model is None or tfidf_vectorizer is None:
        raise ValueError("Logistic model not loaded")

    vec        = tfidf_vectorizer.transform([text])
    pred_label = int(logistic_model.predict(vec)[0])
    probs      = logistic_model.predict_proba(vec)[0]
    confidence = float(max(probs))

    return {
        "verdict":    "FAKE" if pred_label == 1 else "REAL",
        "confidence": round(confidence, 4),
        "model":      "Logistic Regression"
    }
