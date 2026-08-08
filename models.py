import os
import joblib
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Model loads from HF Hub once at startup, runs locally after that
HF_MODEL = "Kanjani25/roberta-fakenews"  # ← your username

SVM_PATH      = "svm_model.joblib"
LOGISTIC_PATH = "logistic_model.joblib"
TFIDF_PATH    = "tfidf_vectorizer.joblib"

svm_model        = None
logistic_model   = None
tfidf_vectorizer = None
roberta_model    = None
roberta_tokenizer = None
device           = torch.device("cpu")


def load_models():
    global svm_model, logistic_model, tfidf_vectorizer
    global roberta_model, roberta_tokenizer
    status = {"roberta": False, "svm": False, "logistic": False}

    try:
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        svm_model        = joblib.load(SVM_PATH)
        status["svm"]    = True
        print("✅ SVM loaded")
    except Exception as e:
        print(f"⚠️  SVM not loaded: {e}")

    try:
        logistic_model     = joblib.load(LOGISTIC_PATH)
        status["logistic"] = True
        print("✅ Logistic Regression loaded")
    except Exception as e:
        print(f"⚠️  Logistic not loaded: {e}")

    try:
        print("Downloading RoBERTa from HuggingFace Hub...")
        roberta_tokenizer = RobertaTokenizer.from_pretrained(HF_MODEL)
        roberta_model     = RobertaForSequenceClassification.from_pretrained(HF_MODEL)
        roberta_model.to(device)
        roberta_model.eval()
        status["roberta"] = True
        print("✅ RoBERTa loaded locally")
    except Exception as e:
        print(f"⚠️  RoBERTa not loaded: {e}")

    return status


def predict_roberta(text: str) -> dict:
    if roberta_model is None:
        raise ValueError("RoBERTa model not loaded")

    inputs = roberta_tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs    = roberta_model(**inputs)
        probs      = torch.softmax(outputs.logits, dim=1)[0]
        pred_label = torch.argmax(probs).item()
        confidence = probs[pred_label].item()

    return {
        "verdict":    "FAKE" if pred_label == 1 else "REAL",
        "confidence": round(confidence, 4),
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
