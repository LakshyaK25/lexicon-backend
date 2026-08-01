import requests
import os

HF_MODEL = "Kanjani25/roberta-fakenews"  # ← replace
HF_TOKEN = os.environ.get("HF_TOKEN")            # loaded from Render env variable
API_URL  = f"https://api-inference.huggingface.co/models/{HF_MODEL}"


def load_models():
    # No model loaded locally — HF API handles it
    print("✅ Using HuggingFace Inference API")
    return {"roberta": True}


def predict_roberta(text: str) -> dict:
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        raise ValueError(f"HF API error: {response.text}")

    result = response.json()

    # HF returns list of label/score dicts
    # e.g. [{"label": "LABEL_1", "score": 0.98}, {"label": "LABEL_0", "score": 0.02}]
    top    = max(result[0], key=lambda x: x["score"])
    label  = top["label"]   # "LABEL_0" = real, "LABEL_1" = fake
    score  = top["score"]

    return {
        "verdict":    "FAKE" if label == "LABEL_1" else "REAL",
        "confidence": round(score, 4),
        "model":      "RoBERTa Base"
    }