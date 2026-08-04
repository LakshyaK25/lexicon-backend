from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models import load_models, predict_roberta, predict_svm, predict_logistic

app = FastAPI(title="Lexicon — Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_status = {}

@app.on_event("startup")
def startup_event():
    global model_status
    model_status = load_models()


class PredictRequest(BaseModel):
    text:  str
    model: str   # "roberta", "svm", or "logistic"


@app.get("/health")
def health():
    return {
        "status": "running",
        "models_loaded": model_status
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if req.model == "roberta":
        if not model_status.get("roberta"):
            raise HTTPException(status_code=503, detail="RoBERTa not available")
        return predict_roberta(req.text)

    elif req.model == "svm":
        if not model_status.get("svm"):
            raise HTTPException(status_code=503, detail="SVM not loaded")
        return predict_svm(req.text)

    elif req.model == "logistic":
        if not model_status.get("logistic"):
            raise HTTPException(status_code=503, detail="Logistic model not loaded")
        return predict_logistic(req.text)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Use 'roberta', 'svm', or 'logistic'"
        )
