from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from feature_extraction import append_trusted_domain
from log_utils import append_log_entry, build_log_entry
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency at runtime
    FastAPI = None
    BaseModel = object


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "detector.pkl"


if FastAPI is not None:
    class AnalyzeRequest(BaseModel):
        source: str
        content: str


    class PredictRequest(BaseModel):
        url: str


    class FalsePositiveRequest(BaseModel):
        url: str


    class LogRequest(BaseModel):
        input: str
        risk: str
        score: float
        ml_prob: float
        reasons: list[str]
        timestamp: str | None = None


    class BatchRequest(BaseModel):
        rows: list[AnalyzeRequest]


def load_system():
    with MODEL_FILE.open("rb") as model_file:
        return pickle.load(model_file)


if FastAPI is not None:
    app = FastAPI(title="Cybercrime Detection Backend", version="1.0.0")
    system = load_system()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/analyze")
    def analyze(payload: AnalyzeRequest) -> dict[str, object]:
        result = system.analyze(payload.source, payload.content)
        return {
            "source": result.source,
            "prediction": result.prediction,
            "probability": result.probability,
            "risk_score": result.risk_score,
            "risk_level": result.risk_level,
            "reasons": result.reasons,
            "debug": result.debug,
        }

    @app.post("/predict")
    def predict(payload: PredictRequest) -> dict[str, object]:
        result = system.analyze("url", payload.url)
        return {
            "risk_level": result.risk_level,
            "risk_score": result.risk_score,
            "ml_probability": result.probability,
            "reason": result.reasons,
            "prediction": result.prediction,
            "debug": result.debug,
        }

    @app.post("/feedback/false-positive")
    def report_false_positive(payload: FalsePositiveRequest) -> dict[str, object]:
        trusted_domain = append_trusted_domain(payload.url)
        return {
            "status": "trusted_domain_added",
            "trusted_domain": trusted_domain,
            "message": f"Added {trusted_domain} to trusted_domains.csv. Restart backend to apply it everywhere.",
        }

    @app.post("/log")
    def log_event(payload: LogRequest) -> dict[str, object]:
        entry = build_log_entry(
            input_value=payload.input,
            risk=payload.risk,
            score=payload.score,
            ml_prob=payload.ml_prob,
            reasons=payload.reasons,
            timestamp=payload.timestamp,
        )
        append_log_entry(entry)
        return {"status": "logged"}

    @app.post("/batch")
    def batch(payload: BatchRequest) -> list[dict[str, object]]:
        frame = pd.DataFrame([row.model_dump() for row in payload.rows])
        return system.analyze_batch(frame).to_dict(orient="records")
else:
    app = None
