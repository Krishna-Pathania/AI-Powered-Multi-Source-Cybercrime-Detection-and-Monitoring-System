from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlsplit, urlunsplit

import requests

try:
    from .config import AgentConfig
except ImportError:
    from config import AgentConfig


@dataclass(slots=True)
class PredictionResponse:
    risk_level: str
    risk_score: float
    ml_probability: float
    reasons: List[str] = field(default_factory=list)
    prediction: str = "Safe"


class BackendApiClient:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def predict_url(self, url: str) -> PredictionResponse | None:
        try:
            response = requests.post(
                self.config.backend_predict_url,
                json={"url": url},
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException:
            return None

        payload = response.json()
        reasons = payload.get("reason", [])
        if isinstance(reasons, str):
            reasons = [reasons]

        return PredictionResponse(
            risk_level=str(payload.get("risk_level", "Unknown")),
            risk_score=float(payload.get("risk_score", 0.0)),
            ml_probability=float(payload.get("ml_probability", 0.0)),
            reasons=[str(reason) for reason in reasons[:3]],
            prediction=str(payload.get("prediction", "Safe")),
        )

    def log_to_dashboard(self, result: dict) -> None:
        endpoint = self._log_endpoint()
        try:
            requests.post(
                endpoint,
                json=result,
                timeout=self.config.request_timeout_seconds,
            )
            print("LOG SENT:", result)
        except requests.RequestException as exc:
            print("LOG ERROR:", exc)

    def _log_endpoint(self) -> str:
        parts = urlsplit(self.config.backend_predict_url)
        path = parts.path
        if path.endswith("/predict"):
            path = f"{path[:-len('/predict')]}/log"
        else:
            path = "/log"
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
