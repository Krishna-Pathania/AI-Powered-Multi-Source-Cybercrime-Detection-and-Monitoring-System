from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
LOGS_FILE = BASE_DIR / "logs.json"


def normalize_risk_label(risk: str) -> str:
    mapping = {
        "High": "High Risk",
        "Medium": "Suspicious",
        "Low": "Safe",
        "High Risk": "High Risk",
        "Suspicious": "Suspicious",
        "Safe": "Safe",
    }
    return mapping.get(str(risk or "").strip(), str(risk or "Unknown").strip() or "Unknown")


def build_log_entry(
    *,
    input_value: str,
    risk: str,
    score: float,
    ml_prob: float,
    reasons: list[str] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "input": str(input_value or ""),
        "risk": normalize_risk_label(risk),
        "score": float(score or 0.0),
        "ml_prob": float(ml_prob or 0.0),
        "reasons": [str(reason) for reason in (reasons or [])],
    }


def append_log_entry(entry: dict[str, Any]) -> None:
    logs: list[dict[str, Any]]
    if LOGS_FILE.exists():
        try:
            with LOGS_FILE.open("r", encoding="utf-8") as logs_file:
                payload = json.load(logs_file)
            if isinstance(payload, list):
                logs = [item for item in payload if isinstance(item, dict)]
            elif isinstance(payload, dict):
                logs = [payload]
            else:
                logs = []
        except (OSError, json.JSONDecodeError):
            logs = []
    else:
        logs = []

    if logs and _is_recent_duplicate(logs[-1], entry):
        return

    logs.append(entry)
    with LOGS_FILE.open("w", encoding="utf-8") as logs_file:
        json.dump(logs, logs_file, indent=2)


def _is_recent_duplicate(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    comparable_keys = ("input", "risk", "score", "ml_prob", "reasons")
    for key in comparable_keys:
        if previous.get(key) != current.get(key):
            return False

    previous_timestamp = previous.get("timestamp")
    current_timestamp = current.get("timestamp")
    try:
        previous_dt = datetime.fromisoformat(str(previous_timestamp))
        current_dt = datetime.fromisoformat(str(current_timestamp))
    except ValueError:
        return False

    return abs((current_dt - previous_dt).total_seconds()) <= 5
