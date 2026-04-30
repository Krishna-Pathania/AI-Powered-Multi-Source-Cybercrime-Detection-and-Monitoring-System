from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(slots=True)
class AgentConfig:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    backend_predict_url: str = "http://localhost:5000/predict"
    scan_interval_seconds: float = 2.5
    request_timeout_seconds: float = 5.0
    start_monitoring_on_launch: bool = True
    notify_levels: Tuple[str, ...] = field(default_factory=lambda: ("High Risk",))
    nlp_alert_threshold: float = 0.7
    app_name: str = "Cybercrime Protection Agent"
    nlp_model_path: Path = field(init=False)
    sms_dataset_candidates: Tuple[Path, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.nlp_model_path = self.base_dir / "protection_agent" / "nlp_model.pkl"
        self.sms_dataset_candidates = (
            self.base_dir / "SMSSpamCollection",
            self.base_dir / "sms_dataset.csv",
        )
