from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .api_client import BackendApiClient
    from .clipboard_monitor import ClipboardMonitor
    from .config import AgentConfig
    from .nlp_model import NLPModel
    from .notifier import DesktopNotifier
    from .tray_app import TrayApplication
except ImportError:
    from api_client import BackendApiClient
    from clipboard_monitor import ClipboardMonitor
    from config import AgentConfig
    from nlp_model import NLPModel
    from notifier import DesktopNotifier
    from tray_app import TrayApplication
from log_utils import build_log_entry


class CybercrimeProtectionAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.notifier = DesktopNotifier(config.app_name)
        self.api_client = BackendApiClient(config)
        self.nlp_model = NLPModel(config)
        self.nlp_model.load_or_train()
        self.clipboard_monitor = ClipboardMonitor(config, self._handle_clipboard_content)
        self._lock = threading.Lock()
        self._last_notified_signature = ""

    def start_monitoring(self) -> None:
        self.clipboard_monitor.start()

    def stop_monitoring(self) -> None:
        self.clipboard_monitor.stop()

    def shutdown(self) -> None:
        self.stop_monitoring()

    def status_text(self) -> str:
        state = "ON" if self.clipboard_monitor.is_running else "OFF"
        return f"Clipboard monitoring is {state}. Backend: {self.config.backend_predict_url}"

    def show_status_notification(self) -> None:
        self.notifier.show(self.config.app_name, self.status_text())

    def _handle_clipboard_content(self, clipboard_text: str, extracted_url: str | None) -> None:
        if extracted_url:
            self._handle_clipboard_url(extracted_url)
            return
        self._handle_clipboard_text(clipboard_text)

    def _handle_clipboard_url(self, url: str) -> None:
        prediction = self.api_client.predict_url(url)
        if prediction is None:
            print(f"[ProtectionAgent] API failure while checking URL: {url}")
            return

        result_payload = build_log_entry(
            input_value=url,
            risk=prediction.risk_level,
            score=prediction.risk_score,
            ml_prob=prediction.ml_probability,
            reasons=prediction.reasons,
        )
        print(f"[ProtectionAgent] Prediction result: {result_payload}")
        self.api_client.log_to_dashboard(result_payload)

        if result_payload["risk"].lower() != "high risk":
            return

        with self._lock:
            if url == self._last_notified_signature:
                return
            self._last_notified_signature = url

        summary = prediction.reasons[0] if prediction.reasons else "The backend marked this URL as dangerous."
        message = f"{url}\nRisk: {prediction.risk_level} ({prediction.risk_score:.1f}/10)\n{summary}"
        print(f"[ProtectionAgent] Alert trigger: {message}")
        self.notifier.show("Cyber Alert", message, popup=True)

    def _handle_clipboard_text(self, text: str) -> None:
        if not text.strip():
            return

        prediction = self.nlp_model.analyze_text(text)
        if prediction.scam_probability >= self.config.nlp_alert_threshold:
            risk_level = "High Risk"
        elif prediction.scam_probability >= 0.45:
            risk_level = "Suspicious"
        else:
            risk_level = "Safe"

        result_payload = build_log_entry(
            input_value=text,
            risk=risk_level,
            score=round(prediction.scam_probability * 10, 2),
            ml_prob=prediction.scam_probability,
            reasons=[
                "Local NLP scam detector analyzed copied text.",
                f"Predicted label: {prediction.label}",
            ],
        )
        print(f"[ProtectionAgent] Prediction result: {result_payload}")
        self.api_client.log_to_dashboard(result_payload)
        if result_payload["risk"].lower() != "high risk":
            return

        with self._lock:
            if text == self._last_notified_signature:
                return
            self._last_notified_signature = text

        message = (
            f"Scam probability: {prediction.scam_probability:.0%}\n"
            f"Copied content: {text[:160]}"
        )
        print(f"[ProtectionAgent] Alert trigger: {message}")
        self.notifier.show("Cyber Alert", message, popup=True)


def main() -> None:
    config = AgentConfig()
    controller = CybercrimeProtectionAgent(config)
    if config.start_monitoring_on_launch:
        controller.start_monitoring()
    TrayApplication(controller).run()


if __name__ == "__main__":
    main()
