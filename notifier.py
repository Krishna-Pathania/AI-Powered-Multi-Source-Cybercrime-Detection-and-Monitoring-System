from __future__ import annotations

import threading

try:
    from .alert_window import show_alert
except ImportError:
    from alert_window import show_alert

try:
    from plyer import notification
except Exception:  # pragma: no cover - optional at runtime
    notification = None

try:
    from win10toast import ToastNotifier
except Exception:  # pragma: no cover - optional at runtime
    ToastNotifier = None


class DesktopNotifier:
    def __init__(self, app_name: str) -> None:
        self.app_name = app_name
        self.toast_notifier = ToastNotifier() if ToastNotifier is not None else None

    def show(self, title: str, message: str, popup: bool = False) -> None:
        if popup:
            self.safe_alert(message=message, title=title)

        if self.toast_notifier is not None:
            try:
                self.toast_notifier.show_toast(title, message, duration=6, threaded=True)
                return
            except Exception:
                pass

        if notification is None:
            print(f"[{self.app_name}] {title}: {message}")
            return

        notification.notify(
            title=title,
            message=message,
            app_name=self.app_name,
            timeout=6,
        )

    @staticmethod
    def safe_alert(message: str, title: str = "Cyber Alert") -> None:
        threading.Thread(target=show_alert, args=(message, title), daemon=True).start()
