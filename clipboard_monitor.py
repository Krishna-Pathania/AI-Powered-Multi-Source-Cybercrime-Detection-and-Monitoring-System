from __future__ import annotations

import re
import threading
import time
from typing import Callable

import pyperclip

try:
    from .config import AgentConfig
except ImportError:
    from config import AgentConfig


URL_PATTERN = re.compile(r"(https?://[^\s]+)")


class ClipboardMonitor:
    def __init__(self, config: AgentConfig, content_callback: Callable[[str, str | None], None]) -> None:
        self.config = config
        self.content_callback = content_callback
        self._running = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_clipboard_value = ""

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        if self.is_running:
            return
        self._running.set()
        self._worker = threading.Thread(target=self._loop, name="clipboard-monitor", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running.clear()

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                raw_clipboard_value = pyperclip.paste()
            except pyperclip.PyperclipException:
                time.sleep(self.config.scan_interval_seconds)
                continue

            clipboard_value = str(raw_clipboard_value or "").strip()
            if not clipboard_value:
                time.sleep(self.config.scan_interval_seconds)
                continue

            if clipboard_value != self._last_clipboard_value:
                self._last_clipboard_value = clipboard_value
                extracted_url = self._extract_url(clipboard_value)
                print(f"[ClipboardMonitor] Clipboard content: {clipboard_value}")
                print(f"[ClipboardMonitor] Extracted URL: {extracted_url or 'None'}")
                self.content_callback(clipboard_value, extracted_url)

            time.sleep(self.config.scan_interval_seconds)

    @staticmethod
    def _extract_url(value: str) -> str | None:
        match = URL_PATTERN.search(value)
        if not match:
            return None
        return match.group(1).strip()
