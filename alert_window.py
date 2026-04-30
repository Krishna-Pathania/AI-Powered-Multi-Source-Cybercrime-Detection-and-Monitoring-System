from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass


@dataclass(slots=True)
class AlertPayload:
    message: str
    title: str = "Cyber Alert"
    auto_close_ms: int = 6000


class AlertWindowManager:
    def __init__(self) -> None:
        self._queue: queue.Queue[AlertPayload] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()

    def show_alert(self, message: str, title: str = "Cyber Alert", auto_close_ms: int = 6000) -> None:
        self._ensure_worker()
        self._queue.put(AlertPayload(message=message, title=title, auto_close_ms=auto_close_ms))

    def _ensure_worker(self) -> None:
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run_loop, name="alert-window", daemon=True)
            self._thread.start()

    def _run_loop(self) -> None:
        root = tk.Tk()
        root.withdraw()
        root.after(150, lambda: self._drain_queue(root))
        root.mainloop()

    def _drain_queue(self, root: tk.Tk) -> None:
        try:
            while True:
                payload = self._queue.get_nowait()
                self._create_popup(root, payload)
        except queue.Empty:
            pass
        root.after(250, lambda: self._drain_queue(root))

    @staticmethod
    def _create_popup(root: tk.Tk, payload: AlertPayload) -> None:
        window = tk.Toplevel(root)
        window.title(payload.title)
        window.configure(bg="#8b0000")
        window.attributes("-topmost", True)
        window.resizable(False, False)

        width = 460
        height = 220
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x_pos = int((screen_width - width) / 2)
        y_pos = int((screen_height - height) / 2)
        window.geometry(f"{width}x{height}+{x_pos}+{y_pos}")

        frame = tk.Frame(window, bg="#8b0000", padx=20, pady=18)
        frame.pack(fill="both", expand=True)

        heading = tk.Label(
            frame,
            text="CYBER THREAT DETECTED",
            fg="white",
            bg="#8b0000",
            font=("Segoe UI", 16, "bold"),
        )
        heading.pack(anchor="w")

        body = tk.Label(
            frame,
            text=payload.message,
            fg="white",
            bg="#8b0000",
            justify="left",
            wraplength=410,
            font=("Segoe UI", 11),
            pady=18,
        )
        body.pack(anchor="w")

        close_button = tk.Button(
            frame,
            text="Close",
            command=window.destroy,
            bg="white",
            fg="#8b0000",
            activebackground="#fee2e2",
            activeforeground="#7f1d1d",
            relief="flat",
            padx=18,
            pady=8,
            font=("Segoe UI", 10, "bold"),
        )
        close_button.pack(anchor="e")

        window.after(payload.auto_close_ms, window.destroy)


_ALERT_MANAGER = AlertWindowManager()


def show_alert(message: str, title: str = "Cyber Alert", auto_close_ms: int = 6000) -> None:
    _ALERT_MANAGER.show_alert(message=message, title=title, auto_close_ms=auto_close_ms)
