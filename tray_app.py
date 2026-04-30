from __future__ import annotations

from PIL import Image, ImageDraw
import pystray


class TrayApplication:
    def __init__(self, controller) -> None:
        self.controller = controller
        self.icon = pystray.Icon(
            "cybercrime-protection-agent",
            self._build_icon(),
            "Cybercrime Protection Agent",
            menu=pystray.Menu(
                pystray.MenuItem("Start Monitoring", self._on_start_monitoring),
                pystray.MenuItem("Stop Monitoring", self._on_stop_monitoring),
                pystray.MenuItem("Show Status", self._on_show_status),
                pystray.MenuItem("Exit", self._on_exit),
            ),
        )

    def run(self) -> None:
        self.icon.run()

    def _on_start_monitoring(self, icon, item) -> None:
        self.controller.start_monitoring()
        self.controller.show_status_notification()

    def _on_stop_monitoring(self, icon, item) -> None:
        self.controller.stop_monitoring()
        self.controller.show_status_notification()

    def _on_show_status(self, icon, item) -> None:
        self.controller.show_status_notification()

    def _on_exit(self, icon, item) -> None:
        self.controller.shutdown()
        icon.stop()

    @staticmethod
    def _build_icon() -> Image.Image:
        image = Image.new("RGBA", (64, 64), (10, 15, 30, 255))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((8, 8, 56, 56), radius=14, fill=(13, 148, 136, 255))
        draw.polygon([(32, 14), (48, 21), (45, 39), (32, 51), (19, 39), (16, 21)], fill=(255, 255, 255, 255))
        draw.line((32, 21, 32, 39), fill=(13, 148, 136, 255), width=4)
        draw.ellipse((30, 42, 34, 46), fill=(13, 148, 136, 255))
        return image
