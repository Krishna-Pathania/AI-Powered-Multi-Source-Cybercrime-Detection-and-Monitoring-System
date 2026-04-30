from __future__ import annotations

import re
from typing import Dict, List


NLP_SIGNAL_GROUPS = {
    "urgency": ["urgent", "immediately", "asap", "now", "today", "final warning", "limited time"],
    "credential_request": ["login", "password", "verify", "otp", "pin", "confirm account", "cvv"],
    "financial_lure": ["refund", "payment", "bank", "wallet", "gift card", "prize", "reward", "bonus"],
    "action_request": ["click", "open", "download", "tap", "install", "reset", "update"],
    "social_engineering": ["confidential", "secret", "kindly", "dear user", "dear customer", "reply now"],
}


class NLPThreatAnalyzer:
    def analyze(self, text: str) -> Dict[str, object]:
        lowered = str(text or "").lower()
        reasons: List[str] = []
        triggered_signals: List[Dict[str, object]] = []
        total_score = 0

        for group, keywords in NLP_SIGNAL_GROUPS.items():
            matched = [keyword for keyword in keywords if keyword in lowered]
            if matched:
                group_score = min(2, len(matched))
                total_score += group_score
                reasons.append(f"{group.replace('_', ' ').title()} language detected: {', '.join(matched[:4])}.")
                triggered_signals.append({"group": group, "keywords": matched[:6], "score": group_score})

        uppercase_ratio = self._uppercase_ratio(text)
        exclamation_count = str(text).count("!")
        if uppercase_ratio >= 0.2 and len(str(text)) >= 20:
            total_score += 1
            reasons.append("Excessive uppercase emphasis detected.")
        if exclamation_count >= 3:
            total_score += 1
            reasons.append("Aggressive punctuation detected.")

        return {
            "score": min(total_score, 6),
            "reasons": reasons,
            "signals": triggered_signals,
            "uppercase_ratio": round(uppercase_ratio, 4),
            "exclamation_count": exclamation_count,
        }

    @staticmethod
    def _uppercase_ratio(text: str) -> float:
        letters = re.findall(r"[A-Za-z]", str(text or ""))
        if not letters:
            return 0.0
        uppercase = [letter for letter in letters if letter.isupper()]
        return len(uppercase) / len(letters)
