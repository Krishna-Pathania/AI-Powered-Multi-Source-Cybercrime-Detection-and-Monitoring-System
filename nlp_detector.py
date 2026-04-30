from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PATTERN_GROUPS = {
    "urgency": [
        r"\bact now\b",
        r"\blimited time\b",
        r"\burgent\b",
        r"\bimmediately\b",
        r"\basap\b",
        r"\btoday only\b",
        r"\bfinal warning\b",
    ],
    "threat": [
        r"\baccount blocked\b",
        r"\baccount suspended\b",
        r"\bsuspended\b",
        r"\bblocked\b",
        r"\bdisabled\b",
        r"\bsecurity alert\b",
        r"\bunauthorized\b",
    ],
    "reward": [
        r"\byou won\b",
        r"\bfree gift\b",
        r"\bclaim reward\b",
        r"\bbonus\b",
        r"\bprize\b",
        r"\breward\b",
        r"\bgift card\b",
    ],
    "credential_request": [
        r"\blogin\b",
        r"\bverify\b",
        r"\botp\b",
        r"\bpin\b",
        r"\bpassword\b",
        r"\bcvv\b",
    ],
}


@dataclass
class NLPDetectionResult:
    scam_probability: float
    detected_patterns: List[str]
    matched_keywords: Dict[str, List[str]]


class NLPScamDetector:
    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=20000,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
            ]
        )
        self.is_trained = False

    def train(self, df: pd.DataFrame) -> None:
        working = df.copy()
        working["text"] = working["text"].astype(str).str.strip()
        working["label"] = working["label"].astype(str).str.strip().str.lower()
        working = working[working["text"].ne("") & working["label"].isin(["safe", "threat"])].copy()
        self.pipeline.fit(working["text"], working["label"])
        self.is_trained = True

    def predict(self, text: str) -> NLPDetectionResult:
        if not self.is_trained:
            raise ValueError("NLP scam detector is not trained yet.")

        probability = float(self.pipeline.predict_proba([str(text)])[0][self._positive_class_index()])
        detected_patterns, matched_keywords = self._detect_patterns(str(text))
        return NLPDetectionResult(
            scam_probability=probability,
            detected_patterns=detected_patterns,
            matched_keywords=matched_keywords,
        )

    def _detect_patterns(self, text: str) -> tuple[List[str], Dict[str, List[str]]]:
        lowered = text.lower()
        detected_patterns: List[str] = []
        matched_keywords: Dict[str, List[str]] = {}

        for group, patterns in PATTERN_GROUPS.items():
            matches: List[str] = []
            for pattern in patterns:
                match = re.search(pattern, lowered)
                if match:
                    matches.append(match.group(0))
            if matches:
                detected_patterns.append(group)
                matched_keywords[group] = sorted(set(matches))

        return detected_patterns, matched_keywords

    def _positive_class_index(self) -> int:
        return list(self.pipeline.classes_).index("threat")
