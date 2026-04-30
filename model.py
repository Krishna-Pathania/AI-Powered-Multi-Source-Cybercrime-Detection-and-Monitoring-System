from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model_training import URLModelBundle, URLModelTrainer
from nlp_detector import NLPScamDetector
from nlp_engine import NLPThreatAnalyzer
from predictor import URLPredictor
from scoring_engine import RiskScoringEngine
from log_utils import append_log_entry, build_log_entry


LABEL_MAP = {
    "1": "threat",
    "0": "safe",
    "spam": "threat",
    "ham": "safe",
    "phishing": "threat",
    "legitimate": "safe",
    "benign": "safe",
    "malicious": "threat",
    "fraud": "threat",
}


def normalize_label(value: object) -> str:
    normalized = str(value).strip().lower()
    return LABEL_MAP.get(normalized, normalized)


def clean_text(value: object) -> str:
    text = str(value or "")
    return re.sub(r"\s+", " ", text).strip()


def extract_urls(text: object) -> List[str]:
    if not isinstance(text, str):
        text = str(text or "")
    return re.findall(r"(https?://[^\s<>\"]+|www\.[^\s<>\"]+)", text)


def build_risk_level(score: int) -> str:
    if score >= 8:
        return "High"
    if score >= 4:
        return "Medium"
    return "Low"


def normalize_batch_columns(frame: pd.DataFrame, default_source: str | None = None) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    source_aliases = ["source", "type", "category", "channel"]
    content_aliases = [
        "content",
        "text",
        "message",
        "body",
        "url",
        "link",
        "email",
        "sms",
        "input",
        "data",
    ]

    source_column = next((column for column in source_aliases if column in normalized.columns), None)
    content_column = next((column for column in content_aliases if column in normalized.columns), None)

    if content_column is None and len(normalized.columns) == 1:
        content_column = normalized.columns[0]

    if content_column is not None and content_column != "content":
        normalized = normalized.rename(columns={content_column: "content"})

    if source_column is not None and source_column != "source":
        normalized = normalized.rename(columns={source_column: "source"})

    if "source" not in normalized.columns and default_source:
        normalized["source"] = default_source

    return normalized


@dataclass
class DetectionResult:
    source: str
    prediction: str
    probability: float
    risk_score: int
    risk_level: str
    reasons: List[str]
    extracted_urls: List[str] = field(default_factory=list)
    debug: Dict[str, object] = field(default_factory=dict)


class CybercrimeDetectionSystem:
    def __init__(self) -> None:
        self.email_model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=25000,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
            ]
        )
        self.sms_model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=10000,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        )
        self.text_model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=18000,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
            ]
        )
        self.url_bundle: URLModelBundle | None = None
        self.url_predictor: URLPredictor | None = None
        self.nlp_analyzer = NLPThreatAnalyzer()
        self.nlp_detector = NLPScamDetector()
        self.risk_engine = RiskScoringEngine()
        self.is_trained = False
        self.metrics: Dict[str, Dict[str, object]] = {}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "text_model"):
            self.text_model = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            stop_words="english",
                            ngram_range=(1, 2),
                            min_df=2,
                            max_features=18000,
                        ),
                    ),
                    ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
                ]
            )
        if not hasattr(self, "nlp_analyzer"):
            self.nlp_analyzer = NLPThreatAnalyzer()
        if not hasattr(self, "nlp_detector"):
            self.nlp_detector = NLPScamDetector()
        self.risk_engine = RiskScoringEngine()

    def train(self, email_df: pd.DataFrame, sms_df: pd.DataFrame, url_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        url_trainer = URLModelTrainer(random_state=42)
        self.url_bundle, url_metrics = url_trainer.train(url_df)
        self.url_predictor = URLPredictor(self.url_bundle)
        text_df = pd.concat([email_df[["text", "label"]], sms_df[["text", "label"]]], ignore_index=True)
        self.nlp_detector.train(text_df)
        self.metrics = {
            "email": self._train_text_model(self.email_model, email_df, "email"),
            "sms": self._train_text_model(self.sms_model, sms_df, "sms"),
            "text": self._train_text_model(self.text_model, text_df, "text"),
            "url": url_metrics,
        }
        self.is_trained = True
        return self.metrics

    def _train_text_model(self, pipeline: Pipeline, df: pd.DataFrame, source_name: str) -> Dict[str, object]:
        working = df.copy()
        working["text"] = working["text"].map(clean_text)
        working["label"] = working["label"].map(normalize_label)
        working = working[(working["text"] != "") & working["label"].isin(["safe", "threat"])].copy()

        x_train, x_test, y_train, y_test = train_test_split(
            working["text"],
            working["label"],
            test_size=0.2,
            random_state=42,
            stratify=working["label"],
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, predictions, labels=["safe", "threat"])
        return {
            "source": source_name,
            "samples": int(len(working)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "class_distribution": working["label"].value_counts().to_dict(),
            "report": report,
            "confusion_matrix": {
                "labels": ["safe", "threat"],
                "matrix": matrix.tolist(),
            },
        }

    def analyze(self, source: str, content: str) -> DetectionResult:
        if not self.is_trained:
            raise ValueError("System is not trained yet. Run train.py first.")

        source = source.strip().lower()
        content = clean_text(content)
        if not content:
            raise ValueError("Input is empty. Please provide URL, email, SMS, or text content.")
        if source not in {"email", "sms", "text", "url"}:
            raise ValueError("Source must be one of: email, sms, text, url")
        if source == "email":
            result = self._analyze_email(content)
        elif source == "sms":
            result = self._analyze_sms(content)
        elif source == "text":
            result = self._analyze_text(content)
        else:
            result = self._analyze_url(content)

        append_log_entry(
            build_log_entry(
                input_value=content,
                risk=result.risk_level,
                score=result.risk_score,
                ml_prob=result.probability,
                reasons=result.reasons,
            )
        )
        return result

    def analyze_batch(self, frame: pd.DataFrame, default_source: str | None = None) -> pd.DataFrame:
        frame = normalize_batch_columns(frame, default_source=default_source)
        required = {"source", "content"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Uploaded file is missing columns: {sorted(missing)}")

        rows = []
        for index, record in enumerate(frame.to_dict(orient="records"), start=1):
            try:
                result = self.analyze(str(record["source"]), str(record["content"]))
                rows.append(
                    {
                        "row": index,
                        "source": result.source,
                        "content": record["content"],
                        "prediction": result.prediction,
                        "probability": round(result.probability, 4),
                        "risk_score": result.risk_score,
                        "risk_level": result.risk_level,
                        "reasons": "; ".join(result.reasons),
                        "urls_found": ", ".join(result.extracted_urls),
                        "rule_triggers": "; ".join(
                            trigger.get("reason", "") for trigger in result.debug.get("rule_triggers", [])
                        ),
                        "api_verdict": (result.debug.get("api_result") or {}).get("verdict", ""),
                        "api_confidence": round(float((result.debug.get("api_result") or {}).get("confidence_score", 0.0)), 4),
                        "nlp_patterns": ", ".join(result.debug.get("nlp_detector", {}).get("detected_patterns", [])),
                        "nlp_scam_probability": round(float(result.debug.get("nlp_detector", {}).get("scam_probability", 0.0)), 4),
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "row": index,
                        "source": str(record.get("source", "")),
                        "content": record.get("content", ""),
                        "prediction": "Error",
                        "probability": 0.0,
                        "risk_score": 0,
                        "risk_level": "Low",
                        "reasons": "",
                        "urls_found": "",
                        "rule_triggers": "",
                        "api_verdict": "",
                        "api_confidence": 0.0,
                        "nlp_patterns": "",
                        "nlp_scam_probability": 0.0,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        return pd.DataFrame(rows)

    def _analyze_email(self, text: str) -> DetectionResult:
        return self._analyze_textual_source("email", text, self.email_model)

    def _analyze_sms(self, text: str) -> DetectionResult:
        return self._analyze_textual_source("sms", text, self.sms_model)

    def _analyze_text(self, text: str) -> DetectionResult:
        return self._analyze_textual_source("text", text, self.text_model)

    def _analyze_url(self, url: str) -> DetectionResult:
        if self.url_predictor is None:
            raise ValueError("URL predictor is not trained yet.")
        payload = self.url_predictor.predict(url)
        return DetectionResult(
            source="url",
            prediction=str(payload["prediction"]),
            probability=float(payload["probability"]),
            risk_score=int(payload["risk_score"]),
            risk_level=str(payload["risk_level"]),
            reasons=list(payload["reasons"]),
            extracted_urls=[url],
            debug=dict(payload.get("debug", {})),
        )

    def _analyze_textual_source(self, source: str, text: str, pipeline: Pipeline) -> DetectionResult:
        probability = float(pipeline.predict_proba([text])[0][self._positive_class_index(pipeline)])
        urls = extract_urls(text)
        nlp_result = self.nlp_analyzer.analyze(text)
        nlp_ml_result = self.nlp_detector.predict(text)
        rule_score, rule_reasons = self._common_text_signals(text)
        embedded_rule_score = 0
        embedded_reasons: List[str] = []
        url_debug_rows = []

        for url in urls[:2]:
            url_result = self._analyze_url(url)
            url_debug_rows.append({"url": url, "prediction": url_result.prediction, "probability": url_result.probability})
            if url_result.prediction == "Threat":
                embedded_rule_score += max(1, url_result.risk_score // 3)
                embedded_reasons.append(f"Embedded URL looks suspicious: {url}")

        combined_rule_score = min(4, rule_score + embedded_rule_score)
        reputation_adjustment = 0
        scoring = self.risk_engine.score(
            ml_probability=max(probability, nlp_ml_result.scam_probability),
            rule_score=combined_rule_score,
            api_score=reputation_adjustment,
            nlp_score=int(nlp_result["score"]),
        )

        pattern_reasons = [
            f"NLP scam patterns detected: {', '.join(nlp_ml_result.matched_keywords.get(pattern, [pattern]))}."
            for pattern in nlp_ml_result.detected_patterns[:3]
        ]
        reasons = (pattern_reasons + nlp_result["reasons"] + rule_reasons + embedded_reasons)[:6]
        if not reasons:
            reasons = ["No strong risk indicators were found."]

        return DetectionResult(
            source=source,
            prediction=scoring["decision"],
            probability=probability,
            risk_score=scoring["risk_score"],
            risk_level=scoring["risk_level"],
            reasons=reasons,
            extracted_urls=urls,
            debug={
                "channel": source,
                "detected_urls": urls,
                "url_checks": url_debug_rows,
                "nlp": nlp_result,
                "nlp_detector": {
                    "scam_probability": nlp_ml_result.scam_probability,
                    "detected_patterns": nlp_ml_result.detected_patterns,
                    "matched_keywords": nlp_ml_result.matched_keywords,
                },
                "rule_score": combined_rule_score,
                "scoring_components": scoring["components"],
            },
        )

    def _common_text_signals(self, text: str) -> tuple[int, List[str]]:
        lowered = text.lower()
        score = 0
        reasons: List[str] = []

        keyword_groups = {
            "credential harvesting language": r"\b(login|password|verify|account|confirm)\b",
            "financial lure language": r"\b(bank|payment|invoice|wallet|refund|transfer)\b",
            "reward or urgency bait": r"\b(prize|reward|won|free|claim|urgent|immediately)\b",
            "suspicious action language": r"\b(click|tap|open|download|reset|update)\b",
        }
        for reason, pattern in keyword_groups.items():
            if re.search(pattern, lowered):
                reasons.append(reason.capitalize() + ".")
                score += 2

        if re.search(r"[A-Z]{6,}", text):
            reasons.append("Contains excessive uppercase emphasis.")
            score += 1
        if extract_urls(text):
            score += 1
        return score, reasons

    @staticmethod
    def _positive_class_index(pipeline: Pipeline) -> int:
        return list(pipeline.classes_).index("threat")
