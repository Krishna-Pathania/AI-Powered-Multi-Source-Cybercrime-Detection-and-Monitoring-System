from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from .config import AgentConfig
except ImportError:
    from config import AgentConfig


SCAM_AUGMENTATION_SAMPLES = [
    ("Act now. Your account is suspended. Verify immediately to avoid blocking.", "threat"),
    ("Urgent action required. Your account has been suspended. Verify now.", "threat"),
    ("Immediate response needed. Your account is blocked until you confirm your login.", "threat"),
    ("Your account will be disabled today unless you verify immediately.", "threat"),
    ("Security warning. Confirm your account now to prevent suspension.", "threat"),
    ("Limited time warning. Update your banking details now.", "threat"),
    ("You won a free gift. Claim your reward immediately.", "threat"),
    ("Security alert: your payroll account has been blocked. Login now.", "threat"),
    ("Your account is under review. Confirm your OTP to restore access.", "threat"),
    ("Congratulations, you have been selected for a cash bonus. Click now.", "threat"),
    ("Final warning: your email account will be disabled unless you verify today.", "threat"),
    ("This is a reminder for tomorrow's team meeting at 11 am.", "safe"),
    ("Can you review the attached project notes when you are free?", "safe"),
    ("Lunch is moved to 1 pm. Please join in the cafeteria.", "safe"),
    ("The course portal is available for regular assignment submission.", "safe"),
    ("Your package is arriving today. No action is needed.", "safe"),
]
MODEL_VERSION = 3


def clean_text(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


@dataclass(slots=True)
class TextPrediction:
    label: str
    scam_probability: float


class NLPModel:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=20000,
                        preprocessor=clean_text,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
            ]
        )
        self.is_trained = False

    def load_or_train(self) -> None:
        if self.config.nlp_model_path.exists():
            try:
                with self.config.nlp_model_path.open("rb") as model_file:
                    saved_payload = pickle.load(model_file)
                if isinstance(saved_payload, dict) and saved_payload.get("version") == MODEL_VERSION:
                    self.pipeline = saved_payload["pipeline"]
                    self.is_trained = True
                    return
            except (pickle.PickleError, AttributeError, ModuleNotFoundError, EOFError):
                pass

        dataset = self._load_training_dataset()
        self.pipeline.fit(dataset["text"], dataset["label"])
        self.config.nlp_model_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.nlp_model_path.open("wb") as model_file:
            pickle.dump({"version": MODEL_VERSION, "pipeline": self.pipeline}, model_file)
        self.is_trained = True

    def analyze_text(self, text: str) -> TextPrediction:
        if not self.is_trained:
            self.load_or_train()

        probability = float(self.pipeline.predict_proba([text])[0][self._positive_class_index()])
        label = "scam" if probability >= 0.5 else "safe"
        return TextPrediction(label=label, scam_probability=probability)

    def _load_training_dataset(self) -> pd.DataFrame:
        for candidate in self.config.sms_dataset_candidates:
            if not candidate.exists():
                continue
            if candidate.name == "SMSSpamCollection":
                dataset = pd.read_csv(candidate, sep="\t", header=None, names=["label", "text"])
                dataset["label"] = dataset["label"].map({"spam": "threat", "ham": "safe"})
                return self._augment_dataset(dataset[["text", "label"]])
            if candidate.suffix.lower() == ".csv":
                dataset = pd.read_csv(candidate)
                if {"text", "label"}.issubset(dataset.columns):
                    normalized = dataset[["text", "label"]].copy()
                    normalized["label"] = normalized["label"].astype(str).str.lower().replace(
                        {"spam": "threat", "ham": "safe"}
                    )
                    return self._augment_dataset(normalized)
        raise FileNotFoundError("No SMS training dataset found for the protection agent NLP model.")

    def _positive_class_index(self) -> int:
        return list(self.pipeline.classes_).index("threat")

    @staticmethod
    def _augment_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
        augmentation = pd.DataFrame(SCAM_AUGMENTATION_SAMPLES, columns=["text", "label"])
        combined = pd.concat([dataset, augmentation], ignore_index=True)
        combined["text"] = combined["text"].astype(str).str.strip()
        combined["label"] = combined["label"].astype(str).str.strip().str.lower()
        return combined[combined["text"].ne("") & combined["label"].isin(["safe", "threat"])].copy()
