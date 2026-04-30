from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from feature_extraction import URLFeatureExtractor, prepare_model_features

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


@dataclass
class URLModelBundle:
    vectorizer: DictVectorizer
    classifier: object
    feature_extractor: URLFeatureExtractor
    model_name: str
    class_labels: List[str]


class URLModelTrainer:
    def __init__(self, random_state: int = 42, model_output_path: str | Path | None = None) -> None:
        self.random_state = random_state
        self.feature_extractor = URLFeatureExtractor()
        self.model_output_path = Path(model_output_path) if model_output_path else None

    def train(self, url_df: pd.DataFrame) -> tuple[URLModelBundle, Dict[str, object]]:
        working = url_df.copy()
        working["url"] = working["url"].astype(str).str.strip()
        working["label"] = working["label"].astype(str).str.strip().str.lower()
        working = working[working["url"].ne("") & working["label"].isin(["safe", "threat"])].copy()

        feature_dicts = [prepare_model_features(self.feature_extractor.extract(url)) for url in working["url"]]
        x_train_dicts, x_test_dicts, y_train, y_test = train_test_split(
            feature_dicts,
            working["label"],
            test_size=0.2,
            random_state=self.random_state,
            stratify=working["label"],
        )

        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform(x_train_dicts)
        x_test = vectorizer.transform(x_test_dicts)

        candidates = self._candidate_models()
        comparison_rows: List[Dict[str, object]] = []
        best_model_name = ""
        best_classifier = None
        best_f1 = -1.0

        for model_name, classifier in candidates:
            local_y_train = y_train
            local_y_test = y_test
            if model_name == "XGBoost":
                local_y_train = y_train.map({"safe": 0, "threat": 1})
                local_y_test = y_test.map({"safe": 0, "threat": 1})
            classifier.fit(x_train, local_y_train)
            predictions = classifier.predict(x_test)
            if model_name == "XGBoost":
                predictions = pd.Series(predictions).map({0: "safe", 1: "threat"}).tolist()
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test,
                predictions,
                average="binary",
                pos_label="threat",
                zero_division=0,
            )
            comparison_rows.append(
                {
                    "model": model_name,
                    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(f1), 4),
                }
            )
            if float(f1) > best_f1:
                best_f1 = float(f1)
                best_model_name = model_name
                best_classifier = classifier

        assert best_classifier is not None
        classifier = best_classifier
        class_labels = ["safe", "threat"]
        predictions = classifier.predict(x_test)
        if best_model_name == "XGBoost":
            predictions = pd.Series(predictions).map({0: "safe", 1: "threat"}).tolist()
        probabilities = classifier.predict_proba(x_test)

        cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        y_for_cv = working["label"]
        if best_model_name == "XGBoost":
            y_for_cv = working["label"].map({"safe": 0, "threat": 1})
        cv_scores = cross_val_score(classifier, vectorizer.transform(feature_dicts), y_for_cv, cv=cross_validator, scoring="f1_weighted")
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, predictions, labels=["safe", "threat"])

        feature_names = vectorizer.get_feature_names_out()
        importance_pairs = sorted(
            zip(feature_names, self._feature_importances(classifier, feature_names)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        top_importances = [
            {"feature": feature_name, "importance": round(float(importance), 6)}
            for feature_name, importance in importance_pairs[:15]
        ]

        bundle = URLModelBundle(
            vectorizer=vectorizer,
            classifier=classifier,
            feature_extractor=self.feature_extractor,
            model_name=best_model_name,
            class_labels=class_labels,
        )
        metrics = {
            "source": "url",
            "samples": int(len(working)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "class_distribution": working["label"].value_counts().to_dict(),
            "report": report,
            "confusion_matrix": {
                "labels": ["safe", "threat"],
                "matrix": matrix.tolist(),
            },
            "cross_validation": {
                "scoring": "f1_weighted",
                "fold_scores": [round(float(score), 6) for score in cv_scores],
                "mean": round(float(cv_scores.mean()), 6),
                "std": round(float(cv_scores.std()), 6),
            },
            "feature_importance": top_importances,
            "test_probabilities_preview": [round(float(row.max()), 4) for row in probabilities[:10]],
            "selected_model": best_model_name,
            "model_comparison": comparison_rows,
        }
        if self.model_output_path is not None:
            with self.model_output_path.open("wb") as model_file:
                pickle.dump(bundle, model_file)
            metrics["saved_model_path"] = str(self.model_output_path)
        return bundle, metrics

    def _candidate_models(self) -> List[tuple[str, object]]:
        candidates: List[tuple[str, object]] = [
            (
                "Logistic Regression",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=self.random_state,
                    solver="liblinear",
                ),
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=18,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=self.random_state,
                    n_jobs=1,
                ),
            ),
        ]
        if XGBClassifier is not None:
            candidates.append(
                (
                    "XGBoost",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=self.random_state,
                    ),
                )
            )
        return candidates

    @staticmethod
    def _feature_importances(classifier: object, feature_names: List[str]) -> List[float]:
        if hasattr(classifier, "feature_importances_"):
            return [float(value) for value in classifier.feature_importances_]
        if hasattr(classifier, "coef_"):
            coefficients = classifier.coef_[0]
            return [abs(float(value)) for value in coefficients]
        return [0.0 for _ in feature_names]
