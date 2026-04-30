from __future__ import annotations

import pickle
import random
import json
from pathlib import Path

import pandas as pd

from feature_extraction import load_trusted_domains
from model import CybercrimeDetectionSystem, clean_text, extract_urls, normalize_label


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "detector.pkl"
URL_MODEL_FILE = BASE_DIR / "url_model_bundle.pkl"
METRICS_FILE = BASE_DIR / "training_metrics.json"
SMS_DATA_FILE = BASE_DIR / "sms_dataset.csv"
SMS_UCI_FILE = BASE_DIR / "SMSSpamCollection"
UNIVERSITY_SCAMS_DIR = BASE_DIR / "universityscams-revert-4e2accff"
PHISHTANK_VERIFIED_FILE = BASE_DIR / "verified_online.csv.gz"
SAFE_URLS_FILE = BASE_DIR / "safe_urls.csv"


def load_text_dataset(path: Path, text_columns: list[str], label_column: str = "label", sample_size: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = [column for column in text_columns + [label_column] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    text_parts = [df[column].fillna("").astype(str).str.strip() for column in text_columns]
    text_series = text_parts[0]
    for part in text_parts[1:]:
        text_series = text_series + " " + part

    return pd.DataFrame({"text": text_series.map(clean_text), "label": df[label_column].map(normalize_label)})


def build_email_dataset() -> pd.DataFrame:
    frames = [
        load_text_dataset(BASE_DIR / "phishing_email.csv", ["text_combined"], sample_size=18000),
        load_text_dataset(BASE_DIR / "SpamAssasin.csv", ["subject", "body"], sample_size=5000),
        load_text_dataset(BASE_DIR / "Enron.csv", ["subject", "body"], sample_size=9000),
        load_text_dataset(BASE_DIR / "CEAS_08.csv", ["subject", "body"], sample_size=7000),
        load_text_dataset(BASE_DIR / "Ling.csv", ["subject", "body"], sample_size=2000),
    ]
    university_scams_df = build_university_scams_dataset()
    if not university_scams_df.empty:
        frames.append(university_scams_df)

    email_df = pd.concat(frames, ignore_index=True)
    email_df = email_df[email_df["label"].isin(["safe", "threat"]) & email_df["text"].ne("")]
    return email_df.reset_index(drop=True)


def build_university_scams_dataset() -> pd.DataFrame:
    bodies_dir = UNIVERSITY_SCAMS_DIR / "with_bodies"
    if not bodies_dir.exists():
        return pd.DataFrame(columns=["text", "label"])

    scam_rows: list[dict[str, str]] = []
    for txt_file in bodies_dir.rglob("*.txt"):
        try:
            raw_text = txt_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        lines = raw_text.splitlines()
        subject = ""
        body_lines: list[str] = []
        in_body = False

        for line in lines:
            if line.startswith("Subject:"):
                subject = line.split(":", 1)[1].strip()
            elif line.startswith("Body:"):
                in_body = True
                body_start = line.split(":", 1)[1].strip()
                if body_start:
                    body_lines.append(body_start)
            elif in_body:
                body_lines.append(line.strip())

        combined = clean_text(" ".join(part for part in [subject, " ".join(body_lines)] if part))
        if combined:
            scam_rows.append({"text": combined, "label": "threat"})

    return pd.DataFrame(scam_rows)


def build_sms_dataset() -> pd.DataFrame:
    if SMS_UCI_FILE.exists():
        sms_df = pd.read_csv(SMS_UCI_FILE, sep="\t", header=None, names=["label", "text"])
    elif SMS_DATA_FILE.exists():
        sms_df = pd.read_csv(SMS_DATA_FILE)
    else:
        raise FileNotFoundError(
            "No SMS dataset found. Expected either 'SMSSpamCollection' or 'sms_dataset.csv' in the project folder."
        )

    sms_df["text"] = sms_df["text"].map(clean_text)
    sms_df["label"] = sms_df["label"].map(normalize_label)
    sms_df = sms_df[sms_df["label"].isin(["safe", "threat"]) & sms_df["text"].ne("")]
    return sms_df.reset_index(drop=True)


def build_url_dataset() -> pd.DataFrame:
    url_rows: list[dict[str, str]] = []
    sources = [
        ("SpamAssasin.csv", ["subject", "body"], 6000),
        ("CEAS_08.csv", ["subject", "body"], 7000),
        ("Nigerian_Fraud.csv", ["subject", "body"], 2500),
    ]

    for filename, text_columns, sample_size in sources:
        df = pd.read_csv(BASE_DIR / filename)
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        merged_text = df[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
        labels = df["label"].map(normalize_label)
        for text, label in zip(merged_text, labels):
            urls = extract_urls(text)
            for url in urls:
                url_rows.append({"url": url, "label": label})

    if PHISHTANK_VERIFIED_FILE.exists():
        phishtank_df = pd.read_csv(PHISHTANK_VERIFIED_FILE, compression="gzip")
        if {"url", "verified", "online"}.issubset(phishtank_df.columns):
            phishtank_df["verified"] = phishtank_df["verified"].astype(str).str.lower().str.strip()
            phishtank_df["online"] = phishtank_df["online"].astype(str).str.lower().str.strip()
            phishtank_df = phishtank_df[
                phishtank_df["url"].notna()
                & phishtank_df["verified"].eq("yes")
                & phishtank_df["online"].eq("yes")
            ]
            for url in phishtank_df["url"].astype(str):
                cleaned_url = clean_text(url)
                if cleaned_url:
                    url_rows.append({"url": cleaned_url, "label": "threat"})

    if SAFE_URLS_FILE.exists():
        safe_urls_df = pd.read_csv(SAFE_URLS_FILE)
        if {"url", "label"}.issubset(safe_urls_df.columns):
            safe_urls_df["label"] = safe_urls_df["label"].map(normalize_label)
            safe_urls_df["url"] = safe_urls_df["url"].map(clean_text)
            safe_urls_df = safe_urls_df[
                safe_urls_df["url"].ne("") & safe_urls_df["label"].isin(["safe", "threat"])
            ]
            for record in safe_urls_df.to_dict(orient="records"):
                url_rows.append({"url": record["url"], "label": record["label"]})

    safe_seed_urls = [
        "https://www.google.com",
        "https://www.wikipedia.org",
        "https://www.microsoft.com",
        "https://www.openai.com",
        "https://mail.yahoo.com",
        "https://github.com/openai",
        "https://www.amazon.in",
        "https://www.apple.com/in",
        "https://www.ibm.com",
        "https://support.microsoft.com",
        "https://www.nasa.gov",
        "https://www.coursera.org",
        "https://www.linkedin.com",
        "https://www.stackoverflow.com",
        "https://news.ycombinator.com",
        "https://open.spotify.com/",
        "https://www.fancode.com/formula1",
        "https://ums.lpu.in/lpuums/",
        "https://lpucolab438.examly.io/mycourses/details?id=2f27841c-39de-42cc-81c6-809542371b65&type=mylabs",
    ]
    threat_seed_urls = [
        "http://198.51.100.10/login/verify-account",
        "http://free-bonus-wallet-secure.com/update",
        "http://banking-check-alert.example.com/signin",
        "http://45.67.23.19/paypal/confirm",
        "http://secure-gift-reward-login.net/claim",
        "http://otp-verify-now.example.org/account",
        "http://paypal-login-security-check.example.com",
        "http://crypto-bonus-airdrop.example.net/free",
    ]
    for url in safe_seed_urls:
        url_rows.append({"url": url, "label": "safe"})
    for url in threat_seed_urls:
        url_rows.append({"url": url, "label": "threat"})
    for domain in load_trusted_domains():
        url_rows.append({"url": f"https://{domain}", "label": "safe"})
        url_rows.append({"url": f"https://www.{domain}", "label": "safe"})

    url_df = pd.DataFrame(url_rows).drop_duplicates()
    url_df = url_df[url_df["url"].ne("") & url_df["label"].isin(["safe", "threat"])]
    safe_df = url_df[url_df["label"] == "safe"]
    threat_df = url_df[url_df["label"] == "threat"]
    if not safe_df.empty and len(threat_df) > len(safe_df) * 2:
        threat_df = threat_df.sample(len(safe_df) * 2, random_state=42)
        url_df = pd.concat([safe_df, threat_df], ignore_index=True)
    return url_df.reset_index(drop=True)


def main() -> None:
    random.seed(42)

    email_df = build_email_dataset()
    sms_df = build_sms_dataset()
    url_df = build_url_dataset()

    print("Training data summary")
    print("Email samples:", len(email_df), email_df["label"].value_counts().to_dict())
    print("SMS samples:", len(sms_df), sms_df["label"].value_counts().to_dict())
    print("URL samples:", len(url_df), url_df["label"].value_counts().to_dict())

    system = CybercrimeDetectionSystem()
    metrics = system.train(email_df=email_df, sms_df=sms_df, url_df=url_df)

    if system.url_bundle is not None:
        with URL_MODEL_FILE.open("wb") as url_model_file:
            pickle.dump(system.url_bundle, url_model_file)
        metrics["url"]["saved_model_path"] = str(URL_MODEL_FILE)

    with MODEL_FILE.open("wb") as model_file:
        pickle.dump(system, model_file)

    with METRICS_FILE.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print("\nTraining complete. Metrics summary:")
    for source, report in metrics.items():
        print(f"- {source}: accuracy={report['accuracy']:.4f}, samples={report['samples']}")
        threat_report = report["report"].get("threat", {})
        safe_report = report["report"].get("safe", {})
        weighted_report = report["report"].get("weighted avg", {})
        matrix = report.get("confusion_matrix", {}).get("matrix", [[0, 0], [0, 0]])
        print(
            "  Safe  -> precision={:.4f}, recall={:.4f}, f1={:.4f}".format(
                safe_report.get("precision", 0.0),
                safe_report.get("recall", 0.0),
                safe_report.get("f1-score", 0.0),
            )
        )
        print(
            "  Threat-> precision={:.4f}, recall={:.4f}, f1={:.4f}".format(
                threat_report.get("precision", 0.0),
                threat_report.get("recall", 0.0),
                threat_report.get("f1-score", 0.0),
            )
        )
        print(
            "  Weighted Avg -> precision={:.4f}, recall={:.4f}, f1={:.4f}".format(
                weighted_report.get("precision", 0.0),
                weighted_report.get("recall", 0.0),
                weighted_report.get("f1-score", 0.0),
            )
        )
        print("  Confusion Matrix [rows=true, cols=predicted] labels=[safe, threat]")
        print(f"    {matrix[0]}")
        print(f"    {matrix[1]}")
        if source == "url":
            if report.get("selected_model"):
                print(f"  Selected URL Model: {report['selected_model']}")
            cv_report = report.get("cross_validation", {})
            if cv_report:
                print(
                    "  Cross Validation ({}) -> mean={:.4f}, std={:.4f}, folds={}".format(
                        cv_report.get("scoring", "n/a"),
                        cv_report.get("mean", 0.0),
                        cv_report.get("std", 0.0),
                        cv_report.get("fold_scores", []),
                    )
                )
            feature_importance = report.get("feature_importance", [])
            if feature_importance:
                print("  Top Feature Importances:")
                for item in feature_importance[:10]:
                    print(f"    {item['feature']}: {item['importance']:.6f}")
            comparison = report.get("model_comparison", [])
            if comparison:
                print("  Model Comparison:")
                for row in comparison:
                    print(
                        "    {model}: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}".format(
                            **row
                        )
                    )

    print(f"\nDetailed metrics saved to: {METRICS_FILE.name}")


if __name__ == "__main__":
    main()
