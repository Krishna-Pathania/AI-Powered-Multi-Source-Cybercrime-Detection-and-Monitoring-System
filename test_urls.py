from __future__ import annotations

import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "detector.pkl"

TEST_URLS = [
    "https://web.whatsapp.com/",
    "https://www.google.com",
    "https://drive.google.com/file/d/123456/view",
    "https://lpucolab438.examly.io/mycourses/details?id=2f27841c-39de-42cc-81c6-809542371b65&type=mylabs",
    "https://www.fancode.com/formula1",
    "https://open.spotify.com/",
    "https://ums.lpu.in/lpuums/",
    "http://198.51.100.10/login/verify-account",
    "http://free-bonus-wallet-secure.com/update",
    "https://kucoin-login.webflow.io/",
]


def main() -> None:
    with MODEL_FILE.open("rb") as model_file:
        system = pickle.load(model_file)

    print("Real-World URL Testing")
    print("=" * 80)
    for url in TEST_URLS:
        result = system.analyze("url", url)
        print(url)
        print(f"Prediction: {result.prediction}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Risk Score: {result.risk_score}/10")
        print(f"Probability: {result.probability:.2%}")
        print("Reasons:")
        for reason in result.reasons[:3]:
            print(f"  - {reason}")
        if result.debug:
            print(f"Decision Reason: {result.debug.get('decision_reason')}")
        print("-" * 80)


if __name__ == "__main__":
    main()
