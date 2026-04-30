from __future__ import annotations

import base64
import os
from typing import Dict, List

import requests

from feature_extraction import EXTENDED_TRUSTED_DOMAINS, load_trusted_domains


DEFAULT_WHITELIST = list(EXTENDED_TRUSTED_DOMAINS)
DEFAULT_BLACKLIST = [
    "free-bonus-wallet-secure.com",
    "otp-verify-now.example.org",
    "banking-check-alert.example.com",
    "paypal-login-security-check.example.com",
]


class URLReputationChecker:
    def __init__(self, whitelist: List[str] | None = None, blacklist: List[str] | None = None) -> None:
        merged_whitelist = set(DEFAULT_WHITELIST)
        if whitelist:
            merged_whitelist.update(whitelist)
        self.whitelist = {domain.lower() for domain in merged_whitelist}
        self.blacklist = {domain.lower() for domain in (blacklist or DEFAULT_BLACKLIST)}
        self.google_safe_browsing_key = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "").strip()
        self.virustotal_key = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
        self.request_timeout = 5

    def check(self, url: str, registered_domain: str) -> Dict[str, object]:
        registered_domain = registered_domain.lower()
        live_whitelist = set(self.whitelist)
        live_whitelist.update(domain.lower() for domain in load_trusted_domains())
        if registered_domain in live_whitelist:
            return {
                "status": "safe_override",
                "score_adjustment": -10,
                "reasons": [f"Whitelisted trusted domain override applied for {registered_domain}."],
                "sources": ["whitelist"],
                "api_result": None,
            }
        if registered_domain in self.blacklist:
            return {
                "status": "malicious_override",
                "score_adjustment": 6,
                "reasons": [f"Domain matched internal blacklist: {registered_domain}."],
                "sources": ["blacklist"],
                "api_result": None,
            }

        api_result = self.check_url_api(url)
        if api_result["verdict"] == "malicious":
            return {
                "status": "api_malicious_override",
                "score_adjustment": 6,
                "reasons": [f"API intelligence flagged the URL as malicious with confidence {api_result['confidence_score']:.2f}."],
                "sources": api_result["sources"],
                "api_result": api_result,
            }

        score_adjustment = 0
        reasons: List[str] = []
        if api_result["verdict"] == "safe" and api_result["confidence_score"] >= 0.8:
            score_adjustment = -1
            reasons.append("API reputation checks found no malicious signal.")

        return {
            "status": "neutral",
            "score_adjustment": score_adjustment,
            "reasons": reasons,
            "sources": api_result["sources"],
            "api_result": api_result,
        }

    def check_url_api(self, url: str) -> Dict[str, object]:
        google_result = self._check_google_safe_browsing(url)
        virustotal_result = self._check_virustotal(url)

        engine_results = [google_result, virustotal_result]
        malicious_hits = [result for result in engine_results if result["malicious"]]
        failures = [result for result in engine_results if result["status"] in {"timeout", "rate_limited", "error"}]
        unavailable = [result for result in engine_results if result["status"] == "mock"]
        sources = [result["engine"] for result in engine_results if result["status"] == "ok"]

        if malicious_hits:
            confidence = max(result["confidence_score"] for result in malicious_hits)
            return {
                "verdict": "malicious",
                "confidence_score": confidence,
                "sources": [result["engine"] for result in malicious_hits],
                "details": engine_results,
                "fallback_used": False,
            }

        if failures and len(failures) == len(engine_results):
            return {
                "verdict": "safe",
                "confidence_score": 0.0,
                "sources": [],
                "details": engine_results,
                "fallback_used": True,
            }

        if unavailable and len(unavailable) == len(engine_results):
            return {
                "verdict": "safe",
                "confidence_score": 0.1,
                "sources": [],
                "details": engine_results,
                "fallback_used": True,
            }

        safe_confidence = 0.85 if any(result["status"] == "ok" for result in engine_results) else 0.2
        return {
            "verdict": "safe",
            "confidence_score": safe_confidence,
            "sources": sources,
            "details": engine_results,
            "fallback_used": bool(failures),
        }

    def _check_google_safe_browsing(self, url: str) -> Dict[str, object]:
        if not self.google_safe_browsing_key:
            return {
                "engine": "google_safe_browsing",
                "status": "mock",
                "malicious": False,
                "confidence_score": 0.0,
                "reason": "No Google Safe Browsing API key configured.",
            }
        try:
            endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={self.google_safe_browsing_key}"
            payload = {
                "client": {"clientId": "cybercrime-detector", "clientVersion": "1.0"},
                "threatInfo": {
                    "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
                    "platformTypes": ["ANY_PLATFORM"],
                    "threatEntryTypes": ["URL"],
                    "threatEntries": [{"url": url}],
                },
            }
            response = requests.post(endpoint, json=payload, timeout=self.request_timeout)
            if response.status_code == 429:
                return self._rate_limited_result("google_safe_browsing")
            response.raise_for_status()
            data = response.json()
            malicious = bool(data.get("matches"))
            return {
                "engine": "google_safe_browsing",
                "status": "ok",
                "malicious": malicious,
                "confidence_score": 0.98 if malicious else 0.75,
                "reason": "Threat match found." if malicious else "No threat match found.",
            }
        except requests.Timeout:
            return self._timeout_result("google_safe_browsing")
        except requests.RequestException as exc:
            return self._error_result("google_safe_browsing", str(exc))

    def _check_virustotal(self, url: str) -> Dict[str, object]:
        if not self.virustotal_key:
            return {
                "engine": "virustotal",
                "status": "mock",
                "malicious": False,
                "confidence_score": 0.0,
                "reason": "No VirusTotal API key configured.",
            }
        try:
            url_id = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").strip("=")
            response = requests.get(
                f"https://www.virustotal.com/api/v3/urls/{url_id}",
                headers={"x-apikey": self.virustotal_key},
                timeout=self.request_timeout,
            )
            if response.status_code == 404:
                return {
                    "engine": "virustotal",
                    "status": "ok",
                    "malicious": False,
                    "confidence_score": 0.6,
                    "reason": "URL not found in VirusTotal cache.",
                }
            if response.status_code == 429:
                return self._rate_limited_result("virustotal")
            response.raise_for_status()
            data = response.json()
            stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            malicious_count = int(stats.get("malicious", 0)) + int(stats.get("suspicious", 0))
            harmless_count = int(stats.get("harmless", 0))
            total = malicious_count + harmless_count + int(stats.get("undetected", 0))
            confidence = min(0.99, malicious_count / total) if total else 0.5
            return {
                "engine": "virustotal",
                "status": "ok",
                "malicious": malicious_count > 0,
                "confidence_score": confidence if malicious_count > 0 else 0.7,
                "reason": f"VirusTotal malicious detections: {malicious_count}.",
            }
        except requests.Timeout:
            return self._timeout_result("virustotal")
        except requests.RequestException as exc:
            return self._error_result("virustotal", str(exc))

    @staticmethod
    def _timeout_result(engine: str) -> Dict[str, object]:
        return {
            "engine": engine,
            "status": "timeout",
            "malicious": False,
            "confidence_score": 0.0,
            "reason": "Request timed out. Falling back to ML.",
        }

    @staticmethod
    def _rate_limited_result(engine: str) -> Dict[str, object]:
        return {
            "engine": engine,
            "status": "rate_limited",
            "malicious": False,
            "confidence_score": 0.0,
            "reason": "Rate limited by external API. Falling back to ML.",
        }

    @staticmethod
    def _error_result(engine: str, message: str) -> Dict[str, object]:
        return {
            "engine": engine,
            "status": "error",
            "malicious": False,
            "confidence_score": 0.0,
            "reason": f"API error: {message}",
        }
