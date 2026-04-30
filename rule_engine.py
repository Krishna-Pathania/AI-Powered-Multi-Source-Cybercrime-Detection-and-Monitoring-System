from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


SUSPICIOUS_TLDS = {
    "xyz",
    "tk",
    "top",
    "gq",
    "cf",
    "ml",
    "sbs",
    "cfd",
    "click",
    "buzz",
    "monster",
    "zip",
}


@dataclass
class RuleMatch:
    rule: str
    score: int
    reason: str


class RuleEngine:
    def evaluate(self, features: Dict[str, object]) -> Dict[str, object]:
        matches: List[RuleMatch] = []
        triggered_rules: List[Dict[str, object]] = []
        trusted_domain = int(features.get("trusted_domain", 0)) == 1
        suspicious_keyword_count = int(features.get("suspicious_keyword_count", 0))
        url_length = int(features.get("url_length", 0))
        query_length = int(features.get("query_length", 0))
        special_char_count = int(features.get("special_char_count", 0))
        uses_https = int(features.get("uses_https", 0)) == 1
        benign_query_profile = uses_https and suspicious_keyword_count == 0 and query_length >= 24

        if int(features.get("subdomain_count", 0)) >= 4 and not trusted_domain:
            matches.append(RuleMatch("high_subdomain_count", 2, "Too many subdomains detected."))
        if str(features.get("tld", "")).lower() in SUSPICIOUS_TLDS:
            matches.append(RuleMatch("suspicious_tld", 2, f"Suspicious TLD detected: .{features.get('tld', 'unknown')}"))
        if suspicious_keyword_count > 0:
            matches.append(RuleMatch("phishing_keyword", 2, "Phishing-related keywords found in the URL."))
        if url_length >= 150 or (url_length >= 110 and suspicious_keyword_count > 0 and not benign_query_profile):
            matches.append(RuleMatch("long_url", 1, "URL is unusually long."))
        if int(features.get("has_ip_address", 0)) == 1:
            matches.append(RuleMatch("ip_address_host", 3, "URL uses an IP address instead of a domain."))
        if int(features.get("uses_https", 0)) == 0:
            matches.append(RuleMatch("no_https", 2, "URL does not use HTTPS."))
        if special_char_count >= 14 and not benign_query_profile:
            matches.append(RuleMatch("heavy_special_chars", 1, "URL contains many special characters."))
        if int(features.get("domain_age_days_mock", 9999)) <= 60:
            matches.append(RuleMatch("young_domain", 1, "Domain appears recently created."))

        for match in matches:
            triggered_rules.append(
                {
                    "rule_name": match.rule,
                    "score": match.score,
                    "reason": match.reason,
                }
            )

        total_rule_score = min(sum(match.score for match in matches), 6)
        return {
            "rule_score": total_rule_score,
            "triggered_rules": triggered_rules,
            "score": total_rule_score,
            "matches": triggered_rules,
            "reasons": [match.reason for match in matches],
        }
