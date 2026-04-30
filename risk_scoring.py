from __future__ import annotations

from typing import Dict


class RiskScoringEngine:
    def score(
        self,
        source: str,
        ml_probability: float,
        rule_score: int = 0,
        reputation_adjustment: int = 0,
        nlp_score: int = 0,
    ) -> Dict[str, object]:
        base_weight = {"url": 6.0, "email": 5.5, "sms": 5.0, "text": 4.5}.get(source, 5.0)
        ml_score = round(ml_probability * base_weight, 2)
        final_score = min(10, max(0, round(ml_score + rule_score + reputation_adjustment + nlp_score)))

        if final_score >= 8:
            risk_level = "High"
            decision = "Threat"
        elif final_score >= 4:
            risk_level = "Medium"
            decision = "Threat" if ml_probability >= 0.5 or rule_score + nlp_score + reputation_adjustment >= 4 else "Safe"
        else:
            risk_level = "Low"
            decision = "Safe"

        return {
            "risk_score": int(final_score),
            "risk_level": risk_level,
            "decision": decision,
            "ml_score": ml_score,
        }
