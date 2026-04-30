from __future__ import annotations

from typing import Dict


class RiskScoringEngine:
    def score(
        self,
        ml_probability: float,
        rule_score: int = 0,
        api_score: float = 0.0,
        nlp_score: int = 0,
    ) -> Dict[str, object]:
        ml_component = float(ml_probability) * 4.0
        rule_component = float(rule_score) * 2.0
        api_component = float(api_score) * 3.0
        nlp_component = float(nlp_score) * 1.0

        raw_score = ml_component + rule_component + api_component + nlp_component
        risk_score = max(0, min(10, round(raw_score)))

        if risk_score >= 8:
            risk_level = "High"
            decision = "Threat"
        elif risk_score >= 4:
            risk_level = "Medium"
            decision = "Threat"
        else:
            risk_level = "Low"
            decision = "Safe"

        return {
            "risk_score": int(risk_score),
            "risk_level": risk_level,
            "decision": decision,
            "components": {
                "ml_component": round(ml_component, 4),
                "rule_component": round(rule_component, 4),
                "api_component": round(api_component, 4),
                "nlp_component": round(nlp_component, 4),
                "raw_score": round(raw_score, 4),
            },
        }
