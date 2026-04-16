from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

from coach     import CoachingReport, Suggestion, get_recommendations
from explainer import Explanation, explain_prediction
from risk_engine import RiskOutput, _validate_input, get_risk_score


@dataclass
class PipelineResult:
    risk_score:      float
    risk_level:      str
    reasons:         list[str]
    recommendations: list[dict]
    positive_notes:  list[str]
    model_score:     float
    fusion_delta:    float
    fusion_flags:    list[str]
    shap_values:     dict
    explainer_mode:  str
    latency_ms:      float
    input_features:  dict

    def to_dict(self) -> dict:
        return {
            "risk_score":      round(self.risk_score, 2),
            "risk_level":      self.risk_level,
            "reasons":         self.reasons,
            "recommendations": self.recommendations,
            "positive_notes":  self.positive_notes,
            "model_score":     round(self.model_score, 2),
            "fusion_delta":    round(self.fusion_delta, 2),
            "fusion_flags":    self.fusion_flags,
            "shap_values":     {k: round(v, 3) for k, v in self.shap_values.items()},
            "explainer_mode":  self.explainer_mode,
            "latency_ms":      round(self.latency_ms, 1),
            "input_features":  self.input_features,
        }

    def __str__(self) -> str:
        w   = 62
        sep = "=" * w
        thin = "-" * w

        lines = [
            sep,
            f"  ATHLIX INJURY PREDICTION PIPELINE",
            f"  Completed in {self.latency_ms:.0f} ms",
            sep,
            "",
            f"  RISK SCORE   :  {self.risk_score:.1f} / 100",
            f"  RISK LEVEL   :  {self.risk_level.upper()}",
            thin,
        ]

        filled = int(self.risk_score / 100 * w)
        bar    = "#" * filled + "-" * (w - filled)
        lines += [f"  [{bar}]", ""]

        lines += [
            f"  Model score  : {self.model_score:.1f}  (XGBoost raw prediction)",
            f"  Fusion delta : +{self.fusion_delta:.1f} (rule-based adjustment)",
            "",
        ]

        if self.fusion_flags:
            lines.append(f"  Active risk conditions:")
            for flag in self.fusion_flags:
                lines.append(f"    ! {flag}")
            lines.append("")

        lines += [thin, f"  WHY IS RISK {self.risk_level.upper()}?", thin]
        for i, reason in enumerate(self.reasons, 1):
            lines.append(f"    {i}. {reason}")
        lines.append("")

        lines += [thin, f"  COACHING RECOMMENDATIONS  ({len(self.recommendations)})", thin]
        icon_map = {"Urgent": "[!!]", "Recommended": "[ >]", "Optional": "[ ?]"}
        for i, rec in enumerate(self.recommendations, 1):
            icon = icon_map.get(rec.get("priority", ""), "[ ]")
            lines.append(f"  {i}. {icon} [{rec['category']}]  {rec['title']}")
            lines.append(f"       {rec['detail']}")
            for drill in rec.get("drills", []):
                lines.append(f"         * {drill}")
            lines.append("")

        if self.positive_notes:
            lines += [thin, "  DOING WELL", thin]
            for note in self.positive_notes:
                lines.append(f"    + {note}")
            lines.append("")

        lines.append(sep)
        return "\n".join(lines)


def run_pipeline(
    input_features: dict,
    model_name: str = "xgboost",
) -> PipelineResult:
    t0 = time.perf_counter()

    validated = _validate_input(input_features)

    risk: RiskOutput = get_risk_score(validated, model_name=model_name)
    explanation: Explanation = explain_prediction(validated, model_name=model_name)
    coaching: CoachingReport = get_recommendations(validated)

    latency_ms = (time.perf_counter() - t0) * 1000

    return PipelineResult(
        risk_score      = risk.risk_score,
        risk_level      = risk.risk_level,
        reasons         = explanation.reasons,
        recommendations = [s.to_dict() for s in coaching.suggestions],
        positive_notes  = coaching.positive_notes,
        model_score     = risk.model_score,
        fusion_delta    = risk.fusion_delta,
        fusion_flags    = risk.flags,
        shap_values     = explanation.shap_values,
        explainer_mode  = explanation.mode,
        latency_ms      = latency_ms,
        input_features  = validated,
    )


if __name__ == "__main__":
    import json

    EXAMPLE_INPUT = {
        "training_load":   8.5,
        "recovery_score":  25.0,
        "fatigue_index":   8.1,
        "form_decay":      0.82,
        "previous_injury": 1,
    }

    result = run_pipeline(EXAMPLE_INPUT)

    output = result.to_dict()
    required_output = {
        "risk_score":      output["risk_score"],
        "risk_level":      output["risk_level"],
        "reasons":         output["reasons"],
        "recommendations": [
            {"priority": r["priority"], "category": r["category"], "title": r["title"]}
            for r in output["recommendations"]
        ],
    }
    print(json.dumps(required_output, indent=2))
