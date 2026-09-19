"""Sprint Intelligence — grounded local Ollama recommendation layer.

The predictive models calculate risk scores. Qwen only selects approved
strategy IDs and supplies a short rationale. The dashboard always displays
canonical strategy text from approved_strategies.py, never raw model advice.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .approved_strategies import get_default_strategy, get_strategies, risk_level_for_probability

DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
VALID_TASKS = {"requirement_volatility", "resolution_risk", "reopen_risk"}
VALID_ACTIONS = {"Include", "Split", "Defer", "Monitor"}

SYSTEM_PROMPT = """You are the recommendation layer of an Agile Sprint Planning tool.
Risk probabilities and top factors were computed by a separate predictive model.
Do not predict, change, or challenge them. Select one or two IDs only from the
approved list. Return exactly one JSON object with selected_strategy_ids (array)
and rationale (one practical sentence). Do not add a strategy, tool, or process
change that is not represented by one of the selected approved strategy IDs."""

USER_TEMPLATE = """Issue: {issue_key}
Risk type: {risk_type}
Risk probability: {risk_probability:.0%}
Risk level: {risk_level}
Planner action: {planner_action}
Top contributing factors: {factors_block}

Approved strategies:
{strategies_block}

Return JSON only."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_strategy_ids": {
            "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2,
        },
        "rationale": {"type": "string"},
    },
    "required": ["selected_strategy_ids", "rationale"],
    "additionalProperties": False,
}


@dataclass
class RiskInput:
    issue_key: str
    risk_type: str
    risk_probability: float
    top_factors: list[str]
    planner_action: str
    risk_level: str = field(default="")

    def __post_init__(self):
        if self.risk_type not in VALID_TASKS:
            raise ValueError(f"Unsupported risk type: {self.risk_type}")
        if not 0 <= self.risk_probability <= 1:
            raise ValueError("risk_probability must be between 0 and 1.")
        if self.planner_action not in VALID_ACTIONS:
            raise ValueError(f"Unsupported planner action: {self.planner_action}")
        if not self.risk_level:
            self.risk_level = risk_level_for_probability(self.risk_probability)


class MitigationRecommender:
    """Call a locally running Ollama Qwen model without API keys or SDKs."""

    def __init__(self, model_name: str | None = None, base_url: str | None = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).rstrip("/")

    @staticmethod
    def _safe_factor(factor: str) -> str:
        return re.sub(r"\s+", " ", str(factor)).strip().replace("{", "(").replace("}", ")")[:160]

    def _build_user_message(self, risk: RiskInput) -> tuple[str, set[str]]:
        approved = get_strategies(risk.risk_type)
        allowed_ids = {strategy["id"] for strategy in approved}
        factors = "; ".join(self._safe_factor(factor) for factor in risk.top_factors[:5]) or "No factor explanation available"
        strategies = "\n".join(f'- {item["id"]}: {item["text"]}' for item in approved)
        return USER_TEMPLATE.format(
            issue_key=risk.issue_key, risk_type=risk.risk_type,
            risk_probability=risk.risk_probability, risk_level=risk.risk_level,
            planner_action=risk.planner_action, factors_block=factors, strategies_block=strategies,
        ), allowed_ids

    def _chat(self, user_message: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "options": {"temperature": 0},
        }
        request = Request(
            f"{self.base_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Ollama could not run '{self.model_name}': {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. Start Ollama, then pull '{self.model_name}'."
            ) from exc
        try:
            return str(body["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama returned an unexpected chat response.") from exc

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def generate(self, risk: RiskInput) -> dict:
        user_message, allowed_ids = self._build_user_message(risk)
        parsed = self._parse_json(self._chat(user_message))
        requested_ids = parsed.get("selected_strategy_ids", [])
        if not isinstance(requested_ids, list):
            requested_ids = []
        selected_ids = [strategy_id for strategy_id in requested_ids if strategy_id in allowed_ids][:2]
        fallback_used = not selected_ids
        if fallback_used:
            selected_ids = [get_default_strategy(risk.risk_type)["id"]]

        approved_by_id = {strategy["id"]: strategy for strategy in get_strategies(risk.risk_type)}
        rationale = str(parsed.get("rationale", "")).strip()[:300]
        if not rationale:
            rationale = "Selected from the approved mitigation strategy list based on the predicted risk signals."
        return {
            "issue_key": risk.issue_key, "risk_type": risk.risk_type,
            "risk_probability": risk.risk_probability, "risk_level": risk.risk_level,
            "planner_action": risk.planner_action, "selected_strategy_ids": selected_ids,
            "approved_recommendations": [approved_by_id[strategy_id]["text"] for strategy_id in selected_ids],
            "rationale": rationale, "fallback_used": fallback_used,
        }
