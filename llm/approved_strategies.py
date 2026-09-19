"""
Approved Agile mitigation strategies for the Sprint Intelligence LLM recommendation layer.

The LLM is only allowed to select and phrase recommendations FROM this list — it is not
responsible for predicting risk and should not invent new mitigation strategies. Keeping
this list in its own file makes it easy for a Scrum Master / Product Owner to review and
edit the allowed vocabulary without touching the generation code.

Each entry has:
  - id: short stable identifier (used for validation and logging)
  - text: the canonical phrasing the LLM should draw from
  - applicable_tasks: which risk task(s) this strategy is commonly used for
                      ("requirement_volatility", "resolution_risk", "reopen_risk", "any")
"""

APPROVED_STRATEGIES = [
    {
        "id": "split_story",
        "text": "Split this issue into smaller, independently deliverable stories.",
        "applicable_tasks": ["requirement_volatility", "resolution_risk"],
    },
    {
        "id": "clarify_acceptance_criteria",
        "text": "Clarify and re-confirm acceptance criteria with the Product Owner before committing.",
        "applicable_tasks": ["requirement_volatility"],
    },
    {
        "id": "stable_ownership",
        "text": "Assign a single stable owner instead of rotating assignees.",
        "applicable_tasks": ["resolution_risk", "reopen_risk"],
    },
    {
        "id": "review_dependencies",
        "text": "Review and resolve upstream/downstream dependencies before sprint commitment.",
        "applicable_tasks": ["resolution_risk"],
    },
    {
        "id": "re_estimate",
        "text": "Re-estimate the work with the team before committing it to the sprint.",
        "applicable_tasks": ["resolution_risk", "requirement_volatility"],
    },
    {
        "id": "defer_for_refinement",
        "text": "Defer this item to a future sprint pending backlog refinement.",
        "applicable_tasks": ["requirement_volatility"],
    },
    {
        "id": "add_qa_checkpoint",
        "text": "Add an explicit QA/testing checkpoint before marking this issue done.",
        "applicable_tasks": ["reopen_risk"],
    },
    {
        "id": "pair_or_review",
        "text": "Use pair programming or an additional code review pass given the risk factors.",
        "applicable_tasks": ["resolution_risk", "reopen_risk"],
    },
    {
        "id": "increase_monitoring",
        "text": "Include the item in the sprint but flag it for daily stand-up follow-up.",
        "applicable_tasks": ["any"],
    },
    {
        "id": "reduce_scope",
        "text": "Reduce scope to the minimum viable version of the requirement for this sprint.",
        "applicable_tasks": ["requirement_volatility"],
    },
]

# Shared risk bands. These match the dashboard and risk-aggregation notebook.
LOW_RISK_MAX = 0.40
MEDIUM_RISK_MAX = 0.70

# Used only when the LLM returns no valid approved strategy ID.
DEFAULT_STRATEGY_BY_TASK = {
    "requirement_volatility": "clarify_acceptance_criteria",
    "resolution_risk": "review_dependencies",
    "reopen_risk": "add_qa_checkpoint",
}


def get_strategy_texts(task: str | None = None) -> list[str]:
    """Return the canonical strategy strings, optionally filtered to one task."""
    if task is None:
        return [s["text"] for s in APPROVED_STRATEGIES]
    return [
        s["text"]
        for s in APPROVED_STRATEGIES
        if task in s["applicable_tasks"] or "any" in s["applicable_tasks"]
    ]


def get_strategies(task: str) -> list[dict]:
    """Return approved strategy records applicable to one risk task."""
    return [
        strategy.copy()
        for strategy in APPROVED_STRATEGIES
        if task in strategy["applicable_tasks"] or "any" in strategy["applicable_tasks"]
    ]


def risk_level_for_probability(probability: float) -> str:
    """Use the same thresholds in the model pipeline, LLM, and dashboard."""
    if probability <= LOW_RISK_MAX:
        return "Low"
    if probability <= MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def get_strategy_by_id(strategy_id: str) -> dict:
    for s in APPROVED_STRATEGIES:
        if s["id"] == strategy_id:
            return s
    raise KeyError(f"No approved strategy with id={strategy_id!r}")


def get_default_strategy(task: str) -> dict:
    return get_strategy_by_id(DEFAULT_STRATEGY_BY_TASK[task])
