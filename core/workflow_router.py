from __future__ import annotations

from typing import Any


def select_team(intent: str, priority: str) -> str:
    if priority == "P1":
        return "Incident Response"
    return {
        "Workspace Setup": "Onboarding Support",
        "Data Import & Migration": "Data Migration Team",
        "Identity & Access": "Identity & Security Team",
        "Permissions & Security": "Identity & Security Team",
        "Integration Setup": "Integration Specialists",
        "Feature Request": "Product Feedback",
        "Product Guidance": "Onboarding Support",
        "Unresolved Implementation Issue": "Implementation Specialists",
        "Other": "Implementation Specialists",
    }.get(intent, "Implementation Specialists")


def route_request(
    classification: dict[str, Any],
    match_score: float,
    article_found: bool,
    config: dict[str, Any],
) -> dict[str, str]:
    """Choose a proposed route. Critical incidents always bypass auto-resolution."""
    priority = classification["priority"]
    complexity = classification["complexity"]
    intent = classification["intent"]
    team = select_team(intent, priority)

    if priority == "P1":
        return {"route": "URGENT_HUMAN_ESCALATION", "status": "ESCALATED", "assigned_team": "Incident Response"}
    if intent == "Feature Request":
        return {"route": "PRODUCT_FEEDBACK_QUEUE", "status": "ASSIGNED", "assigned_team": "Product Feedback"}
    if complexity == "High":
        return {"route": "SPECIALIST_HUMAN_REVIEW", "status": "ESCALATED", "assigned_team": team}
    if not article_found:
        return {"route": "NO_APPROVED_ANSWER_ESCALATE", "status": "ESCALATED", "assigned_team": team}

    auto_threshold = float(config.get("kb_match_threshold", 70))
    review_threshold = float(config.get("kb_review_threshold", 40))
    low_complexity_required = bool(config.get("auto_resolve_requires_low_complexity", True))
    low_risk_priority = priority in {"P3", "P4"}
    complexity_eligible = not low_complexity_required or complexity == "Low"

    if match_score >= auto_threshold and low_risk_priority and complexity_eligible:
        return {"route": "AUTO_RESOLVE_APPROVED_KB", "status": "RESOLVED", "assigned_team": "Automation"}
    if match_score >= review_threshold:
        return {"route": "HUMAN_REVIEW", "status": "PENDING_REVIEW", "assigned_team": team}
    return {"route": "LOW_CONFIDENCE_SPECIALIST_REVIEW", "status": "ESCALATED", "assigned_team": team}
