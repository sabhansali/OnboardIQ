from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

ACTIVE_STATUSES = {
    "OPEN", "ASSIGNED", "IN_PROGRESS", "PENDING_REVIEW", "ON_HOLD",
    "AWAITING_CUSTOMER", "ESCALATED", "APPROVED_FOR_RESPONSE",
}


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def deadline_pair(created_at: datetime, priority: str, config: dict[str, Any]) -> tuple[str, str]:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    first_response_minutes = int(config.get("sla_minutes", {}).get(priority, 240))
    resolution_multiplier = int(config.get("resolution_sla_multiplier", 4))
    first_due = created_at + timedelta(minutes=first_response_minutes)
    resolution_due = created_at + timedelta(minutes=first_response_minutes * resolution_multiplier)
    return first_due.astimezone(timezone.utc).isoformat(), resolution_due.astimezone(timezone.utc).isoformat()


def evaluate_sla(ticket: dict[str, Any], config: dict[str, Any], now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    first_due = parse_utc(ticket.get("first_response_due"))
    resolution_due = parse_utc(ticket.get("resolution_due"))
    first_response_at = parse_utc(ticket.get("first_response_at"))
    resolved_at = parse_utc(ticket.get("resolved_at"))
    at_risk_minutes = int(config.get("sla_at_risk_minutes", 15))

    if ticket.get("status") == "ON_HOLD" and ticket.get("sla_paused_at") and config.get("pause_sla_when_on_hold", False):
        return "PAUSED"

    if ticket.get("status") in {"RESOLVED", "CLOSED"}:
        if resolved_at and resolution_due and resolved_at <= resolution_due and (not first_due or (first_response_at and first_response_at <= first_due)):
            return "WITHIN_SLA"
        return "BREACHED"

    if resolution_due and now > resolution_due:
        return "BREACHED"
    if first_due and not first_response_at and now > first_due:
        return "BREACHED"
    if first_due and not first_response_at and first_due - now <= timedelta(minutes=at_risk_minutes):
        return "AT_RISK"
    if resolution_due and resolution_due - now <= timedelta(minutes=at_risk_minutes):
        return "AT_RISK"
    return "WITHIN_SLA"
