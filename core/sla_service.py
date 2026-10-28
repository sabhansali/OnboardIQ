from __future__ import annotations

from pathlib import Path
from typing import Any

from core.database import create_sla_alert_notification, list_tickets, log_audit, update_ticket_sla
from core.notifications import dispatch_notification
from core.sla_monitor import evaluate_sla


def refresh_sla_states(db_path: str | Path, config: dict[str, Any]) -> dict[str, int]:
    counts = {"updated": 0, "WITHIN_SLA": 0, "AT_RISK": 0, "BREACHED": 0, "PAUSED": 0, "alerts_created": 0}
    for ticket in list_tickets(db_path):
        state = evaluate_sla(ticket, config)
        counts[state] = counts.get(state, 0) + 1
        changed = update_ticket_sla(db_path, ticket["ticket_id"], state)
        if changed:
            counts["updated"] += 1
            if state in {"AT_RISK", "BREACHED"}:
                notification_id = create_sla_alert_notification(db_path, ticket["ticket_id"], state)
                if notification_id:
                    counts["alerts_created"] += 1
                    try:
                        dispatch_notification(
                            str(db_path), notification_id, actor_id="SYSTEM",
                            max_attempts=int(config.get("notification_retry_attempts", 3)),
                            dead_letter_after_cycles=int(config.get("notification_dead_letter_cycles", 2)),
                            failure_mode="normal",
                        )
                    except Exception as exc:
                        log_audit(str(db_path), "SYSTEM", "SLA_ALERT_DISPATCH_ERROR", ticket_id=ticket["ticket_id"],
                                  details=f"SLA state {state} alert could not be simulated as delivered.",
                                  error_details=f"{type(exc).__name__}: {exc}")
    return counts
