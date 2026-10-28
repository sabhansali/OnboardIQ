from __future__ import annotations

import time
from typing import Any

from core.database import (
    AuthorizationError,
    append_notification_attempt,
    finish_notification_cycle,
    get_notification,
    start_notification_cycle,
)


def dispatch_notification(
    db_path: str,
    notification_id: str,
    *,
    actor_id: str = "SYSTEM",
    max_attempts: int = 3,
    failure_mode: str = "normal",
    dead_letter_after_cycles: int = 2,
    backoff_base_seconds: float = 0.05,
) -> dict[str, Any]:
    """Simulate provider delivery with per-attempt logging and exponential backoff.

    failure_mode: normal, fail_first_two_then_success, or always_fail.
    This is a mock transport; no email or chat message leaves the application.
    """
    if failure_mode not in {"normal", "fail_first_two_then_success", "always_fail"}:
        raise ValueError("Unknown notification failure simulation mode.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    cycle_info = start_notification_cycle(db_path, notification_id, actor_id=actor_id)
    if cycle_info["already_sent"]:
        return {"notification_id": notification_id, "status": "SENT", "attempts_this_cycle": 0, "total_attempts": cycle_info["attempts"], "message": "Already sent; duplicate dispatch prevented."}

    cycle = int(cycle_info["cycle_number"])
    attempts_this_cycle = 0
    final_error = ""
    for attempt_in_cycle in range(1, max_attempts + 1):
        attempts_this_cycle += 1
        current = get_notification(db_path, notification_id)
        attempt_number = int(current["attempts"]) + 1 if current else attempt_in_cycle
        if failure_mode == "normal":
            success = True
        elif failure_mode == "fail_first_two_then_success":
            success = attempt_in_cycle >= 3
        else:
            success = False

        if success:
            append_notification_attempt(db_path, notification_id, cycle, attempt_number, "SUCCESS", "", actor_id=actor_id)
            result = finish_notification_cycle(db_path, notification_id, True, actor_id=actor_id, dead_letter_after_cycles=dead_letter_after_cycles)
            final = get_notification(db_path, notification_id) or {}
            return {
                "notification_id": notification_id,
                "status": result["status"],
                "attempts_this_cycle": attempts_this_cycle,
                "total_attempts": int(final.get("attempts", 0)),
                "message": "Simulated delivery succeeded.",
            }

        final_error = f"SIMULATED_TRANSIENT_FAILURE: delivery unavailable on attempt {attempt_in_cycle} of cycle {cycle}."
        append_notification_attempt(db_path, notification_id, cycle, attempt_number, "FAILURE", final_error, actor_id=actor_id)
        if attempt_in_cycle < max_attempts and backoff_base_seconds > 0:
            time.sleep(backoff_base_seconds * (2 ** (attempt_in_cycle - 1)))

    result = finish_notification_cycle(db_path, notification_id, False, final_error, actor_id=actor_id,
                                       dead_letter_after_cycles=dead_letter_after_cycles)
    final = get_notification(db_path, notification_id) or {}
    return {
        "notification_id": notification_id,
        "status": result["status"],
        "attempts_this_cycle": attempts_this_cycle,
        "total_attempts": int(final.get("attempts", 0)),
        "message": "All simulated delivery attempts failed; the notification was retained for investigation.",
    }


def build_notification_preview(notification: dict[str, Any]) -> str:
    return (
        f"To: {notification.get('recipient', '')}\n"
        f"Channel: {notification.get('channel', '')}\n"
        f"Subject: {notification.get('subject', '')}\n\n"
        f"{notification.get('message', '')}\n"
    )
