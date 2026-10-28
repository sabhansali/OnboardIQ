from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any

from sklearn.metrics import precision_recall_fscore_support

from core.database import (
    list_audit_events, list_notification_attempts, list_notifications, list_requests, list_ticket_events, list_tickets,
)


def _pct(numerator: float, denominator: float) -> float | None:
    return round(100.0 * numerator / denominator, 2) if denominator else None


def _duration_seconds(start: str, end: str) -> float | None:
    try:
        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return max(0.0, (b - a).total_seconds())
    except (ValueError, TypeError):
        return None


def compute_metrics(db_path: str) -> dict[str, Any]:
    tickets = list_tickets(db_path)
    requests = list_requests(db_path)
    notifications = list_notifications(db_path)
    attempts = list_notification_attempts(db_path)
    audit = list_audit_events(db_path, limit=100000)
    ticket_events = list_ticket_events(db_path, limit=100000)

    total = len(tickets)
    escalated = sum(t["status"] == "ESCALATED" for t in tickets)
    resolved = [t for t in tickets if t["status"] in {"RESOLVED", "CLOSED"}]
    auto_eligible = [t for t in tickets if t["priority"] in {"P3", "P4"} and t["complexity"] == "Low" and t["route"] != "PRODUCT_FEEDBACK_QUEUE"]
    auto_resolved = [t for t in auto_eligible if t["route"] == "AUTO_RESOLVE_APPROVED_KB" and t["status"] in {"RESOLVED", "CLOSED"}]

    delivery_candidates = [n for n in notifications if n["status"] != "DRAFT" and (not n["requires_approval"] or n["approved_by"])]
    sent = sum(n["status"] == "SENT" for n in delivery_candidates)
    within_sla_resolved = sum(t["sla_state"] == "WITHIN_SLA" for t in resolved)
    resolution_sla_rate = _pct(within_sla_resolved, len(resolved))
    average_processing = []
    for ticket in tickets:
        request = next((r for r in requests if r["request_id"] == ticket["request_id"]), None)
        if request:
            seconds = _duration_seconds(request["imported_at"], ticket["created_at"])
            if seconds is not None:
                average_processing.append(seconds)
    review_assigned_ids = {t["ticket_id"] for t in tickets if t["route"] in {"HUMAN_REVIEW", "SPECIALIST_HUMAN_REVIEW", "LOW_CONFIDENCE_SPECIALIST_REVIEW", "NO_APPROVED_ANSWER_ESCALATE", "URGENT_HUMAN_ESCALATION"}}
    review_assigned = len(review_assigned_ids)
    review_completed_ids = {e["ticket_id"] for e in ticket_events if e["action"] in {"APPROVE", "EDIT", "HOLD", "REQUEST_INFORMATION", "ESCALATE", "RESOLVE"} and e["ticket_id"] in review_assigned_ids}
    reopened = sum(int(t["reopen_count"]) > 0 for t in tickets)
    attempt_by_notification: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        attempt_by_notification.setdefault(attempt["notification_id"], []).append(attempt)
    recovered = 0
    for notification in notifications:
        rows = attempt_by_notification.get(notification["notification_id"], [])
        if notification["status"] == "SENT" and any(row["result"] == "FAILURE" for row in rows):
            recovered += 1
    distinct_ticket_events = Counter(event["ticket_id"] for event in audit if event.get("ticket_id"))
    audit_complete = sum(distinct_ticket_events[t["ticket_id"]] >= 2 for t in tickets)

    labelled_rows = []
    request_by_id = {r["request_id"]: r for r in requests}
    for ticket in tickets:
        raw = request_by_id.get(ticket["request_id"], {}).get("raw_json", "{}")
        try:
            fields = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        expected = {
            "intent": fields.get("expected_intent"),
            "complexity": fields.get("expected_complexity"),
            "priority": fields.get("expected_priority"),
        }
        if all(expected.values()):
            labelled_rows.append((ticket, expected))

    classification = {}
    classification_reports = {}
    for dimension in ("intent", "complexity", "priority"):
        known = [(ticket, expected) for ticket, expected in labelled_rows if expected.get(dimension)]
        correct = sum(ticket[dimension] == expected[dimension] for ticket, expected in known)
        y_true = [expected[dimension] for ticket, expected in known]
        y_pred = [ticket[dimension] for ticket, expected in known]
        labels = sorted(set(y_true) | set(y_pred))
        if known:
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
            report = [{"Class": label, "Precision": round(float(p), 3), "Recall": round(float(r), 3),
                       "F1": round(float(f), 3), "Support": int(n)}
                      for label, p, r, f, n in zip(labels, precision, recall, f1, support)]
            macro_f1 = round(float(sum(f1) / len(f1)), 3) if len(f1) else None
        else:
            report, macro_f1 = [], None
        classification[dimension] = {"correct": correct, "total": len(known), "accuracy_pct": _pct(correct, len(known)), "macro_f1": macro_f1}
        classification_reports[dimension] = report

    return {
        "total_tickets": total,
        "active_tickets": sum(t["status"] not in {"RESOLVED", "CLOSED"} for t in tickets),
        "pending_review": sum(t["status"] == "PENDING_REVIEW" for t in tickets),
        "escalated_tickets": escalated,
        "resolved_tickets": len(resolved),
        "p1_tickets": sum(t["priority"] == "P1" for t in tickets),
        "breached_tickets": sum(t["sla_state"] == "BREACHED" for t in tickets),
        "classification": classification,
        "classification_labelled_total": len(labelled_rows),
        "classification_reports": classification_reports,
        "kb_resolution_rate_pct": _pct(len(auto_resolved), len(auto_eligible)),
        "kb_resolution_successes": len(auto_resolved),
        "kb_resolution_eligible": len(auto_eligible),
        "escalation_rate_pct": _pct(escalated, total),
        "escalation_count": escalated,
        "valid_ticket_count": total,
        "notification_success_rate_pct": _pct(sent, len(delivery_candidates)),
        "notifications_sent": sent,
        "notification_delivery_requests": len(delivery_candidates),
        "sla_compliance_pct": resolution_sla_rate,
        "sla_compliant_resolved": within_sla_resolved,
        "resolved_with_sla": len(resolved),
        "average_processing_seconds": round(sum(average_processing) / len(average_processing), 3) if average_processing else None,
        "review_completion_rate_pct": _pct(len(review_completed_ids), review_assigned),
        "review_actions_recorded": len(review_completed_ids),
        "review_assigned": review_assigned,
        "reopened_ticket_rate_pct": _pct(reopened, total),
        "reopened_tickets": reopened,
        "retry_recovered_notifications": recovered,
        "audit_completeness_pct": _pct(audit_complete, total),
        "tickets_with_audit_events": audit_complete,
        "tickets_total_for_audit": total,
        "false_auto_resolution_policy_violations": sum(t["route"] == "AUTO_RESOLVE_APPROVED_KB" and (t["priority"] in {"P1", "P2"} or t["complexity"] != "Low") for t in tickets),
        "notification_status_counts": dict(Counter(n["status"] for n in notifications)),
    }


def keyword_only_retrieval(request_text: str, knowledge_base: list[dict[str, Any]]) -> tuple[str | None, float]:
    """Baseline retriever used only for comparison against the hybrid retrieval extension."""
    tokens = {x.lower() for x in request_text.replace("/", " ").replace("-", " ").split() if len(x) > 2}
    best_article, best_score = None, 0.0
    for article in knowledge_base:
        if article.get("approved") is not True:
            continue
        keywords = " ".join(article.get("keywords", [])) + " " + article.get("title", "")
        article_tokens = {x.lower() for x in keywords.replace("/", " ").replace("-", " ").split() if len(x) > 2}
        score = 100 * len(tokens & article_tokens) / max(1, len(tokens))
        if score > best_score:
            best_article, best_score = article, score
    return (best_article.get("article_id") if best_article and best_score > 0 else None, round(best_score, 2))
