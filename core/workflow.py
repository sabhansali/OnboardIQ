from __future__ import annotations

from pathlib import Path
from typing import Any
import time

from core.classifier import classify_request
from core.database import (
    DuplicateRequestError,
    create_ticket_bundle,
    get_ticket,
    finalize_import_batch,
    record_rejected_row,
    set_ticket_processing_seconds,
    log_audit,
)
from core.notifications import dispatch_notification
from core.resolver import retrieve_article
from core.workflow_router import route_request


def process_one_request(
    db_path: str | Path,
    request: dict[str, Any],
    batch_id: str,
    actor_id: str,
    knowledge_base: list[dict[str, Any]],
    config: dict[str, Any],
    use_groq: bool = False,
) -> dict[str, Any]:
    # Measure decisioning + ticket persistence with a monotonic high-resolution clock.
    # This intentionally excludes file parsing/validation, notification delivery, and
    # any time spent waiting for a human reviewer.
    processing_started = time.perf_counter()
    classification = classify_request(request["request_text"], use_groq=use_groq)
    article, match_score, retrieval_diagnostics = retrieve_article(
        request["request_text"], knowledge_base,
        min_score=float(config.get("minimum_retrieval_score", 18)),
    )
    routing = route_request(classification, match_score, article is not None, config)
    result = create_ticket_bundle(
        db_path=db_path,
        request=request,
        classification=classification,
        article=article,
        match_score=match_score,
        retrieval_diagnostics=retrieval_diagnostics,
        routing=routing,
        actor_id=actor_id,
        batch_id=batch_id,
        config=config,
    )
    processing_seconds = time.perf_counter() - processing_started
    try:
        set_ticket_processing_seconds(db_path, result["ticket_id"], processing_seconds)
        result["processing_seconds"] = processing_seconds
    except Exception as exc:
        # A metrics-write problem must not turn an already-created ticket into a false
        # rejected import row. Leave the metric NULL and record the diagnostic event.
        try:
            log_audit(
                str(db_path), "SYSTEM", "PROCESSING_TIME_PERSIST_FAILED",
                request_id=request.get("request_id"), ticket_id=result["ticket_id"],
                details="The ticket was created, but its measured processing duration could not be saved.",
                error_details=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
    deliveries: list[dict[str, Any]] = []
    for notification_id in result["notification_ids"]:
        # Auto-dispatch is restricted to system-authorized, non-approval-gated drafts/queues.
        try:
            deliveries.append(dispatch_notification(
                str(db_path), notification_id, actor_id="SYSTEM",
                max_attempts=int(config.get("notification_retry_attempts", 3)),
                failure_mode="normal",
                dead_letter_after_cycles=int(config.get("notification_dead_letter_cycles", 2)),
            ))
        except Exception as exc:
            # A notification fault must not roll back the ticket or hide the failure.
            from core.database import log_audit
            log_audit(str(db_path), "SYSTEM", "NOTIFICATION_DISPATCH_ERROR", ticket_id=result["ticket_id"],
                      details=f"Automatic dispatch raised {type(exc).__name__}.", error_details=str(exc))
    persisted_ticket = get_ticket(db_path, result["ticket_id"])
    if persisted_ticket:
        result["status"] = persisted_ticket["status"]
    result.update({
        "classification": classification,
        "article": article,
        "match_score": match_score,
        "retrieval_diagnostics": retrieval_diagnostics,
        "routing": routing,
        "deliveries": deliveries,
    })
    return result


def process_import_batch(
    db_path: str | Path,
    batch_id: str,
    source_name: str,
    valid_requests: list[dict[str, Any]],
    actor_id: str,
    knowledge_base: list[dict[str, Any]],
    config: dict[str, Any],
    use_groq: bool = False,
) -> dict[str, Any]:
    processed = 0
    created: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, request in enumerate(valid_requests, start=1):
        try:
            created.append(process_one_request(db_path, request, batch_id, actor_id, knowledge_base, config, use_groq))
            processed += 1
        except DuplicateRequestError as exc:
            record_rejected_row(db_path, batch_id, source_name, int(request.get("_source_row_number", index + 1)), request, [str(exc)], actor_id)
            errors.append({"request_id": request.get("request_id", ""), "error": str(exc)})
        except Exception as exc:
            record_rejected_row(db_path, batch_id, source_name, int(request.get("_source_row_number", index + 1)), request,
                                [f"Processing failure: {type(exc).__name__}: {exc}"], actor_id)
            errors.append({"request_id": request.get("request_id", ""), "error": f"{type(exc).__name__}: {exc}"})
    finalize_import_batch(db_path, batch_id, processed, error_message=f"{len(errors)} request(s) failed during processing." if errors else "")
    return {"batch_id": batch_id, "processed_count": processed, "created": created, "errors": errors}
