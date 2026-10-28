from __future__ import annotations

import json
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from core.security import bootstrap_users_from_environment, can, hash_password, verify_password
from core.sla_monitor import deadline_pair


class AuthorizationError(PermissionError):
    pass


class DuplicateRequestError(ValueError):
    pass


class StaleTicketError(RuntimeError):
    pass


class InvalidTransitionError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def _dict_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


@contextmanager
def _connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def init_db(db_path: str | Path) -> None:
    with _connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('ADMIN','MANAGER','SPECIALIST','AGENT')),
                active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS import_batches (
                batch_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                total_rows INTEGER NOT NULL DEFAULT 0,
                accepted_count INTEGER NOT NULL DEFAULT 0,
                rejected_count INTEGER NOT NULL DEFAULT 0,
                processed_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS rejected_records (
                rejected_id TEXT PRIMARY KEY,
                batch_id TEXT,
                source_name TEXT NOT NULL,
                row_number INTEGER,
                raw_record_json TEXT NOT NULL,
                errors_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(batch_id) REFERENCES import_batches(batch_id)
            );
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                company_name TEXT NOT NULL,
                contact_email TEXT NOT NULL,
                request_text TEXT NOT NULL,
                source TEXT NOT NULL,
                request_created_at TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                import_batch_id TEXT,
                raw_json TEXT NOT NULL,
                FOREIGN KEY(import_batch_id) REFERENCES import_batches(batch_id)
            );
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                intent TEXT NOT NULL,
                complexity TEXT NOT NULL,
                priority TEXT NOT NULL CHECK(priority IN ('P1','P2','P3','P4')),
                classifier_source TEXT NOT NULL,
                classifier_confidence REAL NOT NULL,
                article_id TEXT,
                article_title TEXT,
                match_score REAL NOT NULL,
                retrieval_diagnostics_json TEXT NOT NULL,
                proposed_response TEXT NOT NULL DEFAULT '',
                approved_response TEXT NOT NULL DEFAULT '',
                response_version INTEGER NOT NULL DEFAULT 1,
                route TEXT NOT NULL,
                assigned_team TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                processing_seconds REAL,
                first_response_due TEXT NOT NULL,
                resolution_due TEXT NOT NULL,
                first_response_at TEXT,
                resolved_at TEXT,
                sla_state TEXT NOT NULL DEFAULT 'WITHIN_SLA',
                sla_paused_at TEXT,
                sla_pause_seconds INTEGER NOT NULL DEFAULT 0,
                reopen_count INTEGER NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(request_id) REFERENCES requests(request_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tickets_status_priority ON tickets(status, priority);
            CREATE INDEX IF NOT EXISTS idx_tickets_sla ON tickets(sla_state, first_response_due, resolution_due);
            CREATE TABLE IF NOT EXISTS ticket_events (
                event_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                action TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT,
                note TEXT NOT NULL DEFAULT '',
                response_before TEXT NOT NULL DEFAULT '',
                response_after TEXT NOT NULL DEFAULT '',
                ticket_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                request_id TEXT,
                ticket_id TEXT,
                event_type TEXT NOT NULL,
                intent TEXT,
                complexity TEXT,
                priority TEXT,
                article_id TEXT,
                match_score REAL,
                route TEXT,
                reviewer_action TEXT,
                ticket_status TEXT,
                notification_status TEXT,
                details TEXT NOT NULL DEFAULT '',
                error_details TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp DESC);
            CREATE TABLE IF NOT EXISTS notifications (
                notification_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                ticket_id TEXT NOT NULL,
                recipient TEXT NOT NULL,
                channel TEXT NOT NULL CHECK(channel IN ('EMAIL','IN_APP','MESSAGING')),
                event_type TEXT NOT NULL,
                subject TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('DRAFT','QUEUED','SENT','FAILED','DEAD_LETTER')),
                requires_approval INTEGER NOT NULL DEFAULT 0 CHECK(requires_approval IN (0,1)),
                approved_by TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                dispatch_cycles INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                sent_at TEXT,
                FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
            );
            CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status, created_at DESC);
            CREATE TABLE IF NOT EXISTS notification_attempts (
                attempt_id TEXT PRIMARY KEY,
                notification_id TEXT NOT NULL,
                cycle_number INTEGER NOT NULL,
                attempt_number INTEGER NOT NULL,
                result TEXT NOT NULL,
                error_message TEXT NOT NULL DEFAULT '',
                attempted_at TEXT NOT NULL,
                FOREIGN KEY(notification_id) REFERENCES notifications(notification_id)
            );
            """
        )
        ticket_columns = {row[1] for row in conn.execute("PRAGMA table_info(tickets)").fetchall()}
        if "sla_paused_at" not in ticket_columns:
            conn.execute("ALTER TABLE tickets ADD COLUMN sla_paused_at TEXT")
        if "sla_pause_seconds" not in ticket_columns:
            conn.execute("ALTER TABLE tickets ADD COLUMN sla_pause_seconds INTEGER NOT NULL DEFAULT 0")
        # Existing databases are migrated in place. NULL means that a ticket predates
        # processing-time measurement; we deliberately do not fabricate a duration.
        if "processing_seconds" not in ticket_columns:
            conn.execute("ALTER TABLE tickets ADD COLUMN processing_seconds REAL")
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            with _transaction(conn):
                for seed in bootstrap_users_from_environment():
                    password_hash, salt = hash_password(seed["password"])
                    conn.execute(
                        "INSERT OR IGNORE INTO users(user_id, username, password_hash, password_salt, role, active, created_at) VALUES(?,?,?,?,?,1,?)",
                        (new_id("USR"), seed["username"], password_hash, salt, seed["role"], utc_now()),
                    )


def authenticate_user(db_path: str | Path, username: str, password: str) -> dict[str, Any] | None:
    with _connection(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE AND active = 1", (username.strip(),)).fetchone()
    if not row or not verify_password(password, row["password_hash"], row["password_salt"]):
        return None
    return {"user_id": row["user_id"], "username": row["username"], "role": row["role"], "active": bool(row["active"])}


def get_user_by_id(db_path: str | Path, user_id: str) -> dict[str, Any] | None:
    with _connection(db_path) as conn:
        row = conn.execute("SELECT user_id, username, role, active, created_at FROM users WHERE user_id=?", (user_id,)).fetchone()
    return _dict_row(row)


def list_users(db_path: str | Path) -> list[dict[str, Any]]:
    with _connection(db_path) as conn:
        rows = conn.execute("SELECT user_id, username, role, active, created_at FROM users ORDER BY username").fetchall()
    return [dict(row) for row in rows]


def create_user(db_path: str | Path, actor_id: str, username: str, password: str, role: str) -> dict[str, Any]:
    role = role.upper().strip()
    if role not in {"ADMIN", "MANAGER", "SPECIALIST", "AGENT"}:
        raise ValueError("Unsupported user role.")
    if len(username.strip()) < 3:
        raise ValueError("Username must contain at least three characters.")
    if len(password) < 10:
        raise ValueError("Password must contain at least ten characters.")
    password_hash, salt = hash_password(password)
    user = {"user_id": new_id("USR"), "username": username.strip(), "role": role}
    with _connection(db_path) as conn:
        with _transaction(conn):
            actor = conn.execute("SELECT role, active FROM users WHERE user_id=?", (actor_id,)).fetchone()
            if not actor or not actor["active"] or not can(actor["role"], "MANAGE_USERS"):
                raise AuthorizationError("Only an active ADMIN can manage users.")
            try:
                conn.execute(
                    "INSERT INTO users(user_id, username, password_hash, password_salt, role, active, created_at) VALUES(?,?,?,?,?,1,?)",
                    (user["user_id"], user["username"], password_hash, salt, role, utc_now()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("That username already exists.") from exc
            _audit(conn, actor_id, actor["role"], "USER_CREATED", details=f"Created user {user['username']} with role {role}.")
    return user


def set_user_active(db_path: str | Path, actor_id: str, target_user_id: str, active: bool) -> None:
    with _connection(db_path) as conn:
        with _transaction(conn):
            actor = conn.execute("SELECT role, active FROM users WHERE user_id=?", (actor_id,)).fetchone()
            target = conn.execute("SELECT username FROM users WHERE user_id=?", (target_user_id,)).fetchone()
            if not actor or not actor["active"] or not can(actor["role"], "MANAGE_USERS"):
                raise AuthorizationError("Only an active ADMIN can manage users.")
            if not target:
                raise ValueError("User not found.")
            if actor_id == target_user_id and not active:
                raise ValueError("You cannot deactivate your own account.")
            conn.execute("UPDATE users SET active=? WHERE user_id=?", (int(active), target_user_id))
            _audit(conn, actor_id, actor["role"], "USER_ACTIVATED" if active else "USER_DEACTIVATED", details=f"User {target['username']} active={active}.")


def _audit(
    conn: sqlite3.Connection,
    actor_id: str,
    actor_role: str,
    event_type: str,
    request_id: str | None = None,
    ticket_id: str | None = None,
    intent: str | None = None,
    complexity: str | None = None,
    priority: str | None = None,
    article_id: str | None = None,
    match_score: float | None = None,
    route: str | None = None,
    reviewer_action: str | None = None,
    ticket_status: str | None = None,
    notification_status: str | None = None,
    details: str = "",
    error_details: str = "",
) -> None:
    conn.execute(
        """INSERT INTO audit_events(event_id,timestamp,actor_id,actor_role,request_id,ticket_id,event_type,
           intent,complexity,priority,article_id,match_score,route,reviewer_action,ticket_status,
           notification_status,details,error_details) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (new_id("AUD"), utc_now(), actor_id, actor_role, request_id, ticket_id, event_type, intent,
         complexity, priority, article_id, match_score, route, reviewer_action, ticket_status,
         notification_status, details, error_details),
    )


def log_audit(
    db_path: str | Path,
    actor_id: str,
    event_type: str,
    *, request_id: str | None = None, ticket_id: str | None = None,
    details: str = "", error_details: str = "", reviewer_action: str | None = None,
) -> None:
    with _connection(db_path) as conn:
        with _transaction(conn):
            if actor_id == "SYSTEM":
                actor_role = "SYSTEM"
            else:
                user = conn.execute("SELECT role, active FROM users WHERE user_id=?", (actor_id,)).fetchone()
                actor_role = user["role"] if user and user["active"] else "UNKNOWN"
            _audit(conn, actor_id, actor_role, event_type, request_id, ticket_id, reviewer_action=reviewer_action,
                   details=details, error_details=error_details)


def create_import_batch(
    db_path: str | Path,
    source_name: str,
    total_rows: int,
    rejected: list[dict[str, Any]],
    created_by: str,
) -> str:
    batch_id = new_id("BATCH")
    with _connection(db_path) as conn:
        with _transaction(conn):
            actor = conn.execute("SELECT role, active FROM users WHERE user_id=?", (created_by,)).fetchone()
            if not actor or not actor["active"] or not can(actor["role"], "VIEW_TICKETS"):
                raise AuthorizationError("An active, authorized identity is required to import requests.")
            conn.execute(
                "INSERT INTO import_batches(batch_id,source_name,total_rows,accepted_count,rejected_count,processed_count,status,created_by,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (batch_id, source_name, total_rows, max(0, total_rows - len(rejected)), len(rejected), 0, "PROCESSING", created_by, utc_now()),
            )
            for item in rejected:
                conn.execute(
                    "INSERT INTO rejected_records(rejected_id,batch_id,source_name,row_number,raw_record_json,errors_json,created_at) VALUES(?,?,?,?,?,?,?)",
                    (new_id("REJ"), batch_id, source_name, item.get("row_number"),
                     json.dumps(item.get("record", {}), ensure_ascii=False), json.dumps(item.get("errors", []), ensure_ascii=False), utc_now()),
                )
            _audit(conn, created_by, actor["role"], "IMPORT_BATCH_CREATED", details=f"Source={source_name}; rows={total_rows}; schema rejects={len(rejected)}.")
    return batch_id


def record_rejected_row(
    db_path: str | Path,
    batch_id: str,
    source_name: str,
    row_number: int | None,
    record: dict[str, Any],
    errors: list[str],
    actor_id: str,
) -> None:
    with _connection(db_path) as conn:
        with _transaction(conn):
            actor = conn.execute("SELECT role FROM users WHERE user_id=?", (actor_id,)).fetchone()
            conn.execute(
                "INSERT INTO rejected_records(rejected_id,batch_id,source_name,row_number,raw_record_json,errors_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (new_id("REJ"), batch_id, source_name, row_number, json.dumps(record, ensure_ascii=False), json.dumps(errors, ensure_ascii=False), utc_now()),
            )
            conn.execute("UPDATE import_batches SET rejected_count=rejected_count+1, accepted_count=MAX(0, accepted_count-1) WHERE batch_id=?", (batch_id,))
            _audit(conn, actor_id, actor["role"] if actor else "UNKNOWN", "IMPORT_ROW_REJECTED", details=f"Batch={batch_id}; row={row_number}; errors={errors}.")


def finalize_import_batch(db_path: str | Path, batch_id: str, processed_count: int, error_message: str = "") -> None:
    with _connection(db_path) as conn:
        with _transaction(conn):
            conn.execute(
                "UPDATE import_batches SET processed_count=?, status=?, completed_at=?, error_message=? WHERE batch_id=?",
                (processed_count, "COMPLETED_WITH_ERRORS" if error_message else "COMPLETED", utc_now(), error_message, batch_id),
            )


def _insert_notification(
    conn: sqlite3.Connection, ticket_id: str, idempotency_key: str, recipient: str,
    channel: str, event_type: str, subject: str, message: str,
    status: str, requires_approval: bool, approved_by: str | None = None,
) -> str | None:
    notification_id = new_id("NTF")
    try:
        conn.execute(
            """INSERT INTO notifications(notification_id,idempotency_key,ticket_id,recipient,channel,event_type,subject,message,status,
               requires_approval,approved_by,attempts,dispatch_cycles,last_error,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,0,0,'',?,?)""",
            (notification_id, idempotency_key, ticket_id, recipient, channel, event_type, subject, message,
             status, int(requires_approval), approved_by, utc_now(), utc_now()),
        )
        return notification_id
    except sqlite3.IntegrityError:
        row = conn.execute("SELECT notification_id FROM notifications WHERE idempotency_key=?", (idempotency_key,)).fetchone()
        return row[0] if row else None


def create_ticket_bundle(
    db_path: str | Path,
    request: dict[str, Any],
    classification: dict[str, Any],
    article: dict[str, Any] | None,
    match_score: float,
    retrieval_diagnostics: dict[str, Any],
    routing: dict[str, str],
    actor_id: str,
    batch_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    ticket_id = new_id("ONB")
    created_at = utc_now()
    first_response_due, resolution_due = deadline_pair(datetime.fromisoformat(created_at), classification["priority"], config)
    # Auto-resolution is finalized only after the customer response is successfully dispatched.
    initial_status = "IN_PROGRESS" if routing["route"] == "AUTO_RESOLVE_APPROVED_KB" else routing["status"]
    proposed_response = str(article.get("response", "")) if article else ""
    response_version = 1
    with _connection(db_path) as conn:
        with _transaction(conn):
            actor = conn.execute("SELECT role, active FROM users WHERE user_id=?", (actor_id,)).fetchone()
            if not actor or not actor["active"] or not can(actor["role"], "VIEW_TICKETS"):
                raise AuthorizationError("The importing identity is no longer authorized.")
            if conn.execute("SELECT 1 FROM requests WHERE request_id=?", (request["request_id"],)).fetchone():
                raise DuplicateRequestError(f"Request ID {request['request_id']} already exists in the database.")
            conn.execute(
                """INSERT INTO requests(request_id,customer_id,company_name,contact_email,request_text,source,
                   request_created_at,imported_at,import_batch_id,raw_json) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (request["request_id"], request["customer_id"], request["company_name"], request["contact_email"],
                 request["request_text"], request.get("source", "upload"), request["created_at"], created_at,
                 batch_id, json.dumps(request, ensure_ascii=False)),
            )
            conn.execute(
                """INSERT INTO tickets(ticket_id,request_id,intent,complexity,priority,classifier_source,classifier_confidence,
                   article_id,article_title,match_score,retrieval_diagnostics_json,proposed_response,approved_response,response_version,
                   route,assigned_team,status,created_at,updated_at,first_response_due,resolution_due,first_response_at,resolved_at,
                   sla_state,reopen_count,version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,'WITHIN_SLA',0,1)""",
                (ticket_id, request["request_id"], classification["intent"], classification["complexity"], classification["priority"],
                 classification.get("classifier_source", "RULES"), float(classification.get("classifier_confidence", 0.5)),
                 article.get("article_id") if article else None, article.get("title") if article else None, float(match_score),
                 json.dumps(retrieval_diagnostics, ensure_ascii=False), proposed_response, "", response_version, routing["route"],
                 routing["assigned_team"], initial_status, created_at, created_at, first_response_due, resolution_due),
            )
            _ticket_event(conn, ticket_id, actor_id, actor["role"], "REQUEST_PROCESSED", None, initial_status,
                          f"Intent={classification['intent']}; complexity={classification['complexity']}; priority={classification['priority']}; route={routing['route']}; score={match_score}.", "", proposed_response, 1)
            _audit(conn, actor_id, actor["role"], "REQUEST_CLASSIFIED", request["request_id"], ticket_id,
                   classification["intent"], classification["complexity"], classification["priority"],
                   article.get("article_id") if article else None, float(match_score), route=routing["route"],
                   ticket_status=initial_status, details=f"Classifier={classification.get('classifier_source')}; confidence={classification.get('classifier_confidence')}; retrieval={retrieval_diagnostics.get('method')}.")
            _audit(conn, actor_id, actor["role"], "TICKET_CREATED", request["request_id"], ticket_id,
                   classification["intent"], classification["complexity"], classification["priority"],
                   article.get("article_id") if article else None, float(match_score), route=routing["route"],
                   ticket_status=initial_status, details=f"Assigned to {routing['assigned_team']}; batch {batch_id}.")

            customer = request["contact_email"]
            _insert_notification(conn, ticket_id, f"{ticket_id}:customer_receipt:v1", customer, "EMAIL", "CUSTOMER_RECEIPT",
                                 f"Onboarding request received: {ticket_id}",
                                 f"We have received your onboarding request {request['request_id']}. Tracking ticket: {ticket_id}. Our workflow has routed it to {routing['assigned_team']}. This is an automated acknowledgement, not a confirmation that the issue is resolved.",
                                 "QUEUED", False)
            _insert_notification(conn, ticket_id, f"{ticket_id}:team_assignment:v1", routing["assigned_team"], "IN_APP", "TEAM_ASSIGNMENT",
                                 f"New onboarding case {ticket_id} ({classification['priority']})",
                                 f"A case is assigned to {routing['assigned_team']}. Intent: {classification['intent']}; complexity: {classification['complexity']}; priority: {classification['priority']}. Open the ticket dashboard for sanitized request context.",
                                 "QUEUED", False)
            if proposed_response:
                auto_route = routing["route"] == "AUTO_RESOLVE_APPROVED_KB"
                _insert_notification(conn, ticket_id, f"{ticket_id}:customer_response:v{response_version}", customer, "EMAIL", "CUSTOMER_RESPONSE",
                                     f"Onboarding guidance for ticket {ticket_id}", proposed_response,
                                     "QUEUED" if auto_route else "DRAFT", not auto_route)
            return {
                "ticket_id": ticket_id,
                "request_id": request["request_id"],
                "status": initial_status,
                "notification_ids": [row[0] for row in conn.execute("SELECT notification_id FROM notifications WHERE ticket_id=? AND status='QUEUED'", (ticket_id,)).fetchall()],
                "route": routing["route"],
            }


def set_ticket_processing_seconds(
    db_path: str | Path,
    ticket_id: str,
    processing_seconds: float,
) -> None:
    """Persist the measured automated processing duration for a newly created ticket.

    The value is measured with time.perf_counter() in core.workflow. This update does
    not change the ticket's business status, updated_at timestamp, or optimistic-lock
    version because it records a metric rather than a reviewer action.
    """
    import math

    try:
        duration = float(processing_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("Processing duration must be a finite non-negative number.") from exc
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Processing duration must be a finite non-negative number.")

    with _connection(db_path) as conn:
        with _transaction(conn):
            cursor = conn.execute(
                "UPDATE tickets SET processing_seconds=? WHERE ticket_id=?",
                (duration, ticket_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Cannot record processing duration: ticket {ticket_id} was not found.")


def _ticket_event(
    conn: sqlite3.Connection, ticket_id: str, actor_id: str, actor_role: str, action: str,
    from_status: str | None, to_status: str | None, note: str = "", response_before: str = "",
    response_after: str = "", ticket_version: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO ticket_events(event_id,ticket_id,actor_id,actor_role,action,from_status,to_status,note,response_before,response_after,ticket_version,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (new_id("TEV"), ticket_id, actor_id, actor_role, action, from_status, to_status, note,
         response_before, response_after, ticket_version, utc_now()),
    )


TRANSITIONS: dict[str, set[str]] = {
    "OPEN": {"ASSIGNED", "IN_PROGRESS", "PENDING_REVIEW", "ON_HOLD", "AWAITING_CUSTOMER", "ESCALATED", "RESOLVED"},
    "ASSIGNED": {"IN_PROGRESS", "PENDING_REVIEW", "ON_HOLD", "AWAITING_CUSTOMER", "ESCALATED", "RESOLVED"},
    "IN_PROGRESS": {"ASSIGNED", "PENDING_REVIEW", "ON_HOLD", "AWAITING_CUSTOMER", "ESCALATED", "RESOLVED"},
    "PENDING_REVIEW": {"APPROVED_FOR_RESPONSE", "ASSIGNED", "IN_PROGRESS", "ON_HOLD", "AWAITING_CUSTOMER", "ESCALATED", "RESOLVED"},
    "ON_HOLD": {"ASSIGNED", "IN_PROGRESS", "PENDING_REVIEW", "AWAITING_CUSTOMER", "ESCALATED", "RESOLVED"},
    "AWAITING_CUSTOMER": {"ASSIGNED", "IN_PROGRESS", "PENDING_REVIEW", "ESCALATED", "RESOLVED", "CLOSED"},
    "ESCALATED": {"ASSIGNED", "IN_PROGRESS", "PENDING_REVIEW", "ON_HOLD", "AWAITING_CUSTOMER", "RESOLVED"},
    "APPROVED_FOR_RESPONSE": {"ASSIGNED", "IN_PROGRESS", "PENDING_REVIEW", "ESCALATED", "RESOLVED", "CLOSED"},
    "RESOLVED": {"CLOSED", "IN_PROGRESS"},
    "CLOSED": {"IN_PROGRESS"},
}
ACTION_PERMISSIONS = {
    "APPROVE": "APPROVE", "EDIT": "EDIT", "HOLD": "HOLD",
    "REQUEST_INFORMATION": "REQUEST_INFORMATION", "ESCALATE": "ESCALATE",
    "RESOLVE": "RESOLVE", "REOPEN": "REOPEN", "ASSIGN": "ASSIGN",
    "START_WORK": "ASSIGN",
}


def apply_ticket_action(
    db_path: str | Path,
    ticket_id: str,
    reviewer_id: str,
    action: str,
    *,
    expected_version: int,
    note: str = "",
    edited_response: str | None = None,
    assigned_team: str | None = None,
    pause_sla_when_on_hold: bool = False,
) -> dict[str, Any]:
    action = action.upper().strip()
    permission = ACTION_PERMISSIONS.get(action)
    if not permission:
        raise ValueError("Unsupported ticket action.")
    if action == "EDIT" and not (edited_response or "").strip():
        raise ValueError("An edited response cannot be empty.")
    if action == "RESOLVE" and not (note or "").strip():
        raise ValueError("A resolution note is required to resolve a ticket.")
    if action == "ASSIGN" and not (assigned_team or "").strip():
        raise ValueError("Select a team before assigning the ticket.")

    generated_notifications: list[str] = []
    try:
        with _connection(db_path) as conn:
            with _transaction(conn):
                reviewer = conn.execute("SELECT user_id, username, role, active FROM users WHERE user_id=?", (reviewer_id,)).fetchone()
                if not reviewer or not reviewer["active"] or not can(reviewer["role"], permission):
                    _audit(conn, reviewer_id, reviewer["role"] if reviewer else "UNKNOWN", "UNAUTHORIZED_ACTION_DENIED",
                           ticket_id=ticket_id, reviewer_action=action, details="Ticket action denied by server-side authorization.", error_details="Active identity lacks required permission.")
                    raise AuthorizationError(f"Your authenticated role is not allowed to perform {action}.")
                ticket = conn.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
                if not ticket:
                    raise ValueError("Ticket not found.")
                if int(ticket["version"]) != int(expected_version):
                    _audit(conn, reviewer_id, reviewer["role"], "STALE_UPDATE_REJECTED", ticket_id=ticket_id,
                           reviewer_action=action, ticket_status=ticket["status"], details=f"Expected version {expected_version}; actual version {ticket['version']}.")
                    raise StaleTicketError("This ticket changed after the page was loaded. Refresh the ticket and try again.")
                current_status = ticket["status"]
                new_status = current_status
                previous_response = ticket["proposed_response"] or ""
                next_response = previous_response
                team = ticket["assigned_team"]
                resolved_at = ticket["resolved_at"]
                reopen_count = int(ticket["reopen_count"])
                sla_paused_at = ticket["sla_paused_at"]
                sla_pause_seconds = int(ticket["sla_pause_seconds"] or 0)
                first_response_due = ticket["first_response_due"]
                resolution_due = ticket["resolution_due"]

                if action == "APPROVE":
                    if current_status != "PENDING_REVIEW":
                        raise InvalidTransitionError("Only tickets in PENDING_REVIEW can have a response approved.")
                    if not previous_response.strip():
                        raise ValueError("There is no proposed knowledge-base response to approve.")
                    new_status = "APPROVED_FOR_RESPONSE"
                    next_response = previous_response
                    conn.execute("UPDATE tickets SET approved_response=? WHERE ticket_id=?", (next_response, ticket_id))
                    for row in conn.execute("SELECT notification_id FROM notifications WHERE ticket_id=? AND event_type='CUSTOMER_RESPONSE' AND status='DRAFT'", (ticket_id,)).fetchall():
                        conn.execute("UPDATE notifications SET status='QUEUED', requires_approval=0, approved_by=?, updated_at=? WHERE notification_id=?", (reviewer_id, utc_now(), row[0]))
                        generated_notifications.append(row[0])
                elif action == "EDIT":
                    if current_status not in {"PENDING_REVIEW", "IN_PROGRESS", "APPROVED_FOR_RESPONSE", "ESCALATED"}:
                        raise InvalidTransitionError(f"Responses cannot be edited while ticket status is {current_status}.")
                    next_response = edited_response.strip() if edited_response else previous_response
                    new_status = "PENDING_REVIEW"
                    new_version = int(ticket["response_version"]) + 1
                    conn.execute("UPDATE tickets SET proposed_response=?, approved_response='', response_version=? WHERE ticket_id=?", (next_response, new_version, ticket_id))
                    update_notif = conn.execute("UPDATE notifications SET status='DRAFT', message=?, requires_approval=1, approved_by=NULL, updated_at=? WHERE ticket_id=? AND event_type='CUSTOMER_RESPONSE' AND status IN ('DRAFT','QUEUED')", (next_response, utc_now(), ticket_id))
                    if update_notif.rowcount == 0:
                        request_row = conn.execute("SELECT contact_email FROM requests WHERE request_id=?", (ticket["request_id"],)).fetchone()
                        new_notif = _insert_notification(conn, ticket_id, f"{ticket_id}:customer_response:v{new_version}",
                                                         request_row["contact_email"], "EMAIL", "CUSTOMER_RESPONSE",
                                                         f"Revised onboarding guidance for ticket {ticket_id}", next_response,
                                                         "DRAFT", True)
                        if new_notif:
                            generated_notifications.append(new_notif)
                elif action == "HOLD":
                    if current_status in {"CLOSED", "RESOLVED"}:
                        raise InvalidTransitionError(f"Cannot put a {current_status} ticket on hold.")
                    new_status = "ON_HOLD"
                    if pause_sla_when_on_hold and not sla_paused_at:
                        sla_paused_at = utc_now()
                elif action == "REQUEST_INFORMATION":
                    if current_status in {"CLOSED", "RESOLVED"}:
                        raise InvalidTransitionError(f"Cannot request information on a {current_status} ticket.")
                    if not note.strip():
                        raise ValueError("Explain what information is needed from the customer.")
                    new_status = "AWAITING_CUSTOMER"
                    row = conn.execute("SELECT contact_email FROM requests WHERE request_id=?", (ticket["request_id"],)).fetchone()
                    notif_id = _insert_notification(conn, ticket_id, f"{ticket_id}:information_request:{int(ticket['version']) + 1}",
                                                   row["contact_email"], "EMAIL", "CUSTOMER_INFORMATION_REQUEST",
                                                   f"Additional information needed for {ticket_id}", note.strip(), "QUEUED", False, reviewer_id)
                    if notif_id:
                        generated_notifications.append(notif_id)
                elif action == "ESCALATE":
                    if current_status in {"CLOSED", "RESOLVED"}:
                        raise InvalidTransitionError(f"Reopen the {current_status} ticket before escalating.")
                    new_status = "ESCALATED"
                    team = assigned_team.strip() if assigned_team and assigned_team.strip() else ticket["assigned_team"]
                    notif_id = _insert_notification(conn, ticket_id, f"{ticket_id}:escalated:{int(ticket['version']) + 1}", team, "IN_APP", "TEAM_ESCALATION",
                                                   f"Escalated onboarding case {ticket_id}",
                                                   f"Ticket {ticket_id} was escalated by {reviewer['username']}. Priority: {ticket['priority']}. Note: {note.strip() or 'Review required.'}",
                                                   "QUEUED", False, reviewer_id)
                    if notif_id:
                        generated_notifications.append(notif_id)
                elif action == "RESOLVE":
                    if current_status == "CLOSED":
                        raise InvalidTransitionError("A closed ticket must be reopened before it can be resolved again.")
                    if "RESOLVED" not in TRANSITIONS.get(current_status, set()):
                        raise InvalidTransitionError(f"Cannot resolve a ticket from {current_status}.")
                    new_status = "RESOLVED"
                    resolved_at = utc_now()
                    next_response = note.strip()
                    conn.execute("UPDATE tickets SET approved_response=? WHERE ticket_id=?", (next_response, ticket_id))
                    request_row = conn.execute("SELECT contact_email FROM requests WHERE request_id=?", (ticket["request_id"],)).fetchone()
                    notif_id = _insert_notification(conn, ticket_id, f"{ticket_id}:resolution_notice:{int(ticket['version']) + 1}",
                                                   request_row["contact_email"], "EMAIL", "CUSTOMER_RESPONSE",
                                                   f"Resolution update for {ticket_id}", next_response, "QUEUED", False, reviewer_id)
                    if notif_id:
                        generated_notifications.append(notif_id)
                elif action == "REOPEN":
                    if current_status not in {"RESOLVED", "CLOSED"}:
                        raise InvalidTransitionError("Only a resolved or closed ticket can be reopened.")
                    new_status = "IN_PROGRESS"
                    resolved_at = None
                    reopen_count += 1
                elif action == "ASSIGN":
                    if current_status in {"CLOSED", "RESOLVED"}:
                        raise InvalidTransitionError("Reopen the ticket before reassigning it.")
                    team = assigned_team.strip()
                    new_status = "ASSIGNED"
                    notif_id = _insert_notification(conn, ticket_id, f"{ticket_id}:team_assignment:v{int(ticket['version']) + 1}",
                                                    team, "IN_APP", "TEAM_ASSIGNMENT",
                                                    f"Onboarding case {ticket_id} assigned to {team}",
                                                    f"Ticket {ticket_id} was assigned to {team} by {reviewer['username']}. Priority: {ticket['priority']}.",
                                                    "QUEUED", False, reviewer_id)
                    if notif_id:
                        generated_notifications.append(notif_id)
                elif action == "START_WORK":
                    if "IN_PROGRESS" not in TRANSITIONS.get(current_status, set()):
                        raise InvalidTransitionError(f"Cannot start work from {current_status}.")
                    new_status = "IN_PROGRESS"

                if new_status != current_status and new_status not in TRANSITIONS.get(current_status, set()):
                    raise InvalidTransitionError(f"Transition {current_status} → {new_status} is not allowed.")

                if current_status == "ON_HOLD" and new_status != "ON_HOLD" and sla_paused_at:
                    from datetime import timedelta
                    paused_start = datetime.fromisoformat(sla_paused_at.replace("Z", "+00:00"))
                    if paused_start.tzinfo is None:
                        paused_start = paused_start.replace(tzinfo=timezone.utc)
                    pause_duration = max(0, int((datetime.now(timezone.utc) - paused_start.astimezone(timezone.utc)).total_seconds()))
                    sla_pause_seconds += pause_duration
                    for field_name in ("first_response_due", "resolution_due"):
                        old_due = datetime.fromisoformat((first_response_due if field_name == "first_response_due" else resolution_due).replace("Z", "+00:00"))
                        if old_due.tzinfo is None:
                            old_due = old_due.replace(tzinfo=timezone.utc)
                        new_due = (old_due + timedelta(seconds=pause_duration)).astimezone(timezone.utc).isoformat()
                        if field_name == "first_response_due": first_response_due = new_due
                        else: resolution_due = new_due
                    sla_paused_at = None
                elif new_status != "ON_HOLD":
                    sla_paused_at = None

                version = int(ticket["version"]) + 1
                conn.execute(
                    "UPDATE tickets SET status=?, assigned_team=?, proposed_response=?, resolved_at=?, reopen_count=?, updated_at=?, version=?, sla_state=?, first_response_due=?, resolution_due=?, sla_paused_at=?, sla_pause_seconds=? WHERE ticket_id=?",
                    (new_status, team, next_response, resolved_at, reopen_count, utc_now(), version,
                     "PAUSED" if new_status == "ON_HOLD" and pause_sla_when_on_hold else "WITHIN_SLA",
                     first_response_due, resolution_due, sla_paused_at, sla_pause_seconds, ticket_id),
                )
                _ticket_event(conn, ticket_id, reviewer_id, reviewer["role"], action, current_status, new_status,
                              note.strip(), previous_response, next_response, version)
                _audit(conn, reviewer_id, reviewer["role"], "TICKET_ACTION", ticket["request_id"], ticket_id,
                       ticket["intent"], ticket["complexity"], ticket["priority"], ticket["article_id"], ticket["match_score"],
                       route=ticket["route"], reviewer_action=action, ticket_status=new_status,
                       details=note.strip() or f"Assigned team: {team}",
                       notification_status="QUEUED" if generated_notifications else None)
                return {"ticket_id": ticket_id, "status": new_status, "version": version, "notification_ids": generated_notifications}

    except AuthorizationError:
        # The rejected action's transaction is rolled back; write the security event separately.
        try:
            log_audit(db_path, reviewer_id, "UNAUTHORIZED_ACTION_DENIED", ticket_id=ticket_id,
                      details="Ticket action denied by server-side authorization.",
                      error_details="Active identity lacks the required permission.", reviewer_action=action)
        except Exception:
            pass
        raise
    except StaleTicketError:
        try:
            log_audit(db_path, reviewer_id, "STALE_UPDATE_REJECTED", ticket_id=ticket_id,
                      details="Ticket action rejected because the expected version was stale.",
                      error_details=f"Expected version {expected_version} did not match the current version.", reviewer_action=action)
        except Exception:
            pass
        raise


def list_tickets(db_path: str | Path, *, status: str | None = None, priority: str | None = None, intent: str | None = None) -> list[dict[str, Any]]:
    sql = """SELECT t.*, r.customer_id, r.company_name, r.contact_email, r.request_text, r.request_created_at, r.source
             FROM tickets t JOIN requests r ON r.request_id=t.request_id WHERE 1=1"""
    params: list[Any] = []
    if status and status != "All":
        sql += " AND t.status=?"; params.append(status)
    if priority and priority != "All":
        sql += " AND t.priority=?"; params.append(priority)
    if intent and intent != "All":
        sql += " AND t.intent=?"; params.append(intent)
    sql += " ORDER BY CASE t.priority WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 WHEN 'P3' THEN 3 ELSE 4 END, t.created_at DESC"
    with _connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_ticket(db_path: str | Path, ticket_id: str) -> dict[str, Any] | None:
    with _connection(db_path) as conn:
        row = conn.execute("SELECT t.*, r.customer_id, r.company_name, r.contact_email, r.request_text, r.request_created_at, r.source FROM tickets t JOIN requests r ON r.request_id=t.request_id WHERE t.ticket_id=?", (ticket_id,)).fetchone()
    return _dict_row(row)


def get_ticket_history(db_path: str | Path, ticket_id: str) -> list[dict[str, Any]]:
    with _connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM ticket_events WHERE ticket_id=? ORDER BY created_at DESC", (ticket_id,)).fetchall()
    return [dict(row) for row in rows]


def list_notifications(db_path: str | Path, status: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT n.*, t.request_id, t.priority, t.status AS ticket_status FROM notifications n JOIN tickets t ON t.ticket_id=n.ticket_id"
    params: list[Any] = []
    if status and status != "All":
        sql += " WHERE n.status=?"; params.append(status)
    sql += " ORDER BY n.created_at DESC"
    with _connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_notification(db_path: str | Path, notification_id: str) -> dict[str, Any] | None:
    with _connection(db_path) as conn:
        row = conn.execute("SELECT * FROM notifications WHERE notification_id=?", (notification_id,)).fetchone()
    return _dict_row(row)


def list_notification_attempts(db_path: str | Path, notification_id: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT a.*, n.ticket_id, n.event_type, n.recipient FROM notification_attempts a JOIN notifications n ON n.notification_id=a.notification_id"
    params: list[Any] = []
    if notification_id:
        sql += " WHERE a.notification_id=?"; params.append(notification_id)
    sql += " ORDER BY a.attempted_at DESC"
    with _connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def append_notification_attempt(
    db_path: str | Path, notification_id: str, cycle_number: int, attempt_number: int,
    result: str, error_message: str = "", actor_id: str = "SYSTEM",
) -> None:
    with _connection(db_path) as conn:
        with _transaction(conn):
            notif = conn.execute("SELECT n.*, t.request_id FROM notifications n JOIN tickets t ON t.ticket_id=n.ticket_id WHERE n.notification_id=?", (notification_id,)).fetchone()
            if not notif:
                raise ValueError("Notification not found.")
            conn.execute(
                "INSERT INTO notification_attempts(attempt_id,notification_id,cycle_number,attempt_number,result,error_message,attempted_at) VALUES(?,?,?,?,?,?,?)",
                (new_id("ATT"), notification_id, cycle_number, attempt_number, result, error_message, utc_now()),
            )
            conn.execute(
                "UPDATE notifications SET attempts=attempts+1, last_error=?, updated_at=? WHERE notification_id=?",
                (error_message, utc_now(), notification_id),
            )
            role = "SYSTEM"
            if actor_id != "SYSTEM":
                user = conn.execute("SELECT role,active FROM users WHERE user_id=?", (actor_id,)).fetchone()
                if not user or not user["active"] or not can(user["role"], "SEND_NOTIFICATION"):
                    raise AuthorizationError("This identity cannot dispatch notifications.")
                role = user["role"]
            _audit(conn, actor_id, role, "NOTIFICATION_ATTEMPT", notif["request_id"], notif["ticket_id"],
                   notification_status=result, details=f"Notification={notification_id}; cycle={cycle_number}; attempt={attempt_number}.",
                   error_details=error_message)


def start_notification_cycle(db_path: str | Path, notification_id: str, actor_id: str = "SYSTEM") -> dict[str, Any]:
    if actor_id != "SYSTEM":
        identity = get_user_by_id(db_path, actor_id)
        if not identity or not identity.get("active") or not can(identity.get("role", ""), "SEND_NOTIFICATION"):
            try:
                log_audit(db_path, actor_id, "UNAUTHORIZED_NOTIFICATION_DISPATCH_DENIED", ticket_id=None,
                          details=f"Unauthorized dispatch attempt for notification {notification_id}.",
                          error_details="Active identity lacks SEND_NOTIFICATION permission.")
            finally:
                raise AuthorizationError("Your authenticated role cannot dispatch notifications.")
    try:
        with _connection(db_path) as conn:
            with _transaction(conn):
                row = conn.execute("SELECT * FROM notifications WHERE notification_id=?", (notification_id,)).fetchone()
                if not row:
                    raise ValueError("Notification not found.")
                if row["status"] == "SENT":
                    return {"already_sent": True, "cycle_number": row["dispatch_cycles"], "attempts": row["attempts"]}
                if row["status"] == "DEAD_LETTER":
                    raise ValueError("This notification is in the dead-letter queue and cannot be retried automatically. Investigate it before any controlled redrive.")
                if row["requires_approval"] and not row["approved_by"]:
                    if actor_id != "SYSTEM":
                        user = conn.execute("SELECT role,active FROM users WHERE user_id=?", (actor_id,)).fetchone()
                        if not user or not user["active"] or not can(user["role"], "SEND_NOTIFICATION"):
                            raise AuthorizationError("Your identity cannot dispatch this notification.")
                    _audit(conn, actor_id, "SYSTEM" if actor_id == "SYSTEM" else user["role"], "UNAUTHORIZED_NOTIFICATION_DISPATCH_DENIED",
                           ticket_id=row["ticket_id"], notification_status=row["status"],
                           details=f"Attempted to dispatch unapproved notification {notification_id}.", error_details="Human approval is required.")
                    raise AuthorizationError("This customer-facing response requires approval before dispatch.")
                if row["status"] == "DRAFT":
                    raise ValueError("Draft notification has not been authorized for dispatch.")
                cycle = int(row["dispatch_cycles"]) + 1
                conn.execute("UPDATE notifications SET dispatch_cycles=?, updated_at=? WHERE notification_id=?", (cycle, utc_now(), notification_id))
                return {"already_sent": False, "cycle_number": cycle, "attempts": int(row["attempts"])}

    except AuthorizationError:
        # Record the guard decision in a separate transaction because the blocked dispatch rolled back.
        try:
            notification = get_notification(db_path, notification_id)
            log_audit(db_path, actor_id, "NOTIFICATION_APPROVAL_GUARD_BLOCKED",
                      ticket_id=notification.get("ticket_id") if notification else None,
                      details=f"Dispatch blocked for notification {notification_id}.",
                      error_details="The response is not approved or the caller is no longer authorized.")
        except Exception:
            pass
        raise


def finish_notification_cycle(
    db_path: str | Path, notification_id: str, success: bool, last_error: str = "",
    actor_id: str = "SYSTEM", dead_letter_after_cycles: int = 2,
) -> dict[str, Any]:
    with _connection(db_path) as conn:
        with _transaction(conn):
            row = conn.execute("SELECT n.*, t.request_id, t.route, t.status AS ticket_status, t.version AS ticket_version, t.assigned_team, t.proposed_response, t.resolved_at FROM notifications n JOIN tickets t ON t.ticket_id=n.ticket_id WHERE n.notification_id=?", (notification_id,)).fetchone()
            if not row:
                raise ValueError("Notification not found.")
            if row["status"] == "SENT":
                return {"status": "SENT", "ticket_id": row["ticket_id"], "ticket_status": row["ticket_status"], "already_sent": True}
            status = "SENT" if success else ("DEAD_LETTER" if int(row["dispatch_cycles"]) >= dead_letter_after_cycles else "FAILED")
            sent_at = utc_now() if success else None
            conn.execute("UPDATE notifications SET status=?, sent_at=COALESCE(?,sent_at), last_error=?, updated_at=? WHERE notification_id=?",
                         (status, sent_at, last_error, utc_now(), notification_id))
            role = "SYSTEM"
            if actor_id != "SYSTEM":
                user = conn.execute("SELECT role FROM users WHERE user_id=?", (actor_id,)).fetchone()
                role = user["role"] if user else "UNKNOWN"
            _audit(conn, actor_id, role, "NOTIFICATION_SENT" if success else "NOTIFICATION_FAILED",
                   row["request_id"], row["ticket_id"], notification_status=status,
                   ticket_status=row["ticket_status"], details=f"Notification={notification_id}; event={row['event_type']}; channel={row['channel']}.",
                   error_details=last_error)
            # Track a meaningful customer reply, not an internal assignment or receipt acknowledgement.
            if success and row["event_type"] in {"CUSTOMER_RESPONSE", "CUSTOMER_INFORMATION_REQUEST"}:
                conn.execute("UPDATE tickets SET first_response_at=COALESCE(first_response_at,?), updated_at=? WHERE ticket_id=?", (sent_at, sent_at, row["ticket_id"]))
                if row["route"] == "AUTO_RESOLVE_APPROVED_KB" and row["ticket_status"] == "IN_PROGRESS":
                    new_version = int(row["ticket_version"]) + 1
                    conn.execute("UPDATE tickets SET status='RESOLVED', resolved_at=?, updated_at=?, version=?, sla_state='WITHIN_SLA' WHERE ticket_id=?",
                                 (sent_at, sent_at, new_version, row["ticket_id"]))
                    _ticket_event(conn, row["ticket_id"], "SYSTEM", "SYSTEM", "AUTO_RESOLVE_AFTER_DELIVERY", "IN_PROGRESS", "RESOLVED",
                                  "Approved knowledge-base response was successfully simulated as delivered.", row["proposed_response"] or "", row["proposed_response"] or "", new_version)
                    _audit(conn, "SYSTEM", "SYSTEM", "AUTO_RESOLUTION_COMPLETED", row["request_id"], row["ticket_id"],
                           route=row["route"], ticket_status="RESOLVED", notification_status="SENT",
                           details="Auto-resolution completed only after the customer response delivery simulation succeeded.")
            return {"status": status, "ticket_id": row["ticket_id"], "ticket_status": row["ticket_status"], "already_sent": False}



def create_sla_alert_notification(db_path: str | Path, ticket_id: str, sla_state: str) -> str | None:
    if sla_state not in {"AT_RISK", "BREACHED"}:
        return None
    with _connection(db_path) as conn:
        with _transaction(conn):
            ticket = conn.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
            if not ticket:
                return None
            event = "SLA_AT_RISK" if sla_state == "AT_RISK" else "SLA_BREACH"
            subject = f"{event.replace('_', ' ').title()}: {ticket_id}"
            message = (f"Ticket {ticket_id} is {sla_state.lower().replace('_', ' ')}. Priority: {ticket['priority']}; "
                       f"team: {ticket['assigned_team']}; current status: {ticket['status']}. Review the case and record any authorized SLA pause or action.")
            notification_id = _insert_notification(
                conn, ticket_id, f"{ticket_id}:{event.lower()}:v1", "Support Lead", "IN_APP", event,
                subject, message, "QUEUED", False, "SYSTEM",
            )
            _audit(conn, "SYSTEM", "SYSTEM", "SLA_ALERT_CREATED", ticket["request_id"], ticket_id,
                   ticket["intent"], ticket["complexity"], ticket["priority"], ticket["article_id"], ticket["match_score"],
                   route=ticket["route"], ticket_status=ticket["status"], notification_status="QUEUED",
                   details=f"Generated {event} notification for SLA state {sla_state}.")
            return notification_id


def update_ticket_sla(db_path: str | Path, ticket_id: str, sla_state: str) -> bool:
    with _connection(db_path) as conn:
        with _transaction(conn):
            row = conn.execute("SELECT sla_state,request_id,status,priority,intent,complexity FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
            if not row or row["sla_state"] == sla_state:
                return False
            conn.execute("UPDATE tickets SET sla_state=?, updated_at=? WHERE ticket_id=?", (sla_state, utc_now(), ticket_id))
            _audit(conn, "SYSTEM", "SYSTEM", "SLA_STATE_CHANGED", row["request_id"], ticket_id,
                   row["intent"], row["complexity"], row["priority"], ticket_status=row["status"],
                   details=f"SLA state changed from {row['sla_state']} to {sla_state}.")
            return True


def list_import_batches(db_path: str | Path) -> list[dict[str, Any]]:
    with _connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM import_batches ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def list_rejected_records(db_path: str | Path, batch_id: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM rejected_records"
    params: list[Any] = []
    if batch_id:
        sql += " WHERE batch_id=?"; params.append(batch_id)
    sql += " ORDER BY created_at DESC"
    with _connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    output = []
    for row in rows:
        item = dict(row)
        try: item["raw_record"] = json.loads(item.pop("raw_record_json"))
        except json.JSONDecodeError: item["raw_record"] = {}
        try: item["errors"] = json.loads(item.pop("errors_json"))
        except json.JSONDecodeError: item["errors"] = ["Could not parse stored validation errors"]
        output.append(item)
    return output


def list_audit_events(db_path: str | Path, limit: int = 2000) -> list[dict[str, Any]]:
    with _connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM audit_events ORDER BY timestamp DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(row) for row in rows]


def list_ticket_events(db_path: str | Path, limit: int = 3000) -> list[dict[str, Any]]:
    with _connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM ticket_events ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(row) for row in rows]


def list_requests(db_path: str | Path) -> list[dict[str, Any]]:
    with _connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM requests ORDER BY imported_at DESC").fetchall()
    return [dict(row) for row in rows]



def create_test_notification(db_path: str | Path, actor_id: str, ticket_id: str, recipient: str | None = None) -> str:
    """Create a non-customer-facing mock event for testing retry behaviour."""
    with _connection(db_path) as conn:
        with _transaction(conn):
            actor = conn.execute("SELECT role,active,username FROM users WHERE user_id=?", (actor_id,)).fetchone()
            if not actor or not actor["active"] or not can(actor["role"], "RUN_TESTS"):
                raise AuthorizationError("Only an ADMIN or MANAGER can run notification simulation tests.")
            ticket = conn.execute("SELECT ticket_id,request_id,priority FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
            if not ticket:
                raise ValueError("Choose an existing ticket for the notification test.")
            notification_id = _insert_notification(
                conn, ticket_id, f"{ticket_id}:test:{uuid.uuid4().hex}", recipient or "OnboardIQ simulation inbox",
                "IN_APP", "TEST_NOTIFICATION", f"Delivery simulation test for {ticket_id}",
                "This is a simulated notification used to test retry handling. No external message is sent.",
                "QUEUED", False, actor_id,
            )
            _audit(conn, actor_id, actor["role"], "TEST_NOTIFICATION_CREATED", ticket["request_id"], ticket_id,
                   priority=ticket["priority"], notification_status="QUEUED",
                   details=f"User {actor['username']} created a mock notification retry test.")
            return notification_id


def list_notifications_summary(db_path: str | Path) -> dict[str, int]:
    with _connection(db_path) as conn:
        rows = conn.execute("SELECT status,COUNT(*) AS n FROM notifications GROUP BY status").fetchall()
    result = {name: 0 for name in ("DRAFT", "QUEUED", "SENT", "FAILED", "DEAD_LETTER")}
    result.update({row["status"]: int(row["n"]) for row in rows})
    return result


def get_counts(db_path: str | Path) -> dict[str, Any]:
    with _connection(db_path) as conn:
        tickets = conn.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN status NOT IN ('RESOLVED','CLOSED') THEN 1 ELSE 0 END) AS active, SUM(CASE WHEN status='PENDING_REVIEW' THEN 1 ELSE 0 END) AS pending_review, SUM(CASE WHEN status='ESCALATED' THEN 1 ELSE 0 END) AS escalated, SUM(CASE WHEN status IN ('RESOLVED','CLOSED') THEN 1 ELSE 0 END) AS resolved, SUM(CASE WHEN priority='P1' THEN 1 ELSE 0 END) AS p1, SUM(CASE WHEN sla_state='BREACHED' THEN 1 ELSE 0 END) AS breached FROM tickets").fetchone()
        batches = conn.execute("SELECT COUNT(*) AS n FROM import_batches").fetchone()[0]
        rejected = conn.execute("SELECT COUNT(*) AS n FROM rejected_records").fetchone()[0]
        users = conn.execute("SELECT COUNT(*) AS n FROM users WHERE active=1").fetchone()[0]
    return {key: int(value or 0) for key, value in dict(tickets).items()} | {"batches": int(batches), "rejected_records": int(rejected), "active_users": int(users)}
