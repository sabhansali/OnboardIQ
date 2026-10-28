from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.classifier import classify_request, classify_with_groq
from core.config import BASE_DIR, get_db_path, load_config, load_knowledge_base
from core.database import (
    AuthorizationError,
    StaleTicketError,
    apply_ticket_action,
    authenticate_user,
    create_import_batch,
    create_test_notification,
    create_user,
    finalize_import_batch,
    get_counts,
    get_notification,
    get_ticket,
    get_ticket_history,
    get_user_by_id,
    init_db,
    list_audit_events,
    list_import_batches,
    list_notification_attempts,
    list_notifications,
    list_rejected_records,
    list_ticket_events,
    list_tickets,
    list_users,
    log_audit,
    record_rejected_row,
    set_user_active,
    TRANSITIONS,
)
from core.ingestion import parse_uploaded_file, validate_records
from core.metrics import compute_metrics, keyword_only_retrieval
from core.notifications import build_notification_preview, dispatch_notification
from core.resolver import retrieve_article
from core.security import can
from core.sla_service import refresh_sla_states
from core.workflow import process_import_batch, process_one_request

st.set_page_config(
    page_title="OnboardIQ | SaaS Onboarding Operations",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = get_db_path()
CONFIG = load_config()
try:
    KNOWLEDGE_BASE = load_knowledge_base()
    KB_LOAD_ERROR = ""
except Exception as kb_error:  # no unapproved fallback is used when the KB is malformed
    KNOWLEDGE_BASE = []
    KB_LOAD_ERROR = f"Knowledge base could not be loaded: {type(kb_error).__name__}: {kb_error}"

init_db(DB_PATH)

st.markdown(
    """
    <style>
      :root { --ink: #172b4d; --muted: #65758b; --line: #e5eaf0; --navy: #183b56; --teal: #167d83; --paper: #f7f9fb; }
      .stApp { background: #f7f9fb; color: var(--ink); }
      [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5eaf0; }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #183b56; }
      h1, h2, h3 { letter-spacing: -0.025em; color: #183b56; }
      .app-eyebrow { text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; color: #167d83; }
      .app-title { font-size: 2.0rem; line-height: 1.12; font-weight: 760; color: #183b56; margin: .2rem 0 .35rem 0; }
      .app-subtitle { color: #65758b; font-size: .98rem; max-width: 900px; }
      .metric-card { background: white; border: 1px solid #e5eaf0; border-radius: 14px; padding: 16px 18px; min-height: 112px; }
      .metric-label { color: #65758b; font-size: .82rem; font-weight: 600; }
      .metric-value { font-size: 1.85rem; color: #183b56; font-weight: 750; margin: .18rem 0; }
      .metric-foot { color: #7b8999; font-size: .76rem; }
      .section-card { background: white; border: 1px solid #e5eaf0; border-radius: 14px; padding: 18px 20px; }
      .small-muted { color: #65758b; font-size: .86rem; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #e5eaf0; border-radius: 12px; padding: 14px 16px; }
      div[data-testid="stMetricLabel"] { color: #65758b; }
      div.stButton > button, div.stFormSubmitButton > button { border-radius: 9px; font-weight: 600; min-height: 2.55rem; }
      .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: #183b56; border-color: #183b56; }
      div[data-testid="stDataFrame"] { border: 1px solid #e5eaf0; border-radius: 10px; overflow: hidden; }
      .status-chip { background: #edf4f5; color: #17656a; border-radius: 999px; padding: .22rem .55rem; font-size: .78rem; font-weight: 650; }
      hr { border-color: #e5eaf0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="app-eyebrow">{eyebrow}</div><div class="app-title">{title}</div><div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def display_error(error: Exception) -> None:
    if isinstance(error, AuthorizationError):
        st.error(str(error))
    elif isinstance(error, StaleTicketError):
        st.warning(str(error))
    else:
        st.error(f"{type(error).__name__}: {error}")


def process_content(filename: str, content: bytes, actor_id: str, use_groq: bool) -> dict[str, Any]:
    try:
        raw_records = parse_uploaded_file(filename, content)
        valid_records, rejected = validate_records(raw_records)
        batch_id = create_import_batch(DB_PATH, filename, len(raw_records), rejected, actor_id)
        result = process_import_batch(DB_PATH, batch_id, filename, valid_records, actor_id, KNOWLEDGE_BASE, CONFIG, use_groq=use_groq)
        refresh_sla_states(DB_PATH, CONFIG)
        return {"batch_id": batch_id, "total": len(raw_records), "schema_valid": len(valid_records),
                "initially_rejected": len(rejected), **result}
    except (ValueError, UnicodeDecodeError) as exc:
        # Malformed files are persisted as rejected batches too, rather than disappearing silently.
        rejected = [{"row_number": None, "record": {"filename": filename}, "errors": [str(exc)]}]
        batch_id = create_import_batch(DB_PATH, filename, 1, rejected, actor_id)
        finalize_import_batch(DB_PATH, batch_id, 0, error_message=str(exc))
        log_audit(DB_PATH, actor_id, "IMPORT_FILE_REJECTED", details=f"File={filename}.", error_details=str(exc))
        return {"batch_id": batch_id, "total": 1, "schema_valid": 0, "initially_rejected": 1, "processed_count": 0, "created": [], "errors": [{"error": str(exc)}]}


def notify_dispatch(notification_id: str, actor_id: str, failure_mode: str = "normal") -> dict[str, Any]:
    return dispatch_notification(
        str(DB_PATH), notification_id, actor_id=actor_id,
        max_attempts=int(CONFIG.get("notification_retry_attempts", 3)),
        failure_mode=failure_mode,
        dead_letter_after_cycles=int(CONFIG.get("notification_dead_letter_cycles", 2)),
    )


def metric_card(label: str, value: str | int, foot: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def df_download(label: str, df: pd.DataFrame, filename: str, key: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, data=csv_bytes, file_name=filename, mime="text/csv", key=key)


def render_login() -> None:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.write("")
        st.write("")
        st.markdown('<div class="app-eyebrow">SAAS IMPLEMENTATION OPERATIONS</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-title" style="font-size:2.65rem">OnboardIQ</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Grounded onboarding assistance, accountable escalation, and human-controlled workflows.</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown("**A controlled workflow, from request to verified outcome**")
        st.write("Intake and validation · Approved knowledge retrieval · Ticket lifecycle · Human review · Notifications · SLA and audit evidence")
        st.info("Local demo accounts are seeded on first run. Change their credentials before using this application beyond a local demonstration.")
        with st.expander("Local demo credentials"):
            st.code("ADMIN       admin / AdminDemo!2026\nMANAGER     manager / ManagerDemo!2026\nSPECIALIST  specialist / SpecialistDemo!2026")
    with right:
        st.write("")
        st.write("")
        with st.container(border=True):
            st.subheader("Sign in")
            st.caption("Your role controls which review and workflow actions are available.")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                user = authenticate_user(DB_PATH, username, password)
                if user:
                    st.session_state["user_id"] = user["user_id"]
                    st.session_state.pop("selected_page", None)
                    log_audit(DB_PATH, user["user_id"], "USER_LOGIN", details="Successful local sign-in.")
                    st.rerun()
                st.error("Sign-in failed. Check the username/password or ask an administrator to reactivate the account.")


def render_overview(user: dict[str, Any]) -> None:
    show_header("OPERATIONS OVERVIEW", "Onboarding control center", "A live view of implementation requests, human review, escalations, delivery outcomes, and SLA exposure.")
    refresh_sla_states(DB_PATH, CONFIG)
    counts = get_counts(DB_PATH)
    cols = st.columns(5)
    with cols[0]: metric_card("Total cases", counts["total"], "All persisted onboarding tickets")
    with cols[1]: metric_card("Active", counts["active"], "Not resolved or closed")
    with cols[2]: metric_card("Awaiting review", counts["pending_review"], "Human decision required")
    with cols[3]: metric_card("Escalated", counts["escalated"], "Specialist / incident queues")
    with cols[4]: metric_card("SLA breached", counts["breached"], "Deadline missed or response overdue")
    st.write("")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("#### Priority and lifecycle")
        tickets = list_tickets(DB_PATH)
        if tickets:
            lifecycle = pd.DataFrame(tickets)
            status_counts = lifecycle.groupby("status", dropna=False).size().reset_index(name="Cases").sort_values("Cases", ascending=False)
            st.bar_chart(status_counts.set_index("status"), height=245)
            priority_counts = lifecycle.groupby("priority").size().reindex(["P1", "P2", "P3", "P4"], fill_value=0).rename("Cases").to_frame()
            st.caption("Priority distribution")
            st.bar_chart(priority_counts, height=180)
        else:
            st.info("No tickets yet. Open **Intake & Processing** and import the sample request dataset.")
    with right:
        st.markdown("#### Needs attention")
        tickets = list_tickets(DB_PATH)
        attention = [t for t in tickets if t["priority"] in {"P1", "P2"} or t["status"] in {"PENDING_REVIEW", "ESCALATED", "ON_HOLD", "AWAITING_CUSTOMER"} or t["sla_state"] in {"AT_RISK", "BREACHED"}]
        if attention:
            view = pd.DataFrame(attention)[["ticket_id", "company_name", "priority", "status", "assigned_team", "sla_state"]].head(12)
            st.dataframe(view, hide_index=True, use_container_width=True)
        else:
            st.success("No high-priority or overdue cases are currently flagged.")
        st.markdown("#### Safe automation policy")
        st.write("P1/security cases bypass automatic resolution. Only low-complexity P3/P4 requests with a sufficiently strong match to an approved article can be auto-resolved. Other responses stay in human review or specialist queues.")
        if KB_LOAD_ERROR:
            st.error(KB_LOAD_ERROR)
        else:
            st.caption(f"Approved knowledge articles loaded: {sum(bool(a.get('approved')) for a in KNOWLEDGE_BASE)}")


def render_intake(user: dict[str, Any], use_groq: bool) -> None:
    show_header("REQUEST INTAKE", "Import or submit onboarding requests", "Ingest CSV/JSON files or simulate a customer chat. Malformed rows are rejected with reasons; duplicate request IDs cannot create duplicate tickets.")
    tab_upload, tab_chat, tab_batches = st.tabs(["File upload", "Simulated chat", "Import history & rejected rows"])
    with tab_upload:
        st.markdown("#### Import a request file")
        st.caption("Required fields: request_id, customer_id, company_name, contact_email, request_text, created_at. Optional fields include source and expected labels for evaluation.")
        uploaded = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"], key="request_upload")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Load included sample dataset", type="primary", use_container_width=True):
                result = process_content("sample_requests.csv", (BASE_DIR / "data" / "sample_requests.csv").read_bytes(), user["user_id"], use_groq)
                st.session_state["last_import_result"] = result
                st.rerun()
        with col_b:
            if uploaded and st.button("Validate and process upload", use_container_width=True):
                result = process_content(uploaded.name, uploaded.getvalue(), user["user_id"], use_groq)
                st.session_state["last_import_result"] = result
                st.rerun()
        result = st.session_state.get("last_import_result")
        if result:
            st.write("")
            st.markdown("#### Last import result")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows read", result.get("total", 0))
            m2.metric("Schema-valid", result.get("schema_valid", 0))
            m3.metric("Tickets created", result.get("processed_count", 0))
            m4.metric("Rejected", result.get("initially_rejected", 0) + len(result.get("errors", [])))
            if result.get("errors"):
                st.warning("Some records could not be processed. Review Import history & rejected rows for the durable error record.")
                st.dataframe(pd.DataFrame(result["errors"]), hide_index=True, use_container_width=True)
            created = result.get("created", [])
            if created:
                st.success(f"Import batch {result['batch_id']} completed. {len(created)} ticket(s) created.")
                summary = [{"request_id": c["request_id"], "ticket_id": c["ticket_id"], "route": c["routing"]["route"],
                            "status": c["status"], "intent": c["classification"]["intent"], "priority": c["classification"]["priority"],
                            "match_score": c["match_score"]} for c in created]
                st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
    with tab_chat:
        st.markdown("#### Simulate a customer chat request")
        with st.form("simulated_chat", clear_on_submit=True):
            company_name = st.text_input("Company name", placeholder="Example: Northstar Labs")
            customer_id = st.text_input("Synthetic customer ID", placeholder="Example: CUST-020")
            contact_email = st.text_input("Contact email", placeholder="ops@example.test")
            request_text = st.text_area("What does the customer need help with?", placeholder="Describe the onboarding or implementation issue.", height=120)
            submitted = st.form_submit_button("Submit chat request", type="primary")
        if submitted:
            now = datetime.now(timezone.utc).isoformat()
            request = {
                "request_id": f"CHAT-{uuid.uuid4().hex[:10].upper()}",
                "customer_id": customer_id.strip(), "company_name": company_name.strip(),
                "contact_email": contact_email.strip(), "request_text": request_text.strip(),
                "created_at": now, "source": "Simulated Chat",
            }
            valid, rejected = validate_records([request])
            if rejected or not valid:
                st.error("The chat request could not be accepted. Check company, customer ID, contact email, and request text.")
                st.json(rejected[0]["errors"] if rejected else ["Invalid request"])
            else:
                batch = create_import_batch(DB_PATH, "simulated-chat", 1, [], user["user_id"])
                try:
                    result = process_one_request(DB_PATH, valid[0], batch, user["user_id"], KNOWLEDGE_BASE, CONFIG, use_groq=use_groq)
                    finalize_import_batch(DB_PATH, batch, 1)
                    refresh_sla_states(DB_PATH, CONFIG)
                    st.success(f"Created ticket {result['ticket_id']} with route {result['routing']['route']}.")
                    st.json({"ticket_id": result["ticket_id"], "classification": result["classification"], "match_score": result["match_score"], "routing": result["routing"]})
                except Exception as exc:
                    finalize_import_batch(DB_PATH, batch, 0, error_message=str(exc))
                    st.error(f"Request could not be processed: {type(exc).__name__}: {exc}")
    with tab_batches:
        batches = list_import_batches(DB_PATH)
        if batches:
            st.dataframe(pd.DataFrame(batches), hide_index=True, use_container_width=True)
        else:
            st.info("Import batches will appear here after the first file or chat submission.")
        st.markdown("#### Rejected records")
        rejected_rows = list_rejected_records(DB_PATH)
        if rejected_rows:
            normalized = [{"batch_id": r["batch_id"], "source_name": r["source_name"], "row_number": r["row_number"],
                           "record": json.dumps(r["raw_record"], ensure_ascii=False), "errors": "; ".join(r["errors"]), "created_at": r["created_at"]}
                          for r in rejected_rows]
            st.dataframe(pd.DataFrame(normalized), hide_index=True, use_container_width=True)
            df_download("Download rejected records", pd.DataFrame(normalized), "rejected_records.csv", "download_rejected")
        else:
            st.caption("No rejected rows recorded yet.")


def allowed_actions_for(ticket: dict[str, Any], role: str) -> list[str]:
    status = ticket["status"]
    possible = TRANSITIONS.get(status, set())
    actions: list[str] = []
    if "APPROVED_FOR_RESPONSE" in possible and can(role, "APPROVE") and ticket.get("proposed_response", "").strip():
        if status == "PENDING_REVIEW": actions.append("APPROVE")
    if (status == "PENDING_REVIEW" or "PENDING_REVIEW" in possible) and status in {"PENDING_REVIEW", "IN_PROGRESS", "APPROVED_FOR_RESPONSE", "ESCALATED"} and can(role, "EDIT"):
        actions.append("EDIT")
    if "ON_HOLD" in possible and can(role, "HOLD"):
        actions.append("HOLD")
    if "AWAITING_CUSTOMER" in possible and can(role, "REQUEST_INFORMATION"):
        actions.append("REQUEST_INFORMATION")
    if "ESCALATED" in possible and can(role, "ESCALATE"):
        actions.append("ESCALATE")
    if "ASSIGNED" in possible and can(role, "ASSIGN"):
        actions.append("ASSIGN")
    if "IN_PROGRESS" in possible and can(role, "ASSIGN"):
        actions.append("START_WORK")
    if "RESOLVED" in possible and can(role, "RESOLVE"):
        actions.append("RESOLVE")
    if status in {"RESOLVED", "CLOSED"} and "IN_PROGRESS" in possible and can(role, "REOPEN"):
        actions.append("REOPEN")
    return actions


def render_ticket_queue(user: dict[str, Any]) -> None:
    show_header("HUMAN-IN-THE-LOOP", "Ticket queue and review workspace", "Review proposed onboarding guidance, assign implementation teams, request information, resolve verified work, or escalate to specialists. Every action is authorized and audited server-side.")
    tickets = list_tickets(DB_PATH)
    if not tickets:
        st.info("No cases yet. Use Intake & Processing to load the included sample dataset.")
        return
    statuses = ["All"] + sorted({t["status"] for t in tickets})
    priorities = ["All", "P1", "P2", "P3", "P4"]
    intents = ["All"] + sorted({t["intent"] for t in tickets})
    f1, f2, f3 = st.columns(3)
    status_filter = f1.selectbox("Status", statuses)
    priority_filter = f2.selectbox("Priority", priorities)
    intent_filter = f3.selectbox("Intent", intents)
    filtered = list_tickets(DB_PATH, status=status_filter, priority=priority_filter, intent=intent_filter)
    table = [{"Ticket": t["ticket_id"], "Company": t["company_name"], "Intent": t["intent"], "Priority": t["priority"],
              "Complexity": t["complexity"], "Status": t["status"], "Assigned team": t["assigned_team"], "KB match": t["match_score"],
              "SLA": t["sla_state"]} for t in filtered]
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    if not filtered:
        return

    st.markdown("#### Ticket details and authorized actions")
    options = {f"{t['ticket_id']} · {t['priority']} · {t['status']} · {t['company_name']}": t["ticket_id"] for t in filtered}
    selected_label = st.selectbox("Select a ticket", list(options.keys()))
    ticket = get_ticket(DB_PATH, options[selected_label])
    if not ticket:
        st.warning("Ticket no longer exists. Refresh the page.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Priority", ticket["priority"])
    col2.metric("KB match score", f"{ticket['match_score']:.1f}%")
    col3.metric("SLA state", ticket["sla_state"])
    col4.metric("Record version", ticket["version"])
    with st.container(border=True):
        st.markdown(f"**{ticket['company_name']}** · {ticket['customer_id']} · {ticket['contact_email']}")
        st.markdown(f"**Customer request**  \n{ticket['request_text']}")
        s1, s2, s3 = st.columns(3)
        s1.write(f"**Intent**  \n{ticket['intent']}")
        s2.write(f"**Complexity**  \n{ticket['complexity']}")
        s3.write(f"**Route**  \n{ticket['route']}")
        st.write(f"**Assigned team:** {ticket['assigned_team']}  ·  **Knowledge article:** {ticket['article_title'] or 'No approved article matched'}")
        if ticket["proposed_response"]:
            st.markdown("**Proposed response from the approved knowledge base**")
            st.write(ticket["proposed_response"])
            if ticket["approved_response"]:
                st.markdown("**Approved response / resolution summary**")
                st.write(ticket["approved_response"])
        try:
            retrieval = json.loads(ticket["retrieval_diagnostics_json"])
            with st.expander("Retrieval evidence and confidence diagnostics"):
                st.json(retrieval)
        except (json.JSONDecodeError, TypeError):
            st.caption("No retrieval diagnostics are available.")

    actions = allowed_actions_for(ticket, user["role"])
    if not actions:
        st.info(f"Your current role ({user['role']}) has no available review action for this ticket state.")
    else:
        with st.form(f"ticket_action_{ticket['ticket_id']}_{ticket['version']}"):
            action = st.selectbox("Action", actions)
            note = st.text_area("Reviewer note / resolution / information requested", placeholder="Record the reason for this decision. Required for resolving or requesting information.", height=90)
            edited_response = None
            if action == "EDIT":
                edited_response = st.text_area("Revised customer response", value=ticket["proposed_response"], height=150)
            assigned_team = None
            if action in {"ASSIGN", "ESCALATE"}:
                current_index = CONFIG["teams"].index(ticket["assigned_team"]) if ticket["assigned_team"] in CONFIG["teams"] else 0
                assigned_team = st.selectbox("Target team", CONFIG["teams"], index=current_index)
            submitted = st.form_submit_button("Apply authorized action", type="primary", use_container_width=True)
        if submitted:
            try:
                result = apply_ticket_action(DB_PATH, ticket["ticket_id"], user["user_id"], action,
                                             expected_version=int(ticket["version"]), note=note,
                                             edited_response=edited_response, assigned_team=assigned_team,
                                             pause_sla_when_on_hold=bool(CONFIG.get("pause_sla_when_on_hold", False)))
                dispatch_results = []
                for notification_id in result["notification_ids"]:
                    dispatch_results.append(notify_dispatch(notification_id, user["user_id"], "normal"))
                refresh_sla_states(DB_PATH, CONFIG)
                st.success(f"Action {action} recorded. Ticket status: {result['status']}.")
                if dispatch_results:
                    st.caption("Notification results: " + "; ".join(f"{x['status']} ({x['attempts_this_cycle']} attempt(s))" for x in dispatch_results))
                st.rerun()
            except Exception as exc:
                display_error(exc)
    history = get_ticket_history(DB_PATH, ticket["ticket_id"])
    with st.expander("Ticket status history / audit trail"):
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        else:
            st.caption("No history is recorded.")


def render_notifications(user: dict[str, Any]) -> None:
    show_header("DELIVERY AND RETRIES", "Notification center", "Email and in-app notifications are simulated. Each attempt is persisted; unapproved customer-facing responses cannot be dispatched.")
    notifications = list_notifications(DB_PATH)
    counts = {s: sum(n["status"] == s for n in notifications) for s in ["DRAFT", "QUEUED", "SENT", "FAILED", "DEAD_LETTER"]}
    cols = st.columns(5)
    for col, status in zip(cols, counts):
        with col: metric_card(status.replace("_", " ").title(), counts[status], "Notification records")
    st.write("")
    filter_status = st.selectbox("Filter notifications", ["All", "DRAFT", "QUEUED", "SENT", "FAILED", "DEAD_LETTER"])
    filtered = list_notifications(DB_PATH, status=filter_status)
    if filtered:
        table = [{"Notification": n["notification_id"], "Ticket": n["ticket_id"], "Event": n["event_type"], "Recipient": n["recipient"],
                  "Channel": n["channel"], "Status": n["status"], "Attempts": n["attempts"],
                  "Approval required": bool(n["requires_approval"]), "Approved by": n["approved_by"] or "—"} for n in filtered]
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        selected = st.selectbox("Inspect notification", [n["notification_id"] for n in filtered], format_func=lambda x: f"{x} · {next(n['event_type'] for n in filtered if n['notification_id']==x)} · {next(n['status'] for n in filtered if n['notification_id']==x)}")
        notification = get_notification(DB_PATH, selected)
        if notification:
            st.text_area("Message preview", value=build_notification_preview(notification), height=180, disabled=True)
            if notification["status"] == "DRAFT" and notification["requires_approval"] and not notification["approved_by"]:
                st.warning("This is a customer-facing response draft. It must be approved in Ticket Queue before dispatch.")
            elif notification["status"] in {"QUEUED", "FAILED"} and can(user["role"], "SEND_NOTIFICATION"):
                mode = st.selectbox("Delivery simulation", ["normal", "fail_first_two_then_success", "always_fail"],
                                    format_func=lambda x: {"normal": "Success on first attempt", "fail_first_two_then_success": "Fail twice, then succeed", "always_fail": "Fail all attempts"}[x],
                                    key=f"delivery_mode_{selected}")
                if st.button("Dispatch / retry selected notification", type="primary", key=f"dispatch_{selected}"):
                    try:
                        result = notify_dispatch(selected, user["user_id"], mode)
                        st.success(f"{result['status']}: {result['message']} Total attempts: {result['total_attempts']}.")
                        st.rerun()
                    except Exception as exc:
                        display_error(exc)
            elif notification["status"] == "SENT":
                st.caption("This notification is already sent. Re-dispatch is blocked by its idempotency state.")
            elif not can(user["role"], "SEND_NOTIFICATION"):
                st.caption("Your role cannot manually dispatch notifications.")
            attempts = list_notification_attempts(DB_PATH, selected)
            if attempts:
                with st.expander("Delivery attempt history"):
                    st.dataframe(pd.DataFrame(attempts), hide_index=True, use_container_width=True)
    else:
        st.info("No notification records match this filter.")

    st.divider()
    st.markdown("#### Retry-mechanism demonstration")
    st.write("Create a clearly labelled internal test notification and choose a controlled failure mode. No real email or messaging provider is contacted.")
    if can(user["role"], "RUN_TESTS"):
        tickets = list_tickets(DB_PATH)
        if tickets:
            labels = {f"{t['ticket_id']} · {t['company_name']} · {t['priority']}": t["ticket_id"] for t in tickets}
            chosen = st.selectbox("Ticket for retry test", list(labels.keys()), key="test_notification_ticket")
            test_mode = st.selectbox("Test failure scenario", ["fail_first_two_then_success", "always_fail", "normal"],
                                     format_func=lambda x: {"normal": "Normal success", "fail_first_two_then_success": "Two failures then success (3 attempts)", "always_fail": "All attempts fail (dead-letter after repeated cycles)"}[x], key="test_notification_mode")
            if st.button("Create and run notification test", type="primary"):
                try:
                    notification_id = create_test_notification(DB_PATH, user["user_id"], labels[chosen])
                    result = notify_dispatch(notification_id, user["user_id"], test_mode)
                    st.success(f"Test {result['status']}. Attempts this cycle: {result['attempts_this_cycle']}; total attempts: {result['total_attempts']}.")
                    st.caption("Inspect the notification above to see each persisted attempt and final status.")
                    st.rerun()
                except Exception as exc:
                    display_error(exc)
        else:
            st.info("Create at least one ticket before running a notification test.")
    else:
        st.caption("Notification retry tests require ADMIN or MANAGER permission.")


def render_knowledge_base() -> None:
    show_header("APPROVED KNOWLEDGE", "Onboarding knowledge base", "Only approved articles can ground responses. The resolver combines keyword overlap with TF-IDF/cosine similarity and records retrieval evidence for each ticket.")
    if KB_LOAD_ERROR:
        st.error(KB_LOAD_ERROR)
        st.stop()
    approved = [a for a in KNOWLEDGE_BASE if a.get("approved") is True]
    st.caption(f"{len(approved)} approved article(s) available. Unapproved articles are excluded from retrieval.")
    query = st.text_input("Test retrieval", placeholder="Example: The CSV import timed out and some records are missing")
    if query.strip():
        article, score, diagnostic = retrieve_article(query, KNOWLEDGE_BASE, min_score=float(CONFIG.get("minimum_retrieval_score", 18)))
        left, right = st.columns([1, 1])
        with left:
            st.metric("Hybrid relevance score", f"{score:.1f}%")
            if article:
                st.success(f"Top approved match: {article['article_id']} — {article['title']}")
                st.write(article["response"])
            else:
                st.warning("No approved article met the minimum retrieval threshold. The workflow should request review or escalate, not invent an answer.")
        with right:
            st.markdown("**Retrieval evidence**")
            st.json(diagnostic)
    st.divider()
    for article in approved:
        with st.expander(f"{article['article_id']} · {article['title']} · v{article.get('version', '1.0')}"):
            c1, c2 = st.columns([1, 2])
            c1.write(f"**Intent:** {article['intent']}")
            c1.write(f"**Owner:** {article.get('owner', 'Not specified')}")
            c1.write(f"**Last reviewed:** {article.get('last_reviewed', 'Not specified')}")
            c1.write("**Status:** Approved")
            c2.write(article["response"])
            st.caption("Keywords: " + ", ".join(article.get("keywords", [])))


def render_analytics(user: dict[str, Any]) -> None:
    show_header("EVALUATION AND FAIRNESS", "Workflow evaluation", "Metrics are calculated from persisted tickets, notification attempts, status history, and labelled synthetic requests. Unavailable denominators are shown as N/A rather than fabricated percentages.")
    metrics = compute_metrics(str(DB_PATH))
    cards = [
        ("Intent accuracy", metrics["classification"]["intent"]["accuracy_pct"], f"{metrics['classification']['intent']['correct']}/{metrics['classification']['intent']['total']} labelled"),
        ("KB resolution rate", metrics["kb_resolution_rate_pct"], f"{metrics['kb_resolution_successes']}/{metrics['kb_resolution_eligible']} eligible"),
        ("Escalation rate", metrics["escalation_rate_pct"], f"{metrics['escalation_count']}/{metrics['valid_ticket_count']} valid tickets"),
        ("Notification success", metrics["notification_success_rate_pct"], f"{metrics['notifications_sent']}/{metrics['notification_delivery_requests']} delivery requests"),
        ("Resolved SLA compliance", metrics["sla_compliance_pct"], f"{metrics['sla_compliant_resolved']}/{metrics['resolved_with_sla']} resolved"),
        ("Mean processing time", metrics["average_processing_seconds"], "Seconds from import timestamp to ticket creation"),
    ]
    rows = [cards[:3], cards[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, (label, value, foot) in zip(cols, row):
            formatted = "N/A" if value is None else (f"{value:.2f}%" if isinstance(value, float) and "rate" in label.lower() or isinstance(value, float) and "accuracy" in label.lower() or isinstance(value, float) and "compliance" in label.lower() else (f"{value:.3f}" if isinstance(value, float) else str(value)))
            with col: metric_card(label, formatted, foot)
        st.write("")
    st.markdown("#### Classification evaluation against manually labelled sample records")
    classification_table = []
    for dim, val in metrics["classification"].items():
        classification_table.append({"Dimension": dim.title(), "Correct": val["correct"], "Labelled total": val["total"],
                                     "Accuracy": f"{val['accuracy_pct']:.2f}%" if val["accuracy_pct"] is not None else "N/A"})
    st.dataframe(pd.DataFrame(classification_table), hide_index=True, use_container_width=True)
    st.caption("Metrics use expected labels only where all corresponding ground-truth fields exist. Empty or unlabeled records do not silently count as correct.")
    with st.expander("Per-class precision, recall, F1, and support"):
        dim_selected = st.selectbox("Classification dimension", ["intent", "complexity", "priority"], key="class_report_dimension")
        report = metrics["classification_reports"].get(dim_selected, [])
        if report:
            st.dataframe(pd.DataFrame(report), use_container_width=True, hide_index=True)
            st.caption(f"Macro-F1: {metrics['classification'][dim_selected]['macro_f1']}")
        else:
            st.caption("No manually labelled records are available for this dimension.")

    more = pd.DataFrame([
        {"Metric": "Human review action rate", "Value": f"{metrics['review_completion_rate_pct']:.2f}%" if metrics["review_completion_rate_pct"] is not None else "N/A", "Count": f"{metrics['review_actions_recorded']}/{metrics['review_assigned']}"},
        {"Metric": "Reopened ticket rate", "Value": f"{metrics['reopened_ticket_rate_pct']:.2f}%" if metrics["reopened_ticket_rate_pct"] is not None else "N/A", "Count": f"{metrics['reopened_tickets']}/{metrics['total_tickets']}"},
        {"Metric": "Retry-recovered notifications", "Value": str(metrics["retry_recovered_notifications"]), "Count": "Notifications that had a failed attempt before success"},
        {"Metric": "Audit completeness", "Value": f"{metrics['audit_completeness_pct']:.2f}%" if metrics["audit_completeness_pct"] is not None else "N/A", "Count": f"{metrics['tickets_with_audit_events']}/{metrics['tickets_total_for_audit']} tickets"},
        {"Metric": "Auto-resolution policy violations", "Value": str(metrics["false_auto_resolution_policy_violations"]), "Count": "P1/P2 or non-low-complexity auto-resolutions; expected 0"},
    ])
    st.markdown("#### Additional workflow quality metrics")
    st.dataframe(more, hide_index=True, use_container_width=True)
    counts = metrics["notification_status_counts"]
    if counts:
        st.markdown("#### Notification delivery outcomes")
        st.bar_chart(pd.DataFrame.from_dict(counts, orient="index", columns=["Notifications"]))

    st.divider()
    st.markdown("#### Variation 2 comparison: keyword-only vs hybrid semantic retrieval")
    tickets = list_tickets(DB_PATH)
    if tickets:
        compare_rows = []
        for ticket in tickets[:30]:
            keyword_id, keyword_score = keyword_only_retrieval(ticket["request_text"], KNOWLEDGE_BASE)
            hybrid_article, hybrid_score, _ = retrieve_article(ticket["request_text"], KNOWLEDGE_BASE, float(CONFIG.get("minimum_retrieval_score", 18)))
            hybrid_id = hybrid_article.get("article_id") if hybrid_article else None
            compare_rows.append({"Ticket": ticket["ticket_id"], "Request": ticket["request_text"],
                                 "Keyword baseline article": keyword_id or "No match", "Keyword score": keyword_score,
                                 "Hybrid article": hybrid_id or "No match", "Hybrid score": hybrid_score,
                                 "Article selection agrees": keyword_id == hybrid_id})
        st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)
        st.caption("This is a retrieval-method comparison, not a claim that semantic retrieval is always superior. Review article correctness alongside match rate and escalation outcomes.")
    else:
        st.caption("Import tickets to populate retrieval comparison.")

    st.divider()
    st.markdown("#### Variation 1 comparison: deterministic vs Groq classifier")
    api_key_present = bool(os.getenv("GROQ_API_KEY", "").strip())
    if not api_key_present:
        st.info("GROQ_API_KEY is not configured. Rule-based classification is active. Add the key to .env to enable this optional comparison.")
    elif not can(user["role"], "RUN_TESTS"):
        st.caption("Classifier comparisons require ADMIN or MANAGER permission.")
    else:
        st.write("The comparison makes live API calls only after the button is pressed. It runs on at most five stored requests.")
        if st.button("Compare rules against Groq on up to 5 requests", type="primary"):
            rows = []
            for ticket in tickets[:5]:
                start = time.perf_counter()
                rule_result = classify_request(ticket["request_text"], use_groq=False)
                try:
                    groq_result = classify_with_groq(ticket["request_text"])
                    error = ""
                except Exception as exc:
                    groq_result = {}
                    error = f"{type(exc).__name__}: {exc}"
                latency = round(time.perf_counter() - start, 3)
                rows.append({"Ticket": ticket["ticket_id"], "Rules intent": rule_result["intent"], "Groq intent": groq_result.get("intent", "ERROR"),
                             "Intent agrees": rule_result["intent"] == groq_result.get("intent"),
                             "Rules complexity": rule_result["complexity"], "Groq complexity": groq_result.get("complexity", "—"),
                             "Rules priority": rule_result["priority"], "Groq priority": groq_result.get("priority", "—"),
                             "Groq latency (s)": latency, "Error": error})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### Automated test suite")
    st.write("Runs validation, classification, retrieval, authorization, stale-update, notification retry/idempotency, and SLA tests against temporary test databases.")
    if can(user["role"], "RUN_TESTS"):
        if st.button("Run automated workflow tests", type="primary"):
            with st.spinner("Running automated tests in an isolated temporary database…"):
                proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=BASE_DIR,
                                      capture_output=True, text=True, timeout=120)
            if proc.returncode == 0:
                st.success("All automated tests passed.")
            else:
                st.error(f"Test suite returned exit code {proc.returncode}.")
            st.code((proc.stdout or "") + ("\nSTDERR:\n" + proc.stderr if proc.stderr else ""))
    else:
        st.caption("Running automated tests requires ADMIN or MANAGER permission.")


def render_audit(user: dict[str, Any]) -> None:
    show_header("SECURITY AND EVIDENCE", "Audit trail and data quality", "Review persisted decisions, authorization failures, ticket state transitions, retry history, malformed inputs, and import errors. Audit rows exclude secrets and full credential data.")
    events = list_audit_events(DB_PATH, limit=5000)
    event_types = ["All"] + sorted({e["event_type"] for e in events})
    chosen_type = st.selectbox("Audit event type", event_types)
    filtered = events if chosen_type == "All" else [e for e in events if e["event_type"] == chosen_type]
    if filtered:
        audit_df = pd.DataFrame(filtered)
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
        if can(user["role"], "EXPORT_AUDIT"):
            df_download("Download audit_log.csv", audit_df, "audit_log.csv", "download_audit")
    else:
        st.info("No audit events for the selected filter.")

    st.markdown("#### Ticket status history")
    events = list_ticket_events(DB_PATH, limit=5000)
    if events:
        event_df = pd.DataFrame(events)
        st.dataframe(event_df, use_container_width=True, hide_index=True)
        if can(user["role"], "EXPORT_AUDIT"):
            df_download("Download ticket_history.csv", event_df, "ticket_history.csv", "download_ticket_history")
    st.markdown("#### Notification log")
    notifications = list_notifications(DB_PATH)
    if notifications:
        notif_df = pd.DataFrame(notifications)
        st.dataframe(notif_df, use_container_width=True, hide_index=True)
        if can(user["role"], "EXPORT_AUDIT"):
            df_download("Download notification_log.csv", notif_df, "notification_log.csv", "download_notification_log")
            if st.button("Write audit_log.csv and notification_log.csv into exports/", key="write_exports"):
                out = BASE_DIR / "exports"
                out.mkdir(exist_ok=True)
                pd.DataFrame(list_audit_events(DB_PATH, limit=100000)).to_csv(out / "audit_log.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(list_notifications(DB_PATH)).to_csv(out / "notification_log.csv", index=False, encoding="utf-8-sig")
                st.success(f"CSV exports written to {out}.")


def render_users(user: dict[str, Any]) -> None:
    show_header("ACCESS CONTROL", "User and reviewer management", "Reviewer role is loaded from the database at action time. The role shown in the browser is never trusted as an authorization decision.")
    if not can(user["role"], "MANAGE_USERS"):
        st.error("Only ADMIN can manage users.")
        return
    users = list_users(DB_PATH)
    st.dataframe(pd.DataFrame(users), hide_index=True, use_container_width=True)
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Create a user")
            with st.form("create_user_form", clear_on_submit=True):
                username = st.text_input("Username", placeholder="e.g. reviewer_01")
                role = st.selectbox("Role", ["MANAGER", "SPECIALIST", "AGENT", "ADMIN"])
                password = st.text_input("Initial password (minimum 10 characters)", type="password")
                submit = st.form_submit_button("Create user", type="primary")
            if submit:
                try:
                    created = create_user(DB_PATH, user["user_id"], username, password, role)
                    st.success(f"Created {created['username']} with role {created['role']}.")
                    st.rerun()
                except Exception as exc:
                    display_error(exc)
    with right:
        with st.container(border=True):
            st.markdown("#### Activate or deactivate an account")
            targets = [u for u in users if u["user_id"] != user["user_id"]]
            if targets:
                target_options = {f"{u['username']} · {u['role']} · {'active' if u['active'] else 'inactive'}": u for u in targets}
                chosen = st.selectbox("Account", list(target_options.keys()))
                target = target_options[chosen]
                next_active = not bool(target["active"])
                if st.button("Activate account" if next_active else "Deactivate account"):
                    try:
                        set_user_active(DB_PATH, user["user_id"], target["user_id"], next_active)
                        st.success("Account state updated.")
                        st.rerun()
                    except Exception as exc:
                        display_error(exc)
            else:
                st.caption("No other user accounts are available.")
    st.caption("Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes. Demo bootstrap credentials are intended for local coursework only.")


def main() -> None:
    if "user_id" not in st.session_state:
        render_login()
        return
    user = get_user_by_id(DB_PATH, st.session_state["user_id"])
    if not user or not user.get("active"):
        st.session_state.pop("user_id", None)
        st.warning("Your session is no longer active. Sign in again.")
        st.rerun()
        return

    refresh_sla_states(DB_PATH, CONFIG)
    with st.sidebar:
        st.markdown('<div class="app-eyebrow">SAAS OPERATIONS</div>', unsafe_allow_html=True)
        st.markdown("## OnboardIQ")
        st.caption("Onboarding & implementation")
        st.divider()
        st.write(f"**{user['username']}**")
        st.caption(f"Role: {user['role']}")
        pages = ["Overview", "Intake & Processing", "Ticket Queue", "Notifications", "Knowledge Base", "Analytics & Evaluation", "Audit & Security"]
        if can(user["role"], "MANAGE_USERS"):
            pages.append("User Management")
        page = st.radio("Workspace", pages, key="selected_page", label_visibility="collapsed")
        st.divider()
        groq_available = bool(os.getenv("GROQ_API_KEY", "").strip())
        use_groq = False
        if groq_available:
            default_groq = os.getenv("USE_GROQ_CLASSIFIER", "false").lower() in {"1", "true", "yes"}
            use_groq = st.toggle("Use Groq classification", value=default_groq, help="Enabled only when GROQ_API_KEY is configured. Deterministic safety overrides remain active.")
        else:
            st.caption("Classifier: deterministic rules")
        st.divider()
        st.caption("Mock notifications · SQLite persistence · Local knowledge base")
        if st.button("Sign out", use_container_width=True):
            log_audit(DB_PATH, user["user_id"], "USER_LOGOUT", details="User signed out.")
            st.session_state.pop("user_id", None)
            st.rerun()

    if page == "Overview":
        render_overview(user)
    elif page == "Intake & Processing":
        render_intake(user, use_groq)
    elif page == "Ticket Queue":
        render_ticket_queue(user)
    elif page == "Notifications":
        render_notifications(user)
    elif page == "Knowledge Base":
        render_knowledge_base()
    elif page == "Analytics & Evaluation":
        render_analytics(user)
    elif page == "Audit & Security":
        render_audit(user)
    elif page == "User Management":
        render_users(user)


if __name__ == "__main__":
    main()
