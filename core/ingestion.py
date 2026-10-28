from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from typing import Any

REQUIRED_FIELDS = {
    "request_id", "customer_id", "company_name", "contact_email", "request_text", "created_at"
}
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _normalize_record(raw: dict[str, Any]) -> dict[str, str]:
    return {
        str(key).strip(): "" if value is None else str(value).strip()
        for key, value in raw.items()
        if key is not None
    }


def _validate_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except (ValueError, TypeError):
        return False


def parse_uploaded_file(filename: str, content: bytes) -> list[dict[str, Any]]:
    """Parse a CSV or JSON upload into raw row dictionaries."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    decoded = content.decode("utf-8-sig")
    if suffix == "csv":
        reader = csv.DictReader(io.StringIO(decoded, newline=""))
        if not reader.fieldnames:
            raise ValueError("CSV is empty or has no header row.")
        headers = {str(name).strip() for name in reader.fieldnames if name is not None}
        missing = sorted(REQUIRED_FIELDS - headers)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        return [dict(row) for row in reader]
    if suffix == "json":
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}).") from exc
        if isinstance(payload, dict) and isinstance(payload.get("requests"), list):
            payload = payload["requests"]
        if not isinstance(payload, list):
            raise ValueError("JSON must be an array of request objects or an object containing a 'requests' array.")
        if any(not isinstance(item, dict) for item in payload):
            raise ValueError("Every item in the JSON array must be an object.")
        return payload
    raise ValueError("Unsupported file format. Upload a .csv or .json file.")


def validate_records(raw_records: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Validate field presence, timestamp, email shape, and batch-local ID uniqueness."""
    valid: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row_number, raw in enumerate(raw_records, start=2):
        row = _normalize_record(raw)
        errors: list[str] = []
        missing = [field for field in sorted(REQUIRED_FIELDS) if not row.get(field, "").strip()]
        if missing:
            errors.append("Missing required field(s): " + ", ".join(missing))
        request_id = row.get("request_id", "")
        if request_id:
            if request_id in seen_ids:
                errors.append("Duplicate request_id within this import batch")
            else:
                seen_ids.add(request_id)
        if row.get("request_text", "").strip() == "":
            errors.append("Request text is empty or contains only whitespace")
        email = row.get("contact_email", "")
        if email and not _EMAIL_RE.match(email):
            errors.append("Contact email is not in an expected email format")
        timestamp = row.get("created_at", "")
        if timestamp and not _validate_timestamp(timestamp):
            errors.append("created_at must be an ISO-8601 timestamp")
        row["_source_row_number"] = str(row_number)
        row.setdefault("source", "upload")
        row["source"] = row.get("source") or "upload"

        if errors:
            rejected.append({"row_number": row_number, "record": row, "errors": errors})
        else:
            valid.append(row)

    return valid, rejected
