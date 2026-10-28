from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Any

PASSWORD_ITERATIONS = 310_000

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "ADMIN": {
        "VIEW_TICKETS", "APPROVE", "EDIT", "HOLD", "REQUEST_INFORMATION",
        "ESCALATE", "RESOLVE", "REOPEN", "ASSIGN", "SEND_NOTIFICATION",
        "MANAGE_USERS", "EXPORT_AUDIT", "RUN_TESTS",
    },
    "MANAGER": {
        "VIEW_TICKETS", "APPROVE", "EDIT", "HOLD", "REQUEST_INFORMATION",
        "ESCALATE", "RESOLVE", "REOPEN", "ASSIGN", "SEND_NOTIFICATION",
        "EXPORT_AUDIT", "RUN_TESTS",
    },
    "SPECIALIST": {
        "VIEW_TICKETS", "EDIT", "HOLD", "REQUEST_INFORMATION", "ESCALATE",
        "RESOLVE", "REOPEN", "ASSIGN", "SEND_NOTIFICATION", "EXPORT_AUDIT",
    },
    "AGENT": {"VIEW_TICKETS", "HOLD", "REQUEST_INFORMATION", "SEND_NOTIFICATION"},
}


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    if not password:
        raise ValueError("Password cannot be empty.")
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return digest.hex(), salt.hex()


def verify_password(password: str, expected_hash: str, salt_hex: str) -> bool:
    actual_hash, _ = hash_password(password, salt_hex)
    return hmac.compare_digest(actual_hash, expected_hash)


def can(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get((role or "").upper(), set())


def bootstrap_users_from_environment() -> list[dict[str, str]]:
    """Local-demo bootstrap credentials. Change them before using outside localhost."""
    return [
        {
            "username": os.getenv("BOOTSTRAP_ADMIN_USERNAME", "admin"),
            "password": os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "AdminDemo!2026"),
            "role": "ADMIN",
        },
        {
            "username": os.getenv("BOOTSTRAP_MANAGER_USERNAME", "manager"),
            "password": os.getenv("BOOTSTRAP_MANAGER_PASSWORD", "ManagerDemo!2026"),
            "role": "MANAGER",
        },
        {
            "username": os.getenv("BOOTSTRAP_SPECIALIST_USERNAME", "specialist"),
            "password": os.getenv("BOOTSTRAP_SPECIALIST_PASSWORD", "SpecialistDemo!2026"),
            "role": "SPECIALIST",
        },
    ]
