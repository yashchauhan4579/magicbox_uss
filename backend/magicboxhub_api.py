"""
Thin client for app.magicboxhub.net used by the Crowd Live module.

Exposes login() + fetch_tree() and an in-process token cache so the
operator-facing UI doesn't need to re-enter credentials on every request.

The cache stores creds in memory only; on a backend restart you either
need to re-enter via the UI or set MAGICBOXHUB_EMAIL/MAGICBOXHUB_PASSWORD
in the env so the first call auto-logs-in.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import requests


BASE_URL = os.environ.get("MAGICBOXHUB_BASE_URL", "https://app.magicboxhub.net").rstrip("/")
DEFAULT_TIMEOUT = 15

# JWT is valid 7 days per the API doc; refresh proactively after 6 days.
TOKEN_TTL_SEC = 6 * 24 * 3600

_lock = threading.Lock()
_state: dict[str, Any] = {
    "email": os.environ.get("MAGICBOXHUB_EMAIL", "").strip() or None,
    "password": os.environ.get("MAGICBOXHUB_PASSWORD", "").strip() or None,
    "token": None,
    "token_at": 0.0,
}


class MagicboxError(RuntimeError):
    """Raised when the upstream API rejects a request."""


def _do_login(email: str, password: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": email, "password": password},
        timeout=DEFAULT_TIMEOUT,
    )
    if resp.status_code != 200:
        raise MagicboxError(f"login failed ({resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    token = data.get("token")
    if not token:
        raise MagicboxError("login response missing token")
    return token


def login(email: str, password: str) -> str:
    """Log in and cache token + creds for subsequent calls."""
    email = (email or "").strip()
    password = (password or "").strip()
    if not email or not password:
        raise MagicboxError("email and password required")
    token = _do_login(email, password)
    with _lock:
        _state["email"] = email
        _state["password"] = password
        _state["token"] = token
        _state["token_at"] = time.time()
    return token


def _ensure_token() -> str:
    """Return a valid cached token, refreshing if expired or missing."""
    with _lock:
        token = _state["token"]
        age = time.time() - _state["token_at"]
        email = _state["email"]
        password = _state["password"]
    if token and age < TOKEN_TTL_SEC:
        return token
    if not (email and password):
        raise MagicboxError("not logged in — call POST /api/crowd-live/magicbox/login first")
    token = _do_login(email, password)
    with _lock:
        _state["token"] = token
        _state["token_at"] = time.time()
    return token


def has_credentials() -> bool:
    with _lock:
        return bool(_state["email"] and _state["password"])


def status() -> dict[str, Any]:
    with _lock:
        return {
            "logged_in": bool(_state["token"]),
            "has_credentials": bool(_state["email"] and _state["password"]),
            "email": _state["email"],
            "base_url": BASE_URL,
        }


def logout() -> None:
    with _lock:
        _state["token"] = None
        _state["token_at"] = 0.0
        _state["email"] = None
        _state["password"] = None


def fetch_tree() -> dict[str, Any]:
    """GET /api/camera-tree — auto-refresh token on 401."""
    token = _ensure_token()
    resp = requests.get(
        f"{BASE_URL}/api/camera-tree",
        headers={"Authorization": f"Bearer {token}"},
        timeout=DEFAULT_TIMEOUT,
    )
    if resp.status_code == 401:
        # token may have been revoked early — force re-login if creds available
        with _lock:
            _state["token"] = None
        token = _ensure_token()
        resp = requests.get(
            f"{BASE_URL}/api/camera-tree",
            headers={"Authorization": f"Bearer {token}"},
            timeout=DEFAULT_TIMEOUT,
        )
    if resp.status_code != 200:
        raise MagicboxError(f"camera-tree failed ({resp.status_code}): {resp.text[:200]}")
    return resp.json()
