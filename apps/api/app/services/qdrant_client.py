from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx

try:
    from qdrant_client import QdrantClient
except ModuleNotFoundError:
    QdrantClient = None  # type: ignore[assignment]

_LOCK = threading.RLock()
_RUNTIME_QDRANT_URL: Optional[str] = None


def _default_config_path() -> Path:
    override = str(
        os.getenv("SATURDAY_QDRANT_CONFIG_PATH")
        or os.getenv("SATURDAY_QDRANT_CONFIG_FILE")
        or ""
    ).strip()
    if override:
        return Path(override).expanduser()

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "desktop" / "qdrant" / "qdrant.json"

    if sys.platform.startswith("win"):
        appdata = str(os.getenv("APPDATA", "")).strip()
        if appdata:
            return Path(appdata) / "desktop" / "qdrant" / "qdrant.json"
        return home / "AppData" / "Roaming" / "desktop" / "qdrant" / "qdrant.json"

    xdg_config = str(os.getenv("XDG_CONFIG_HOME", "")).strip()
    base = Path(xdg_config) if xdg_config else home / ".config"
    return base / "desktop" / "qdrant" / "qdrant.json"


def _load_url_from_file() -> Optional[str]:
    config_path = _default_config_path()
    if not config_path.exists():
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    raw_url = payload.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None

    try:
        return _normalize_url(raw_url)
    except Exception:
        return None


def _normalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        raise ValueError("Qdrant URL must be a non-empty string.")

    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Qdrant URL must use http or https.")
    if not parsed.netloc:
        raise ValueError("Qdrant URL must include host:port.")
    return f"{parsed.scheme.lower()}://{parsed.netloc}".rstrip("/")


def set_qdrant_url(url: str) -> str:
    normalized = _normalize_url(url)
    with _LOCK:
        global _RUNTIME_QDRANT_URL
        _RUNTIME_QDRANT_URL = normalized
        os.environ["QDRANT_URL"] = normalized
    return normalized


def get_qdrant_url() -> Optional[str]:
    with _LOCK:
        runtime_url = _RUNTIME_QDRANT_URL
    if runtime_url:
        return runtime_url

    env_url = str(os.getenv("QDRANT_URL", "")).strip()
    if env_url:
        return env_url.rstrip("/")

    file_url = _load_url_from_file()
    if file_url:
        return file_url.rstrip("/")
    return None


def build_client(url: Optional[str] = None) -> "QdrantClient":
    resolved_url = str(url or get_qdrant_url() or "").strip()
    if not resolved_url:
        raise RuntimeError("Qdrant URL is not configured.")
    if QdrantClient is None:
        raise RuntimeError("qdrant-client dependency is not available.")
    return QdrantClient(url=resolved_url)  # type: ignore[misc]


def is_qdrant_reachable(url: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    resolved_url = str(url or get_qdrant_url() or "").strip()
    if not resolved_url:
        return False, "Qdrant URL is not configured."

    try:
        with httpx.Client(timeout=1.5, follow_redirects=True) as client:
            response = client.get(f"{resolved_url.rstrip('/')}/collections")
        if response.status_code >= 400:
            return False, f"Qdrant returned HTTP {response.status_code}."
    except Exception as exc:
        return False, str(exc)

    return True, None
