from __future__ import annotations

import hashlib
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import UploadFile

from app import db

_ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_suffix(file: UploadFile, mime: str) -> str:
    filename = str(file.filename or "").strip()
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix
    guessed = mimetypes.guess_extension(mime) or ""
    return guessed


def _infer_suffix_from_name(filename: str, mime: str) -> str:
    normalized_name = str(filename or "").strip()
    if normalized_name:
        suffix = Path(normalized_name).suffix
        if suffix:
            return suffix
    guessed = mimetypes.guess_extension(mime) or ""
    return guessed


def save_bytes(*, filename: str, mime: str, content: bytes) -> Dict[str, Any]:
    normalized_mime = str(mime or "").strip().lower()
    if not normalized_mime.startswith("image/"):
        raise ValueError("Only image uploads are supported.")

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact_id = str(uuid.uuid4())
    suffix = _infer_suffix_from_name(filename, normalized_mime)
    artifact_path = (_ARTIFACTS_DIR / f"{artifact_id}{suffix}").resolve()

    data = bytes(content or b"")
    artifact_path.write_bytes(data)

    payload = {
        "artifact_id": artifact_id,
        "path": str(artifact_path),
        "mime": normalized_mime,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

    db.add_artifact(
        artifact_id=artifact_id,
        path=payload["path"],
        mime=payload["mime"],
        size=payload["size"],
        sha256=payload["sha256"],
        created_at=_utc_now_iso(),
    )
    return payload


def save_upload(file: UploadFile) -> Dict[str, Any]:
    file.file.seek(0)
    data = file.file.read()
    file.file.seek(0)
    return save_bytes(
        filename=str(file.filename or ""),
        mime=str(file.content_type or ""),
        content=bytes(data or b""),
    )


def read_artifact(artifact_id: str) -> Dict[str, Any]:
    row = db.read_artifact(str(artifact_id))
    if row is None:
        raise FileNotFoundError("Artifact not found.")

    artifact_path = Path(str(row.get("path") or ""))
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError("Artifact file is missing from disk.")

    return {
        "artifact_id": str(row.get("artifact_id") or artifact_id),
        "path": str(artifact_path),
        "mime": str(row.get("mime") or "application/octet-stream"),
        "size": int(row.get("size") or artifact_path.stat().st_size),
        "sha256": str(row.get("sha256") or ""),
        "bytes": artifact_path.read_bytes(),
    }
