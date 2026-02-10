from __future__ import annotations

import base64
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict

import httpx

from saturday_agent.llms.ollama_chat import extract_assistant_text
from saturday_agent.llms.vision_registry import VisionModelRegistry

TOOL_ID = "vision.analyze"
TOOL_NAME = "Vision Analyze"
TOOL_DESCRIPTION = "Analyze an image using a local vision model."

VISION_ANALYZE_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "artifact_id": {"type": "string"},
        "prompt": {"type": "string"},
        "vision_model_id": {"type": "string"},
        "detail": {"type": "string", "enum": ["low", "high"]},
    },
    "required": ["artifact_id", "prompt", "vision_model_id"],
    "additionalProperties": False,
}

VISION_ANALYZE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "artifact_id": {"type": "string"},
        "vision_model_id": {"type": "string"},
        "result_text": {"type": "string"},
        "raw": {},
    },
    "required": ["artifact_id", "vision_model_id", "result_text"],
}


def _resolve_db_path() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    env_value = os.getenv("SATURDAY_DB_PATH")
    if env_value:
        env_path = Path(env_value)
        if env_path.is_absolute():
            return env_path
        if env_path.parts and env_path.parts[0] == "apps":
            return repo_root / env_path
        return Path.cwd() / env_path
    return repo_root / "apps/api/saturday.db"


def _read_artifact_row(artifact_id: str) -> Dict[str, Any] | None:
    db_path = _resolve_db_path()
    if not db_path.exists():
        return None

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT artifact_id, path, mime, size, sha256, created_at
            FROM artifacts
            WHERE artifact_id = ?
            """,
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)


def _make_error(
    *,
    artifact_id: str,
    vision_model_id: str,
    error_type: str,
    message: str,
    raw: Any = None,
) -> Dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "vision_model_id": vision_model_id,
        "result_text": "",
        "raw": raw,
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def _analyze_single_artifact(
    *,
    artifact_id: str,
    prompt_text: str,
    vision_model_id: str,
    base_url: str,
    timeout_seconds: float,
    is_known_vision_model: bool,
) -> Dict[str, Any]:
    artifact = _read_artifact_row(artifact_id)
    if artifact is None:
        return _make_error(
            artifact_id=artifact_id,
            vision_model_id=vision_model_id,
            error_type="missing_artifact",
            message=f"Artifact '{artifact_id}' was not found.",
        )

    artifact_path = Path(str(artifact.get("path") or ""))
    mime = str(artifact.get("mime") or "application/octet-stream").strip().lower()
    if not artifact_path.exists():
        return _make_error(
            artifact_id=artifact_id,
            vision_model_id=vision_model_id,
            error_type="missing_file",
            message=f"Artifact file is missing on disk: {artifact_path}",
        )
    if not mime.startswith("image/"):
        return _make_error(
            artifact_id=artifact_id,
            vision_model_id=vision_model_id,
            error_type="invalid_artifact_type",
            message=f"Artifact '{artifact_id}' is not an image (mime={mime}).",
        )

    image_b64 = base64.b64encode(artifact_path.read_bytes()).decode("ascii")
    payload: Dict[str, Any] = {
        "model": vision_model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
                "images": [image_b64],
            }
        ],
        "stream": False,
    }

    try:
        with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
            response = client.post("/api/chat", json=payload)
    except httpx.HTTPError as exc:
        return _make_error(
            artifact_id=artifact_id,
            vision_model_id=vision_model_id,
            error_type="http_error",
            message=f"Ollama request failed: {exc}",
        )

    raw: Any
    try:
        raw = response.json()
    except ValueError:
        raw = {"status_code": response.status_code, "text": response.text}

    if response.status_code >= 400:
        error_text = ""
        if isinstance(raw, dict):
            error_text = str(raw.get("error") or raw.get("message") or "").strip()
        if not error_text:
            error_text = str(response.text).strip()
        lowered = error_text.lower()
        if (
            ("vision" in lowered or "image" in lowered or "multimodal" in lowered)
            and ("unsupported" in lowered or "not support" in lowered or "capab" in lowered)
        ) or (not is_known_vision_model and "image" in lowered):
            error_type = "unsupported_model"
        else:
            error_type = "ollama_error"

        return _make_error(
            artifact_id=artifact_id,
            vision_model_id=vision_model_id,
            error_type=error_type,
            message=error_text or f"Ollama returned HTTP {response.status_code}.",
            raw=raw,
        )

    response_text = extract_assistant_text(raw if isinstance(raw, dict) else {}).strip()
    if not response_text and isinstance(raw, dict):
        response_text = str(raw.get("response") or "").strip()

    if not response_text:
        return _make_error(
            artifact_id=artifact_id,
            vision_model_id=vision_model_id,
            error_type="empty_response",
            message="Vision model returned an empty response.",
            raw=raw,
        )

    return {
        "artifact_id": artifact_id,
        "vision_model_id": vision_model_id,
        "result_text": response_text,
        "raw": raw,
    }


def analyze_image_ollama(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tool_input, dict):
        raise ValueError("vision.analyze input must be an object.")

    artifact_id = str(tool_input.get("artifact_id") or "").strip()
    prompt = str(tool_input.get("prompt") or "").strip()
    vision_model_id = str(tool_input.get("vision_model_id") or "").strip()
    detail = str(tool_input.get("detail") or "").strip().lower()

    artifact_ids = [item.strip() for item in artifact_id.split(",") if item.strip()]
    if not artifact_ids:
        raise ValueError("vision.analyze requires a non-empty 'artifact_id'.")
    if not prompt:
        raise ValueError("vision.analyze requires a non-empty 'prompt'.")
    if not vision_model_id:
        raise ValueError("vision.analyze requires a non-empty 'vision_model_id'.")
    if detail and detail not in {"low", "high"}:
        raise ValueError("vision.analyze 'detail' must be either 'low' or 'high'.")

    prompt_text = prompt
    if detail == "high":
        prompt_text = f"{prompt}\n\nProvide a detailed visual analysis."
    elif detail == "low":
        prompt_text = f"{prompt}\n\nRespond concisely."

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT", "60"))
    registry = VisionModelRegistry(
        base_url=base_url,
        default_model=(
            os.getenv("VITE_VISION_DEFAULT_MODEL")
            or os.getenv("VISION_DEFAULT_MODEL")
            or os.getenv("OLLAMA_MODEL", "")
        ),
        timeout_seconds=timeout_seconds,
        allowlist_raw=os.getenv("VISION_MODEL_ALLOWLIST", ""),
    )
    is_known_vision_model = registry.is_vision_model(vision_model_id)

    successful_texts: list[str] = []
    per_artifact_raw: list[Dict[str, Any]] = []
    errors: list[Dict[str, str]] = []

    for current_artifact_id in artifact_ids:
        result = _analyze_single_artifact(
            artifact_id=current_artifact_id,
            prompt_text=prompt_text,
            vision_model_id=vision_model_id,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            is_known_vision_model=is_known_vision_model,
        )
        per_artifact_raw.append(
            {
                "artifact_id": current_artifact_id,
                "raw": result.get("raw"),
            }
        )
        if "error" in result:
            error_payload = result.get("error")
            if isinstance(error_payload, dict):
                errors.append(
                    {
                        "artifact_id": current_artifact_id,
                        "type": str(error_payload.get("type") or "unknown_error"),
                        "message": str(error_payload.get("message") or "Vision analysis failed."),
                    }
                )
            continue

        text = str(result.get("result_text") or "").strip()
        if text:
            if len(artifact_ids) > 1:
                successful_texts.append(f"[{current_artifact_id}] {text}")
            else:
                successful_texts.append(text)

    if not successful_texts:
        first_error = errors[0] if errors else {"type": "unknown_error", "message": "Vision analysis failed."}
        return _make_error(
            artifact_id=artifact_id,
            vision_model_id=vision_model_id,
            error_type=str(first_error.get("type") or "unknown_error"),
            message=str(first_error.get("message") or "Vision analysis failed."),
            raw={"results": per_artifact_raw, "errors": errors},
        )

    response_payload: Dict[str, Any] = {
        "artifact_id": artifact_id,
        "vision_model_id": vision_model_id,
        "result_text": "\n\n".join(successful_texts),
    }
    if len(artifact_ids) > 1 or errors:
        response_payload["raw"] = {"results": per_artifact_raw, "errors": errors}
    elif per_artifact_raw:
        response_payload["raw"] = per_artifact_raw[0].get("raw")
    return response_payload
