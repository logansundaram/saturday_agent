from __future__ import annotations

import json
from typing import Any, Dict


def json_dumps(value: Dict[str, Any]) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def encode_sse_message(payload: Dict[str, Any]) -> bytes:
    return f"event: message\ndata: {json_dumps(payload)}\n\n".encode("utf-8")
