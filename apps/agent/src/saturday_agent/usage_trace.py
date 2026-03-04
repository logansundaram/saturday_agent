from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_TRACE_LOCK = threading.Lock()
_TRACE_ENV = "SATURDAY_USAGE_TRACE_PATH"


def emit_usage_trace(event_type: str, payload: Dict[str, Any] | None = None) -> None:
    trace_path = str(os.getenv(_TRACE_ENV, "")).strip()
    if not trace_path:
        return

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "agent",
        "event_type": str(event_type or "").strip() or "unknown",
        **dict(payload or {}),
    }
    destination = Path(trace_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str))
            handle.write("\n")
