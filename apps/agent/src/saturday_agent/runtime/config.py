from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    ollama_base_url: str
    default_model: str
    ollama_timeout_seconds: float
    max_verify_retries: int = 1


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        default_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        ollama_timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT", "30")),
        max_verify_retries=int(os.getenv("WORKFLOW_MAX_VERIFY_RETRIES", "1")),
    )
