from __future__ import annotations

import re
from typing import List

MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4000
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150


def _normalize_chunk_size(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_CHUNK_SIZE
    return max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, parsed))


def _normalize_overlap(value: int, *, chunk_size: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_CHUNK_OVERLAP
    return max(0, min(parsed, max(chunk_size - 1, 0)))


def _normalize_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    resolved_chunk_size = _normalize_chunk_size(chunk_size)
    resolved_overlap = _normalize_overlap(overlap, chunk_size=resolved_chunk_size)
    step = max(1, resolved_chunk_size - resolved_overlap)

    chunks: List[str] = []
    cursor = 0
    total_length = len(normalized)

    while cursor < total_length:
        chunk = normalized[cursor : cursor + resolved_chunk_size].strip()
        if chunk and (not chunks or chunks[-1] != chunk):
            chunks.append(chunk)
        if cursor + resolved_chunk_size >= total_length:
            break
        cursor += step

    return chunks
