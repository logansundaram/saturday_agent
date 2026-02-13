from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from saturday_agent.rag.embeddings import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_EMBED_MODEL,
    get_ollama_embeddings,
)
from saturday_agent.rag.qdrant_store import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_URL,
    get_vectorstore,
)


def index_chunks_to_qdrant(
    chunks: Sequence[str],
    *,
    doc_id: str,
    filename: str,
    sha256: str,
    source_path: str,
    collection: str,
    embeddings_model: str,
) -> Dict[str, Any]:
    texts = [str(item or "").strip() for item in chunks if str(item or "").strip()]
    if not texts:
        return {"ok": True, "added": 0}

    resolved_collection = str(
        collection or os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)
    ).strip()
    resolved_model = str(
        embeddings_model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
    ).strip()
    qdrant_url = str(os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)).strip()
    ollama_base_url = str(os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).strip()

    embeddings = get_ollama_embeddings(resolved_model, ollama_base_url)
    vectorstore = get_vectorstore(resolved_collection, embeddings, qdrant_url)

    metadatas: list[Dict[str, Any]] = []
    ids: list[str] = []
    for chunk_index, _ in enumerate(texts):
        ids.append(f"{doc_id}:{chunk_index}")
        metadatas.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_index,
                "chunk_index": chunk_index,
                "filename": filename,
                "sha256": sha256,
                "source_path": source_path,
                "source": source_path,
            }
        )

    added = vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    added_count = len(added) if isinstance(added, list) else len(texts)
    return {"ok": True, "added": int(added_count)}
