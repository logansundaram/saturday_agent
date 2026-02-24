from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient

if TYPE_CHECKING:
    from langchain_ollama.embeddings import OllamaEmbeddings
    from langchain_qdrant import QdrantVectorStore
else:
    OllamaEmbeddings = Any
    QdrantVectorStore = Any

DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_QDRANT_COLLECTION = "saturday_docs"


class QdrantDependencyError(RuntimeError):
    """Raised when the LangChain Qdrant dependency is missing."""


def _resolve_qdrant_vectorstore_class() -> type[Any]:
    try:
        from langchain_qdrant import QdrantVectorStore as _QdrantVectorStore
    except ModuleNotFoundError as exc:
        raise QdrantDependencyError(
            "langchain-qdrant is required for rag.retrieve. "
            "Install it with `pip install langchain-qdrant`."
        ) from exc

    return _QdrantVectorStore


def _build_vectorstore_with_compat(
    *,
    vectorstore_class: type[Any],
    collection: str,
    embeddings: OllamaEmbeddings,
    url: str,
    client: QdrantClient,
) -> QdrantVectorStore:
    if hasattr(vectorstore_class, "from_existing_collection"):
        creators = [
            {"collection_name": collection, "embedding": embeddings, "url": url},
            {"collection_name": collection, "embeddings": embeddings, "url": url},
            {"collection_name": collection, "embedding": embeddings, "client": client},
            {"collection_name": collection, "embeddings": embeddings, "client": client},
        ]
        for kwargs in creators:
            try:
                return vectorstore_class.from_existing_collection(**kwargs)
            except TypeError:
                continue

    constructors = [
        {"collection_name": collection, "embedding": embeddings, "url": url},
        {"collection_name": collection, "embeddings": embeddings, "url": url},
        {"collection_name": collection, "embedding": embeddings, "client": client},
        {"collection_name": collection, "embeddings": embeddings, "client": client},
    ]
    for kwargs in constructors:
        try:
            return vectorstore_class(**kwargs)
        except TypeError:
            continue

    raise RuntimeError(
        "Unable to initialize QdrantVectorStore with this langchain-qdrant version. "
        "Please update langchain-qdrant to a recent release."
    )


def get_vectorstore(
    collection: str,
    embeddings: OllamaEmbeddings,
    url: str,
) -> QdrantVectorStore:
    resolved_collection = str(collection or os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)).strip()
    if not resolved_collection:
        raise ValueError("Qdrant collection is required (input.collection or QDRANT_COLLECTION).")

    resolved_url = str(url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)).strip()
    if not resolved_url:
        raise ValueError(
            "Qdrant URL is required (input.qdrant_url, context.qdrant_url, or QDRANT_URL)."
        )

    vectorstore_class = _resolve_qdrant_vectorstore_class()

    try:
        client = QdrantClient(url=resolved_url)
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to connect to Qdrant at '{resolved_url}'. "
            "Ensure embedded Qdrant is running and API runtime config is set "
            "or provide qdrant_url explicitly."
        ) from exc

    try:
        client.get_collection(collection_name=resolved_collection)
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant collection '{resolved_collection}' was not found at '{resolved_url}'. "
            "Ingest data into the collection before using rag.retrieve."
        ) from exc

    return _build_vectorstore_with_compat(
        vectorstore_class=vectorstore_class,
        collection=resolved_collection,
        embeddings=embeddings,
        url=resolved_url,
        client=client,
    )
