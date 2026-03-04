from __future__ import annotations

import unittest
import uuid
from unittest import mock

try:
    from saturday_agent.rag import qdrant_store
    from saturday_agent.rag.ingest import qdrant_index
    from saturday_agent.tools import rag_delete_doc

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dependency/environment gated
    qdrant_store = None  # type: ignore[assignment]
    qdrant_index = None  # type: ignore[assignment]
    rag_delete_doc = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Qdrant deps unavailable: {_IMPORT_ERROR}")
class QdrantBootstrapTests(unittest.TestCase):
    def test_missing_collection_raises_specific_error(self) -> None:
        mock_client = mock.Mock()
        mock_client.get_collections.return_value = {}
        mock_client.get_collection.side_effect = Exception("missing")

        with (
            mock.patch.object(qdrant_store, "QdrantClient", return_value=mock_client),
            mock.patch.object(qdrant_store, "_resolve_qdrant_vectorstore_class", return_value=object()),
        ):
            with self.assertRaises(qdrant_store.QdrantCollectionMissingError) as exc_info:
                qdrant_store.get_vectorstore(
                    "saturday_docs",
                    embeddings=object(),
                    url="http://127.0.0.1:6333",
                )

        self.assertIn("saturday_docs", str(exc_info.exception))

    def test_first_ingest_bootstraps_collection(self) -> None:
        embeddings = object()

        with (
            mock.patch.object(qdrant_index, "get_ollama_embeddings", return_value=embeddings),
            mock.patch.object(
                qdrant_index,
                "get_vectorstore",
                side_effect=qdrant_store.QdrantCollectionMissingError("missing"),
            ),
            mock.patch.object(qdrant_index, "create_vectorstore_from_texts") as create_mock,
        ):
            result = qdrant_index.index_chunks_to_qdrant(
                ["alpha", "beta"],
                doc_id="doc-123",
                filename="notes.pdf",
                sha256="abc123",
                source_path="/tmp/notes.pdf",
                collection="saturday_docs",
                embeddings_model="nomic-embed-text",
                qdrant_url="http://127.0.0.1:6333",
            )

        self.assertEqual(result, {"ok": True, "added": 2})
        create_mock.assert_called_once()
        self.assertEqual(create_mock.call_args.args[0], ["alpha", "beta"])
        self.assertEqual(create_mock.call_args.kwargs["collection"], "saturday_docs")
        self.assertIs(create_mock.call_args.kwargs["embeddings"], embeddings)
        self.assertEqual(create_mock.call_args.kwargs["url"], "http://127.0.0.1:6333")
        ids = create_mock.call_args.kwargs["ids"]
        self.assertEqual(
            ids,
            [
                str(uuid.uuid5(uuid.NAMESPACE_URL, "doc-123:0")),
                str(uuid.uuid5(uuid.NAMESPACE_URL, "doc-123:1")),
            ],
        )
        for point_id in ids:
            parsed = uuid.UUID(str(point_id))
            self.assertEqual(str(parsed), str(point_id))
        self.assertEqual(
            create_mock.call_args.kwargs["metadatas"],
            [
                {
                    "doc_id": "doc-123",
                    "chunk_id": 0,
                    "chunk_index": 0,
                    "filename": "notes.pdf",
                    "sha256": "abc123",
                    "source_path": "/tmp/notes.pdf",
                    "source": "/tmp/notes.pdf",
                },
                {
                    "doc_id": "doc-123",
                    "chunk_id": 1,
                    "chunk_index": 1,
                    "filename": "notes.pdf",
                    "sha256": "abc123",
                    "source_path": "/tmp/notes.pdf",
                    "source": "/tmp/notes.pdf",
                },
            ],
        )

    def test_delete_skips_qdrant_when_collection_is_missing(self) -> None:
        mock_client = mock.Mock()
        mock_client.collection_exists.return_value = False

        with mock.patch.object(rag_delete_doc, "QdrantClient", return_value=mock_client):
            rag_delete_doc._delete_from_qdrant(
                doc_id="doc-123",
                collection="saturday_docs",
                qdrant_url="http://127.0.0.1:6333",
            )

        mock_client.delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
