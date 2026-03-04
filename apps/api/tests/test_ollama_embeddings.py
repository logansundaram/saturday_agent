from __future__ import annotations

import unittest
from unittest import mock

try:
    from saturday_agent.rag import embeddings

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dependency/environment gated
    embeddings = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Embeddings deps unavailable: {_IMPORT_ERROR}")
class OllamaEmbeddingResolutionTests(unittest.TestCase):
    def test_resolves_single_installed_tagged_variant(self) -> None:
        with mock.patch.object(
            embeddings,
            "_fetch_ollama_model_names",
            return_value=["nomic-embed-text:v1.5"],
        ):
            resolved = embeddings._resolve_installed_ollama_model(
                "nomic-embed-text",
                "http://localhost:11434",
            )

        self.assertEqual(resolved, "nomic-embed-text:v1.5")

    def test_keeps_requested_name_when_multiple_tagged_variants_exist(self) -> None:
        with mock.patch.object(
            embeddings,
            "_fetch_ollama_model_names",
            return_value=["nomic-embed-text:v1.5", "nomic-embed-text:latest"],
        ):
            resolved = embeddings._resolve_installed_ollama_model(
                "nomic-embed-text",
                "http://localhost:11434",
            )

        self.assertEqual(resolved, "nomic-embed-text")

    def test_get_ollama_embeddings_uses_resolved_installed_model(self) -> None:
        factory = mock.Mock(return_value=object())

        with (
            mock.patch.object(
                embeddings,
                "_fetch_ollama_model_names",
                return_value=["nomic-embed-text:v1.5"],
            ),
            mock.patch.object(
                embeddings,
                "_resolve_embeddings_class",
                return_value=factory,
            ),
        ):
            embeddings.get_ollama_embeddings(
                "nomic-embed-text",
                "http://localhost:11434",
            )

        factory.assert_called_once_with(
            model="nomic-embed-text:v1.5",
            base_url="http://localhost:11434",
        )


if __name__ == "__main__":
    unittest.main()
