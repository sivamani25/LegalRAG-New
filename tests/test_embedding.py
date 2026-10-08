import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from legalrag.retrieval.embedding import SentenceTransformerEmbedder, Embedder

def test_embedder_initialization():
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        MockST.return_value = mock_instance
        
        embedder = SentenceTransformerEmbedder("nomic-embed-text")
        
        MockST.assert_called_once_with("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        assert embedder.dimension == 768
        assert embedder.model_name == "nomic-embed-text"

def test_embedder_unknown_model():
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        embedder = SentenceTransformerEmbedder("my-custom-model/v1")
        MockST.assert_called_once_with("my-custom-model/v1", trust_remote_code=True)

def test_embedder_from_config():
    cfg = {"embedding_model": "nomic-embed-text"}
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        embedder = Embedder.from_config(cfg)
        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert embedder.model_name == "nomic-embed-text"

def test_embedder_encode_normalization_and_prefix():
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[1.0, 2.0, 3.0]])
        MockST.return_value = mock_instance
        
        embedder = SentenceTransformerEmbedder("test-model")
        
        texts = ["hello", "world"]
        
        # Test with normalize=True, no prefix, default batch_size
        res = embedder.encode(texts, normalize=True)
        mock_instance.encode.assert_called_with(
            texts, normalize_embeddings=True, batch_size=32
        )
        
        # Test with normalize=False, with prefix, default batch_size
        res = embedder.encode(texts, normalize=False, prompt_prefix="prefix: ")
        mock_instance.encode.assert_called_with(
            ["prefix: hello", "prefix: world"], normalize_embeddings=False, batch_size=32
        )


def test_embedder_encode_batch_size_forwarded():
    """batch_size kwarg must reach sentence_transformers model.encode() exactly."""
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((3, 768), dtype=np.float32)
        MockST.return_value = mock_instance

        embedder = SentenceTransformerEmbedder("test-model")
        texts = ["a", "b", "c"]

        # Pass a non-default batch_size and confirm it arrives at the underlying call.
        embedder.encode(texts, normalize=True, batch_size=8)
        mock_instance.encode.assert_called_once_with(
            texts, normalize_embeddings=True, batch_size=8
        )

        mock_instance.encode.reset_mock()

        # Explicit batch_size=1 (stress test, smallest meaningful value).
        embedder.encode(texts, normalize=False, batch_size=1)
        mock_instance.encode.assert_called_once_with(
            texts, normalize_embeddings=False, batch_size=1
        )


def test_embedder_encode_batch_size_default_is_32():
    """Omitting batch_size must default to 32 (matches config.yaml default)."""
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((1, 768), dtype=np.float32)
        MockST.return_value = mock_instance

        embedder = SentenceTransformerEmbedder("test-model")
        embedder.encode(["text"])

        _, kwargs = mock_instance.encode.call_args
        assert kwargs["batch_size"] == 32, (
            f"Default batch_size must be 32, got {kwargs['batch_size']}"
        )

