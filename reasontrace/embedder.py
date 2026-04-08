"""
embedder.py — Local sentence-transformer embeddings for semantic drift scoring.

Uses sentence-transformers/all-MiniLM-L6-v2, downloaded once and cached
locally by the sentence-transformers library. No API calls, no external
accounts required.

The model is loaded lazily on first use so that import of the package
does not block on model download in environments where embeddings are
not needed (e.g., during installer-only use).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: "SentenceTransformer | None" = None
_model_lock = threading.Lock()


def _get_model() -> "SentenceTransformer":
    """Return the shared SentenceTransformer model, loading it if necessary.

    Thread-safe via a module-level lock.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover
                    raise ImportError(
                        "sentence-transformers is required for semantic drift scoring. "
                        "Install it with: pip install sentence-transformers"
                    ) from exc
                _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public embedding API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of text strings into a 2-D float32 array.

    Returns an array of shape (N, embedding_dim) where N = len(texts).
    Returns a zero-vector of standard embedding dim (384) when the list
    is empty, to allow downstream code to remain vectorised.

    Parameters
    ----------
    texts:
        Free-form strings to embed (e.g., reasoning checkpoint content).
    """
    if not texts:
        return np.zeros((1, 384), dtype=np.float32)
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def embed_session_reasoning(reasoning_checkpoints: list[dict]) -> np.ndarray:
    """Average-embed all reasoning checkpoint texts for a session.

    Parameters
    ----------
    reasoning_checkpoints:
        List of checkpoint dicts, each with at least a ``content`` key.

    Returns
    -------
    np.ndarray
        A 1-D array of shape (embedding_dim,) representing the session's
        average semantic position.
    """
    texts = [cp.get("content", "") for cp in reasoning_checkpoints if cp.get("content")]
    if not texts:
        return np.zeros(384, dtype=np.float32)
    embeddings = embed_texts(texts)
    return embeddings.mean(axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two 1-D vectors.

    Returns a float in [0.0, 1.0].  Returns 0.0 if either vector is
    all-zeros (no drift can be measured against an empty baseline).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [-1, 1] before converting to distance
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity
