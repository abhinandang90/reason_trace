"""
conftest.py — Shared pytest fixtures and configuration for reasontrace tests.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Embedder mock — allows tests to run without sentence-transformers installed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    """Replace sentence-transformers with a deterministic stub.

    See DECISIONS.md — "Architectural Decision: Deterministic Embedding Stub in Tests"
    for the rationale behind this approach.

    The stub produces reproducible random-unit embeddings keyed on the hash
    of the input text, so semantically identical texts get the same embedding
    and different texts get different embeddings.

    This fixture is autouse so ALL tests run without needing the ML library.
    """
    import reasontrace.embedder as embedder_module

    def _fake_embed_texts(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((1, 384), dtype=np.float32)
        results = []
        for text in texts:
            rng = np.random.default_rng(hash(text) & 0xFFFFFFFF)
            vec = rng.standard_normal(384).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            results.append(vec)
        return np.array(results, dtype=np.float32)

    def _fake_embed_session_reasoning(reasoning_checkpoints: list[dict]) -> np.ndarray:
        texts = [cp.get("content", "") for cp in reasoning_checkpoints if cp.get("content")]
        if not texts:
            return np.zeros(384, dtype=np.float32)
        embeddings = _fake_embed_texts(texts)
        return embeddings.mean(axis=0)

    monkeypatch.setattr(embedder_module, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(embedder_module, "embed_session_reasoning", _fake_embed_session_reasoning)

    # Also patch wherever it's imported
    import reasontrace.baseline as baseline_module
    monkeypatch.setattr(baseline_module, "embed_session_reasoning", _fake_embed_session_reasoning)

    import reasontrace.scorer as scorer_module
    monkeypatch.setattr(scorer_module, "embed_session_reasoning", _fake_embed_session_reasoning)

    yield


# ---------------------------------------------------------------------------
# Isolated storage
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_reasontrace_dir_env(monkeypatch, tmp_path):
    """Ensure REASONTRACE_DIR always points to a fresh temp directory.

    This autouse fixture runs for every test that does not already have
    its own isolated_storage fixture, providing belt-and-suspenders
    isolation across the entire test suite.

    Individual test modules may override this with their own monkeypatch
    of REASONTRACE_DIR — that's fine, the last monkeypatch wins.
    """
    monkeypatch.setenv("REASONTRACE_DIR", str(tmp_path))
    yield
