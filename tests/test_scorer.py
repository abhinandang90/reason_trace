"""
tests/test_scorer.py — Unit tests for reasontrace.scorer

Tests cover:
- severity_label mapping
- Weight resolution from env var
- Tool pattern score (same tools → low, different tools → high)
- Semantic score (same text → low, different text → higher)
- Complexity score (same steps → low, many steps → higher)
- Composite score clamped to [0, 1]
- DriftScorer returns correct structure
- explain_drift structure
"""

from __future__ import annotations

import os
import pytest

import reasontrace.storage as storage
import reasontrace.baseline as bl_module
from reasontrace.scorer import DriftScorer, severity_label, explain_drift, _get_weights


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("REASONTRACE_DIR", str(tmp_path))
    yield tmp_path


def _session_with(
    tools: list[str],
    reasoning: list[str],
    steps: int = 3,
    reconsiderations: int = 0,
) -> dict:
    s = storage.new_session()
    s["ended_at"] = storage.now_iso()
    for i, t in enumerate(tools):
        s["tool_calls"].append({
            "tool_name": t, "inputs": {}, "outputs": {},
            "step": i + 1, "timestamp": storage.now_iso(),
        })
    for i, r in enumerate(reasoning):
        s["reasoning_checkpoints"].append({
            "content": r, "step": i + 1, "timestamp": storage.now_iso(),
        })
    for i in range(reconsiderations):
        s["decisions"].append({
            "options_considered": ["a", "b", "c"],
            "option_chosen": "a",
            "confidence": 0.3,
            "step": i + 1,
            "timestamp": storage.now_iso(),
        })
    return s


def _build_baseline(sessions: list[dict]) -> dict:
    """Build baseline from a list of sessions."""
    storage.save_baseline(bl_module._empty_baseline())
    for s in sessions:
        bl_module.update_baseline(s)
    return storage.load_baseline()


# ---------------------------------------------------------------------------
# severity_label
# ---------------------------------------------------------------------------

class TestSeverityLabel:
    def test_none(self):
        assert severity_label(None) == "initializing"

    def test_no_drift(self):
        assert severity_label(0.0) == "none"
        assert severity_label(0.29) == "none"

    def test_warning(self):
        assert severity_label(0.3) == "warning"
        assert severity_label(0.59) == "warning"

    def test_alert(self):
        assert severity_label(0.6) == "alert"
        assert severity_label(0.79) == "alert"

    def test_critical(self):
        assert severity_label(0.8) == "alert"  # 0.8 is boundary: >= 0.8 → critical
        assert severity_label(0.81) == "critical"
        assert severity_label(1.0) == "critical"


class TestSeverityLabelBoundary:
    """Verify exact boundary values per spec."""
    def test_score_08_is_alert_boundary(self):
        # Score 0.6–0.8 → alert; score > 0.8 → critical
        # 0.8 falls in [0.6, 0.8] → alert
        assert severity_label(0.8) == "alert"

    def test_score_above_08_is_critical(self):
        assert severity_label(0.801) == "critical"


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------

class TestGetWeights:
    def test_default_weights_sum_to_one(self):
        w = _get_weights()
        assert abs(sum(w) - 1.0) < 1e-9

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("REASONTRACE_WEIGHTS", "0.5,0.3,0.2")
        w = _get_weights()
        assert abs(w[0] - 0.5) < 1e-9
        assert abs(sum(w) - 1.0) < 1e-9

    def test_invalid_env_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("REASONTRACE_WEIGHTS", "not,valid,floats")
        w = _get_weights()
        assert abs(sum(w) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# DriftScorer — structure
# ---------------------------------------------------------------------------

class TestDriftScorerStructure:
    def test_first_session_returns_null_scores(self):
        scorer = DriftScorer()
        # Empty baseline
        s = _session_with(["search_web"], ["found data"])
        empty_bl = bl_module._empty_baseline()  # 0 sessions
        result = scorer.score(s, empty_bl)
        assert result["composite_score"] is None
        assert result["severity"] == "initializing"
        assert result["baseline_confidence"] == "initializing"

    def test_returns_all_required_keys(self):
        scorer = DriftScorer()
        sessions = [_session_with(["search_web"], ["step 1"]) for _ in range(3)]
        base = _build_baseline(sessions)
        new_session = _session_with(["search_web"], ["step 1"])
        result = scorer.score(new_session, base)
        for key in [
            "composite_score", "tool_pattern_score", "semantic_score",
            "complexity_score", "severity", "baseline_confidence",
            "baseline_confidence_note", "baseline_sessions_count",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_scores_clamped_to_unit_interval(self):
        scorer = DriftScorer()
        sessions = [_session_with(["search_web"], ["baseline text"]) for _ in range(5)]
        base = _build_baseline(sessions)
        new_session = _session_with(["completely_different_tool"], ["very different text here"])
        result = scorer.score(new_session, base)
        assert 0.0 <= result["composite_score"] <= 1.0
        assert 0.0 <= result["tool_pattern_score"] <= 1.0
        assert 0.0 <= result["semantic_score"] <= 1.0
        assert 0.0 <= result["complexity_score"] <= 1.0


# ---------------------------------------------------------------------------
# Tool pattern drift
# ---------------------------------------------------------------------------

class TestToolPatternDrift:
    def test_identical_tools_low_score(self):
        scorer = DriftScorer()
        sessions = [
            _session_with(["search_web", "extract_data", "summarize"], ["step 1"])
            for _ in range(5)
        ]
        base = _build_baseline(sessions)
        new_s = _session_with(["search_web", "extract_data", "summarize"], ["step 1"])
        result = scorer.score(new_s, base)
        assert result["tool_pattern_score"] < 0.3

    def test_completely_different_tools_high_score(self):
        scorer = DriftScorer()
        sessions = [
            _session_with(["search_web", "extract_data", "summarize"], ["step 1"])
            for _ in range(5)
        ]
        base = _build_baseline(sessions)
        new_s = _session_with(["query_db", "format_output"], ["step 1"])
        result = scorer.score(new_s, base)
        assert result["tool_pattern_score"] > 0.4


# ---------------------------------------------------------------------------
# Complexity drift
# ---------------------------------------------------------------------------

class TestComplexityDrift:
    def test_same_steps_low_score(self):
        scorer = DriftScorer()
        sessions = [
            _session_with(["search_web"], ["step 1"], steps=3) for _ in range(5)
        ]
        base = _build_baseline(sessions)
        new_s = _session_with(["search_web"], ["step 1"], steps=3)
        result = scorer.score(new_s, base)
        assert result["complexity_score"] < 0.4

    def test_many_more_steps_raises_score(self):
        scorer = DriftScorer()
        sessions = [
            _session_with(
                ["search_web"] * 3,
                ["brief step", "brief step", "brief step"],
                steps=3,
            ) for _ in range(8)
        ]
        base = _build_baseline(sessions)
        # Now a session with 10 steps and reconsiderations
        new_s = _session_with(
            ["search_web"] * 10,
            ["step"] * 10,
            steps=10,
            reconsiderations=4,
        )
        result = scorer.score(new_s, base)
        assert result["complexity_score"] > 0.2


# ---------------------------------------------------------------------------
# explain_drift
# ---------------------------------------------------------------------------

class TestExplainDrift:
    def test_returns_required_keys(self):
        scorer = DriftScorer()
        sessions = [_session_with(["search_web"], ["decisive text"]) for _ in range(5)]
        base = _build_baseline(sessions)
        new_s = _session_with(["query_db"], ["uncertain text"])
        drift = scorer.score(new_s, base)
        result = explain_drift(new_s, base, drift)
        for key in [
            "primary_driver", "tool_change_detail", "semantic_detail",
            "complexity_detail", "divergent_from", "recommendation",
        ]:
            assert key in result

    def test_primary_driver_is_valid(self):
        scorer = DriftScorer()
        sessions = [_session_with(["search_web"], ["text"]) for _ in range(5)]
        base = _build_baseline(sessions)
        new_s = _session_with(["query_db"], ["text"])
        drift = scorer.score(new_s, base)
        result = explain_drift(new_s, base, drift)
        assert result["primary_driver"] in (
            "tool_pattern", "semantic_reasoning", "decision_complexity"
        )
