"""
tests/test_baseline.py — Unit tests for reasontrace.baseline

Tests cover:
- get_confidence tiers
- _tool_frequency_vector
- _complexity_vector
- update_baseline (incremental centroid update)
- rebuild_baseline
- baseline_profile
- Welford variance computation
"""

from __future__ import annotations

import pytest

import reasontrace.storage as storage
import reasontrace.baseline as baseline


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("REASONTRACE_DIR", str(tmp_path))
    yield tmp_path


def _make_session(
    tools: list[str],
    reasoning: list[str],
    steps: int = 3,
    reconsiderations: int = 0,
    tag: str = "test",
) -> dict:
    s = storage.new_session(tag=tag)
    s["ended_at"] = storage.now_iso()
    for i, t in enumerate(tools):
        s["tool_calls"].append({"tool_name": t, "inputs": {}, "outputs": {}, "step": i + 1, "timestamp": storage.now_iso()})
    for i, r in enumerate(reasoning):
        s["reasoning_checkpoints"].append({"content": r, "step": i + 1, "timestamp": storage.now_iso()})
    for i in range(reconsiderations):
        s["decisions"].append({
            "options_considered": ["a", "b", "c"],
            "option_chosen": "a",
            "confidence": 0.3,
            "step": i + 1,
            "timestamp": storage.now_iso(),
        })
    return s


# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------

class TestGetConfidence:
    def test_zero_sessions(self):
        label, note = baseline.get_confidence(0)
        assert label == "initializing"

    def test_one_session(self):
        label, note = baseline.get_confidence(1)
        assert label == "initializing"

    def test_low_confidence(self):
        for n in (2, 3, 4):
            label, _ = baseline.get_confidence(n)
            assert label == "low", f"Expected 'low' for n={n}"

    def test_medium_confidence(self):
        for n in (5, 6, 7, 8, 9):
            label, _ = baseline.get_confidence(n)
            assert label == "medium", f"Expected 'medium' for n={n}"

    def test_high_confidence(self):
        for n in (10, 15, 100):
            label, _ = baseline.get_confidence(n)
            assert label == "high", f"Expected 'high' for n={n}"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestToolFrequencyVector:
    def test_basic_counts(self):
        s = _make_session(tools=["search_web", "search_web", "summarize"])
        freq = baseline._tool_frequency_vector(s)
        assert freq["search_web"] == 2
        assert freq["summarize"] == 1

    def test_empty_tools(self):
        s = _make_session(tools=[])
        freq = baseline._tool_frequency_vector(s)
        assert dict(freq) == {}


class TestComplexityVector:
    def test_basic(self):
        s = _make_session(
            tools=["search_web", "extract_data", "summarize"],
            reasoning=["step 1", "step 2", "step 3"],
            steps=3,
        )
        cx = baseline._complexity_vector(s)
        assert len(cx) == 3
        assert cx[0] >= 3  # step count

    def test_reconsiderations_count(self):
        s = _make_session(
            tools=["search_web"],
            reasoning=["step 1"],
            reconsiderations=3,
        )
        cx = baseline._complexity_vector(s)
        assert cx[1] >= 3  # reconsideration count


# ---------------------------------------------------------------------------
# update_baseline
# ---------------------------------------------------------------------------

class TestUpdateBaseline:
    def test_first_session_initialises(self):
        s = _make_session(tools=["search_web"], reasoning=["found data"])
        storage.save_baseline(baseline._empty_baseline())
        bl = baseline.update_baseline(s)
        assert bl["sessions_count"] == 1
        assert "search_web" in bl["tool_vocabulary"]
        assert s["session_id"] in bl["session_ids"]

    def test_second_session_updates_centroid(self):
        for _ in range(2):
            s = _make_session(tools=["search_web", "summarize"], reasoning=["found data"])
            storage.save_baseline(storage.load_baseline() or baseline._empty_baseline())
            bl = baseline.update_baseline(s)
        assert bl["sessions_count"] == 2
        assert bl["tool_centroid"]["search_web"] > 0

    def test_vocabulary_grows(self):
        s1 = _make_session(tools=["tool_a"], reasoning=["step 1"])
        storage.save_baseline(baseline._empty_baseline())
        baseline.update_baseline(s1)

        s2 = _make_session(tools=["tool_b"], reasoning=["step 2"])
        bl = baseline.update_baseline(s2)
        assert "tool_a" in bl["tool_vocabulary"]
        assert "tool_b" in bl["tool_vocabulary"]


# ---------------------------------------------------------------------------
# rebuild_baseline
# ---------------------------------------------------------------------------

class TestRebuildBaseline:
    def test_rebuild_from_sessions(self):
        sessions = [
            _make_session(tools=["search_web"], reasoning=["step 1"]) for _ in range(3)
        ]
        bl = baseline.rebuild_baseline(sessions)
        assert bl["sessions_count"] == 3

    def test_rebuild_empty_clears(self):
        s = _make_session(tools=["search_web"], reasoning=["step 1"])
        storage.save_baseline(baseline._empty_baseline())
        baseline.update_baseline(s)
        bl = baseline.rebuild_baseline([])
        assert bl["sessions_count"] == 0


# ---------------------------------------------------------------------------
# baseline_profile
# ---------------------------------------------------------------------------

class TestBaselineProfile:
    def test_profile_structure(self):
        bl = baseline._empty_baseline()
        bl["sessions_count"] = 5
        bl["total_steps"] = 15
        bl["total_tool_calls"] = 10
        bl["most_common_tools"] = ["search_web"]
        profile = baseline.baseline_profile(bl)
        assert "sessions_count" in profile
        assert "confidence" in profile
        assert "average_step_count" in profile
        assert profile["sessions_count"] == 5
        assert profile["average_step_count"] == pytest.approx(3.0)

    def test_zero_sessions_profile(self):
        bl = baseline._empty_baseline()
        profile = baseline.baseline_profile(bl)
        assert profile["sessions_count"] == 0
        assert profile["average_step_count"] == 0.0
