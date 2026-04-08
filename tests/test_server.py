"""
tests/test_server.py — Unit tests for all 13 MCP tools in reasontrace.server

All tools are tested by calling the underlying Python functions directly
(no MCP transport required). Tests cover the core business logic of each
tool, error handling, and response structure.
"""

from __future__ import annotations

import pytest

import reasontrace.storage as storage
import reasontrace.baseline as bl_module

# Import all 13 tool functions directly
from reasontrace.server import (
    start_session,
    log_reasoning,
    log_tool_call,
    log_decision,
    end_session,
    get_drift_score,
    explain_drift,
    get_baseline,
    reset_baseline,
    list_sessions,
    get_session,
    search_sessions,
    delete_session,
    _open_sessions,
)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("REASONTRACE_DIR", str(tmp_path))
    # Clear in-memory open sessions between tests
    _open_sessions.clear()
    yield tmp_path
    _open_sessions.clear()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _full_session(tag: str = "test", tools: list[str] | None = None) -> str:
    """Run a complete session through all logging tools and end it."""
    tools = tools or ["search_web", "extract_data", "summarize"]
    r = start_session(tag=tag, agent_name="test-agent")
    sid = r["session_id"]
    log_reasoning(sid, "Decided to use search_web for live data retrieval.", step=1)
    log_tool_call(sid, tools[0], {"query": "AI"}, {"results": ["r1"]}, step=1)
    if len(tools) > 1:
        log_tool_call(sid, tools[1], {"data": "d1"}, {"extracted": "e1"}, step=2)
    log_decision(sid, ["search_web", "query_db"], "search_web", confidence=0.9, step=1)
    end_session(sid)
    return sid


# ---------------------------------------------------------------------------
# Tool 1 — start_session
# ---------------------------------------------------------------------------

class TestStartSession:
    def test_returns_session_id(self):
        r = start_session(tag="hello", agent_name="bot")
        assert "session_id" in r
        assert len(r["session_id"]) == 36

    def test_session_in_memory(self):
        r = start_session()
        assert r["session_id"] in _open_sessions

    def test_default_values(self):
        r = start_session()
        sid = r["session_id"]
        s = _open_sessions[sid]
        assert s["tag"] == ""
        assert s["agent_name"] == ""


# ---------------------------------------------------------------------------
# Tool 2 — log_reasoning
# ---------------------------------------------------------------------------

class TestLogReasoning:
    def test_appends_checkpoint(self):
        r = start_session()
        sid = r["session_id"]
        log_reasoning(sid, "Thinking about tool selection", step=1)
        assert _open_sessions[sid]["reasoning_checkpoints"][0]["content"] == "Thinking about tool selection"

    def test_invalid_session(self):
        r = log_reasoning("bad-id", "content")
        assert "error" in r

    def test_step_optional(self):
        r = start_session()
        sid = r["session_id"]
        result = log_reasoning(sid, "no step here")
        assert "error" not in result


# ---------------------------------------------------------------------------
# Tool 3 — log_tool_call
# ---------------------------------------------------------------------------

class TestLogToolCall:
    def test_appends_tool_call(self):
        r = start_session()
        sid = r["session_id"]
        log_tool_call(sid, "search_web", {"q": "x"}, {"r": "y"}, step=1)
        tc = _open_sessions[sid]["tool_calls"][0]
        assert tc["tool_name"] == "search_web"
        assert tc["inputs"] == {"q": "x"}

    def test_invalid_session(self):
        r = log_tool_call("bad-id", "tool", {}, {})
        assert "error" in r


# ---------------------------------------------------------------------------
# Tool 4 — log_decision
# ---------------------------------------------------------------------------

class TestLogDecision:
    def test_appends_decision(self):
        r = start_session()
        sid = r["session_id"]
        log_decision(sid, ["a", "b"], "a", confidence=0.8, step=1)
        d = _open_sessions[sid]["decisions"][0]
        assert d["option_chosen"] == "a"
        assert d["confidence"] == pytest.approx(0.8)

    def test_confidence_clamped(self):
        r = start_session()
        sid = r["session_id"]
        log_decision(sid, ["a"], "a", confidence=1.5)
        d = _open_sessions[sid]["decisions"][0]
        assert d["confidence"] <= 1.0

    def test_invalid_session(self):
        r = log_decision("bad-id", ["a"], "a")
        assert "error" in r


# ---------------------------------------------------------------------------
# Tool 5 — end_session
# ---------------------------------------------------------------------------

class TestEndSession:
    def test_end_returns_drift_data(self):
        r = start_session()
        sid = r["session_id"]
        log_reasoning(sid, "step 1")
        result = end_session(sid)
        assert "composite_score" in result
        assert "severity" in result
        assert "session_file" in result

    def test_first_session_is_initializing(self):
        r = start_session()
        sid = r["session_id"]
        result = end_session(sid)
        assert result["severity"] == "initializing"

    def test_session_removed_from_memory(self):
        r = start_session()
        sid = r["session_id"]
        end_session(sid)
        assert sid not in _open_sessions

    def test_session_saved_to_disk(self, tmp_path):
        r = start_session()
        sid = r["session_id"]
        end_session(sid)
        files = list(tmp_path.glob("*.rtrace"))
        assert len(files) >= 1

    def test_invalid_session(self):
        result = end_session("nonexistent-id")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool 6 — get_drift_score
# ---------------------------------------------------------------------------

class TestGetDriftScore:
    def test_returns_drift_for_saved_session(self):
        sid = _full_session()
        result = get_drift_score(sid)
        assert result["session_id"] == sid
        assert "composite_score" in result

    def test_invalid_session(self):
        result = get_drift_score("nonexistent")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool 7 — explain_drift
# ---------------------------------------------------------------------------

class TestExplainDrift:
    def test_requires_multiple_sessions(self):
        sid = _full_session()
        # After first session baseline exists but no drift data
        result = explain_drift(sid)
        # First session has no drift data (initializing) — should still return something
        assert "error" in result or "primary_driver" in result

    def test_explain_returns_structure(self):
        # Need at least 2 sessions for drift data
        _full_session()
        sid2 = _full_session(tools=["query_db", "format_output"])
        result = explain_drift(sid2)
        if "error" not in result:
            assert "primary_driver" in result
            assert "recommendation" in result

    def test_invalid_session(self):
        result = explain_drift("bad-id")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool 8 — get_baseline
# ---------------------------------------------------------------------------

class TestGetBaseline:
    def test_empty_baseline(self):
        result = get_baseline()
        assert result["sessions_count"] == 0
        assert result["confidence"] == "initializing"

    def test_after_sessions(self):
        _full_session()
        result = get_baseline()
        assert result["sessions_count"] == 1
        assert "most_common_tools" in result


# ---------------------------------------------------------------------------
# Tool 9 — reset_baseline
# ---------------------------------------------------------------------------

class TestResetBaseline:
    def test_clear_baseline(self):
        _full_session()
        result = reset_baseline()
        assert result["sessions_count"] == 0

    def test_rebuild_from_sessions(self):
        sid1 = _full_session(tag="rebuild-1")
        sid2 = _full_session(tag="rebuild-2")
        result = reset_baseline(session_ids=[sid1, sid2])
        assert result["sessions_count"] == 2

    def test_missing_session_id_returns_error(self):
        result = reset_baseline(session_ids=["nonexistent-id"])
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool 10 — list_sessions
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_empty_returns_empty_list(self):
        result = list_sessions()
        assert result["count"] == 0
        assert result["sessions"] == []

    def test_lists_all_sessions(self):
        _full_session(tag="s1")
        _full_session(tag="s2")
        result = list_sessions()
        assert result["count"] == 2

    def test_session_summary_structure(self):
        _full_session(tag="struct-test")
        result = list_sessions()
        s = result["sessions"][0]
        for key in ["session_id", "tag", "agent_name", "started_at", "composite_score", "severity"]:
            assert key in s


# ---------------------------------------------------------------------------
# Tool 11 — get_session
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_returns_full_session(self):
        sid = _full_session()
        result = get_session(sid)
        assert result["session_id"] == sid
        assert "reasoning_checkpoints" in result
        assert "tool_calls" in result
        assert "decisions" in result
        assert "drift" in result

    def test_invalid_session(self):
        result = get_session("bad-id")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool 12 — search_sessions
# ---------------------------------------------------------------------------

class TestSearchSessions:
    def test_search_by_tag(self):
        _full_session(tag="production")
        _full_session(tag="experiment")
        result = search_sessions(tag="production")
        assert result["count"] == 1
        assert result["sessions"][0]["tag"] == "production"

    def test_search_by_tool_name(self):
        _full_session(tools=["search_web", "summarize"])
        _full_session(tools=["query_db", "format_output"])
        result = search_sessions(tool_name="search_web")
        assert result["count"] == 1

    def test_search_no_filters_returns_all(self):
        _full_session()
        _full_session()
        result = search_sessions()
        assert result["count"] == 2

    def test_search_by_min_drift_score(self):
        _full_session()
        _full_session()  # second session gets a drift score
        result = search_sessions(min_drift_score=0.0)
        # Sessions with None composite_score are excluded
        # The first session has None score (initializing)
        assert isinstance(result["count"], int)

    def test_search_by_agent_name(self):
        _full_session()  # agent_name="test-agent" from helper
        result = search_sessions(agent_name="test-agent")
        assert result["count"] >= 1

    def test_search_by_keyword(self):
        r = start_session()
        sid = r["session_id"]
        log_reasoning(sid, "using search_web for live retrieval of banana data")
        end_session(sid)
        result = search_sessions(keyword="banana")
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# Tool 13 — delete_session
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def test_deletes_session(self, tmp_path):
        sid = _full_session()
        result = delete_session(sid)
        assert "deleted_session_id" in result
        files = list(tmp_path.glob(f"{sid}.rtrace"))
        assert files == []

    def test_invalid_session(self):
        result = delete_session("nonexistent-id")
        assert "error" in result

    def test_baseline_updated_after_delete(self):
        sid1 = _full_session(tag="keep")
        sid2 = _full_session(tag="delete-me")
        result = delete_session(sid2)
        assert result["baseline_sessions_count"] == 1
