"""
tests/test_tracer.py — Unit tests for reasontrace.tracer

Tests cover:
- Tracer.start() returns a session ID
- Tracer.end() closes session, computes drift, saves file
- Tracer.log_reasoning() appends checkpoints
- Tracer.log_tool_call() appends tool calls
- Tracer.log_decision() appends decisions
- @tracer.watch decorator captures function calls
- Tracer.attach_anthropic() hooks messages.create
- Thread-local session context
- Error on missing session
"""

from __future__ import annotations

import pytest

import reasontrace.storage as storage
from reasontrace.tracer import Tracer


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("REASONTRACE_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture()
def tracer():
    return Tracer()


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    def test_start_returns_string_id(self, tracer):
        sid = tracer.start(tag="test", agent_name="bot")
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID4 format

    def test_end_returns_session_dict(self, tracer):
        sid = tracer.start()
        result = tracer.end(sid)
        assert result["session_id"] == sid
        assert result["ended_at"] is not None
        assert "drift" in result

    def test_end_saves_to_disk(self, tracer, tmp_path):
        sid = tracer.start()
        tracer.end(sid)
        rtrace_files = list(tmp_path.glob("*.rtrace"))
        assert len(rtrace_files) == 1

    def test_end_missing_session_raises(self, tracer):
        with pytest.raises(ValueError, match="not found"):
            tracer.end("nonexistent-session-id")

    def test_end_without_start_raises(self, tracer):
        with pytest.raises(ValueError):
            tracer.end()

    def test_first_session_has_initializing_drift(self, tracer):
        sid = tracer.start()
        result = tracer.end(sid)
        assert result["drift"]["severity"] == "initializing"
        assert result["drift"]["composite_score"] is None


# ---------------------------------------------------------------------------
# Logging methods
# ---------------------------------------------------------------------------

class TestLoggingMethods:
    def test_log_reasoning(self, tracer):
        sid = tracer.start()
        tracer.log_reasoning("Decided to use search_web", session_id=sid, step=1)
        session = tracer.get_open_session(sid)
        assert len(session["reasoning_checkpoints"]) == 1
        assert session["reasoning_checkpoints"][0]["content"] == "Decided to use search_web"
        tracer.end(sid)

    def test_log_tool_call(self, tracer):
        sid = tracer.start()
        tracer.log_tool_call(
            "search_web", {"query": "AI"}, {"results": []}, session_id=sid, step=1
        )
        session = tracer.get_open_session(sid)
        assert len(session["tool_calls"]) == 1
        assert session["tool_calls"][0]["tool_name"] == "search_web"
        tracer.end(sid)

    def test_log_decision(self, tracer):
        sid = tracer.start()
        tracer.log_decision(
            ["search_web", "query_db"], "search_web", confidence=0.9, session_id=sid
        )
        session = tracer.get_open_session(sid)
        assert len(session["decisions"]) == 1
        assert session["decisions"][0]["option_chosen"] == "search_web"
        assert session["decisions"][0]["confidence"] == pytest.approx(0.9)
        tracer.end(sid)

    def test_log_without_session_raises(self, tracer):
        with pytest.raises(ValueError, match="No active session"):
            tracer.log_reasoning("hello")

    def test_log_reasoning_step_stored(self, tracer):
        sid = tracer.start()
        tracer.log_reasoning("step content", session_id=sid, step=42)
        session = tracer.get_open_session(sid)
        assert session["reasoning_checkpoints"][0]["step"] == 42
        tracer.end(sid)


# ---------------------------------------------------------------------------
# @tracer.watch decorator
# ---------------------------------------------------------------------------

class TestWatchDecorator:
    def test_decorator_does_not_change_return_value(self, tracer):
        @tracer.watch
        def add(a, b):
            return a + b

        sid = tracer.start()
        result = add(2, 3, session_id=sid)
        assert result == 5
        tracer.end(sid)

    def test_decorator_logs_checkpoint(self, tracer):
        @tracer.watch
        def choose(options):
            return options[0]

        sid = tracer.start()
        choose(["a", "b"])  # No session_id kwarg — uses thread-local
        session = tracer.get_open_session(sid)
        assert len(session["reasoning_checkpoints"]) == 1
        assert "choose" in session["reasoning_checkpoints"][0]["content"]
        tracer.end(sid)

    def test_decorator_with_explicit_session_id(self, tracer):
        @tracer.watch
        def run():
            return "done"

        sid = tracer.start()
        result = run(session_id=sid)
        assert result == "done"
        session = tracer.get_open_session(sid)
        assert len(session["reasoning_checkpoints"]) >= 1
        tracer.end(sid)

    def test_decorator_does_not_break_on_no_session(self, tracer):
        """@tracer.watch should never raise even if no session is active."""
        @tracer.watch
        def safe_fn():
            return 42

        result = safe_fn()
        assert result == 42


# ---------------------------------------------------------------------------
# Anthropic SDK hook
# ---------------------------------------------------------------------------

class TestAttachAnthropic:
    def test_hooks_messages_create(self, tracer):
        """attach_anthropic should intercept tool_use blocks."""
        class FakeToolUseBlock:
            type = "tool_use"
            name = "search_web"
            input = {"query": "test"}

        class FakeResponse:
            content = [FakeToolUseBlock()]

        class FakeMessages:
            def create(self, *args, **kwargs):
                return FakeResponse()

        class FakeClient:
            messages = FakeMessages()

        client = FakeClient()
        sid = tracer.start()
        tracer.attach_anthropic(client, sid)

        # Call the patched create
        response = client.messages.create(model="claude-3", messages=[])
        assert isinstance(response, FakeResponse)

        session = tracer.get_open_session(sid)
        assert len(session["tool_calls"]) == 1
        assert session["tool_calls"][0]["tool_name"] == "search_web"
        tracer.end(sid)

    def test_attach_does_not_break_on_no_tool_use(self, tracer):
        """attach_anthropic should be safe when response has no tool_use blocks."""
        class FakeTextBlock:
            type = "text"
            text = "Hello"

        class FakeResponse:
            content = [FakeTextBlock()]

        class FakeClient:
            class messages:
                @staticmethod
                def create(*args, **kwargs):
                    return FakeResponse()

        client = FakeClient()
        sid = tracer.start()
        tracer.attach_anthropic(client, sid)
        response = client.messages.create()
        assert isinstance(response, FakeResponse)
        tracer.end(sid)
