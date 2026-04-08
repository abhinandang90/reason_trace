"""
tracer.py — High-level Tracer class for instrumenting Python agents.

Provides:
  - @tracer.watch decorator: auto-captures function calls as reasoning checkpoints
  - tracer.attach_anthropic(client, session_id): hooks into Anthropic SDK to
    auto-log tool_use content blocks as tool calls
  - tracer.start() / tracer.end(): thin Python wrappers around MCP session lifecycle
    for agents that call reasontrace from Python (rather than via MCP tools)
  - Thread-local session context so @tracer.watch can find the active session
    without requiring explicit session_id kwargs on every call
"""

from __future__ import annotations

import functools
import inspect
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from reasontrace import storage as _storage
from reasontrace.baseline import update_baseline
from reasontrace.scorer import DriftScorer

# ---------------------------------------------------------------------------
# Thread-local session context
# ---------------------------------------------------------------------------

_local = threading.local()
_scorer = DriftScorer()


def _get_active_session_id() -> str | None:
    """Return the session ID stored in thread-local context, or None."""
    return getattr(_local, "session_id", None)


def _set_active_session_id(session_id: str | None) -> None:
    """Store a session ID in thread-local context."""
    _local.session_id = session_id


# ---------------------------------------------------------------------------
# In-memory session registry (for multi-session environments)
# ---------------------------------------------------------------------------

# Maps session_id -> in-memory session dict for open sessions
_open_sessions: dict[str, dict[str, Any]] = {}
_sessions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Tracer class
# ---------------------------------------------------------------------------

class Tracer:
    """Instrument an AI agent to capture reasoning checkpoints and tool calls.

    Typical usage (Python SDK)
    --------------------------
    >>> tracer = Tracer()
    >>> session_id = tracer.start("my-task", agent_name="content-bot")
    >>>
    >>> @tracer.watch
    ... def choose_tool(context, options):
    ...     return options[0]
    >>>
    >>> result = choose_tool(ctx, opts)   # auto-logs reasoning checkpoint
    >>> tracer.end(session_id)            # scores drift, saves session
    """

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start(self, tag: str = "", agent_name: str = "") -> str:
        """Begin a new traced session.

        Parameters
        ----------
        tag:
            Optional label for this session (e.g. "prod-v2", "experiment-3").
        agent_name:
            Optional name of the agent being traced.

        Returns
        -------
        The new session ID (UUID4 string).
        """
        session = _storage.new_session(tag=tag, agent_name=agent_name)
        session_id = session["session_id"]
        with _sessions_lock:
            _open_sessions[session_id] = session
        _set_active_session_id(session_id)
        return session_id

    def end(self, session_id: str | None = None) -> dict[str, Any]:
        """Close a session, compute drift score, persist to disk.

        Parameters
        ----------
        session_id:
            The session to end.  Defaults to the thread-local active session.

        Returns
        -------
        The full session dict including the ``drift`` sub-dict.
        """
        sid = session_id or _get_active_session_id()
        if sid is None:
            raise ValueError("No active session. Call tracer.start() first.")

        with _sessions_lock:
            session = _open_sessions.pop(sid, None)
        if session is None:
            raise ValueError(f"Session {sid!r} not found or already ended.")

        session["ended_at"] = _storage.now_iso()

        # Score against baseline
        baseline = _storage.load_baseline()
        if baseline and baseline.get("sessions_count", 0) >= 1:
            drift = _scorer.score(session, baseline)
        else:
            drift = {
                "composite_score": None,
                "tool_pattern_score": None,
                "semantic_score": None,
                "complexity_score": None,
                "severity": "initializing",
                "baseline_confidence": "initializing",
                "baseline_confidence_note": "Session 1 — initialising baseline. No score yet.",
                "baseline_sessions_count": 0,
            }
        session["drift"] = drift

        # Persist
        path = _storage.save_session(session)
        drift["session_file"] = str(path)
        session["drift"]["session_file"] = str(path)

        # Update baseline
        update_baseline(session)

        if _get_active_session_id() == sid:
            _set_active_session_id(None)

        return session

    # ------------------------------------------------------------------
    # Manual logging helpers (Python-side equivalents of MCP tools)
    # ------------------------------------------------------------------

    def log_reasoning(
        self,
        content: str,
        session_id: str | None = None,
        step: int | None = None,
    ) -> None:
        """Record a reasoning checkpoint for the active session.

        Parameters
        ----------
        content:
            Free-form text describing what the agent saw, considered, and chose.
        session_id:
            Session to log to.  Defaults to thread-local active session.
        step:
            Optional step number within the session.
        """
        sid = session_id or _get_active_session_id()
        if sid is None:
            raise ValueError("No active session.")
        with _sessions_lock:
            session = _open_sessions.get(sid)
        if session is None:
            raise ValueError(f"Session {sid!r} not found.")
        cp: dict[str, Any] = {
            "content": content,
            "timestamp": _storage.now_iso(),
        }
        if step is not None:
            cp["step"] = step
        session["reasoning_checkpoints"].append(cp)

    def log_tool_call(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        session_id: str | None = None,
        step: int | None = None,
    ) -> None:
        """Record a tool invocation.

        Parameters
        ----------
        tool_name:
            Name of the tool called.
        inputs:
            Dict of input arguments passed to the tool.
        outputs:
            Dict of outputs returned by the tool.
        session_id:
            Session to log to.  Defaults to thread-local active session.
        step:
            Optional step number.
        """
        sid = session_id or _get_active_session_id()
        if sid is None:
            raise ValueError("No active session.")
        with _sessions_lock:
            session = _open_sessions.get(sid)
        if session is None:
            raise ValueError(f"Session {sid!r} not found.")
        tc: dict[str, Any] = {
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": _storage.now_iso(),
        }
        if step is not None:
            tc["step"] = step
        session["tool_calls"].append(tc)

    def log_decision(
        self,
        options_considered: list[str],
        option_chosen: str,
        confidence: float | None = None,
        session_id: str | None = None,
        step: int | None = None,
    ) -> None:
        """Record a decision point.

        Parameters
        ----------
        options_considered:
            All options the agent evaluated.
        option_chosen:
            The option the agent selected.
        confidence:
            Optional confidence score (0.0–1.0).
        session_id:
            Session to log to.  Defaults to thread-local active session.
        step:
            Optional step number.
        """
        sid = session_id or _get_active_session_id()
        if sid is None:
            raise ValueError("No active session.")
        with _sessions_lock:
            session = _open_sessions.get(sid)
        if session is None:
            raise ValueError(f"Session {sid!r} not found.")
        decision: dict[str, Any] = {
            "options_considered": options_considered,
            "option_chosen": option_chosen,
            "timestamp": _storage.now_iso(),
        }
        if confidence is not None:
            decision["confidence"] = confidence
        if step is not None:
            decision["step"] = step
        session["decisions"].append(decision)

    # ------------------------------------------------------------------
    # @tracer.watch decorator
    # ------------------------------------------------------------------

    def watch(self, func: Callable) -> Callable:
        """Wrap a function so every call is logged as a reasoning checkpoint.

        The decorator inspects the function's arguments and return value and
        records them as a reasoning checkpoint in the active session.

        Thread-local context is used to find the active session.  You can also
        pass ``session_id`` as a keyword argument to the wrapped function.

        Parameters
        ----------
        func:
            The function to instrument.

        Examples
        --------
        >>> @tracer.watch
        ... def choose_tool(context, options):
        ...     return options[0]
        >>>
        >>> session_id = tracer.start()
        >>> result = choose_tool(context, options)
        >>> tracer.end()
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract session_id from kwargs without passing to wrapped fn
            sid = kwargs.pop("session_id", None) or _get_active_session_id()

            # Call the original function
            result = func(*args, **kwargs)

            # Build reasoning checkpoint content
            sig = inspect.signature(func)
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                arg_summary = ", ".join(
                    f"{k}={repr(v)[:60]}" for k, v in list(bound.arguments.items())[:5]
                )
            except Exception:
                arg_summary = repr(args)[:120]

            content = (
                f"Called {func.__name__}({arg_summary}) "
                f"→ returned {repr(result)[:120]}"
            )

            if sid:
                try:
                    self.log_reasoning(content, session_id=sid)
                except Exception:
                    pass  # Never let tracing break agent logic

            return result

        return wrapper

    # ------------------------------------------------------------------
    # Anthropic SDK hook
    # ------------------------------------------------------------------

    def attach_anthropic(self, client: Any, session_id: str) -> None:
        """Intercept tool_use content blocks from an Anthropic API client.

        Wraps ``client.messages.create`` non-destructively.  Every
        ``tool_use`` block in a response triggers ``log_tool_call``.

        Parameters
        ----------
        client:
            An ``anthropic.Anthropic`` (or ``AsyncAnthropic``) client instance.
        session_id:
            The session to log tool calls into.
        """
        original_create = client.messages.create

        @functools.wraps(original_create)
        def patched_create(*args: Any, **kwargs: Any) -> Any:
            response = original_create(*args, **kwargs)
            try:
                for block in getattr(response, "content", []):
                    if getattr(block, "type", None) == "tool_use":
                        self.log_tool_call(
                            tool_name=getattr(block, "name", "unknown"),
                            inputs=dict(getattr(block, "input", {}) or {}),
                            outputs={},  # outputs not available at request time
                            session_id=session_id,
                        )
            except Exception:
                pass  # Never break the client
            return response

        client.messages.create = patched_create

    # ------------------------------------------------------------------
    # Internal: retrieve open session (for MCP server bridge)
    # ------------------------------------------------------------------

    def get_open_session(self, session_id: str) -> dict[str, Any] | None:
        """Return the in-memory session dict for an open session, or None."""
        with _sessions_lock:
            return _open_sessions.get(session_id)

    def put_open_session(self, session: dict[str, Any]) -> None:
        """Register a session dict as open (used by MCP server)."""
        with _sessions_lock:
            _open_sessions[session["session_id"]] = session
