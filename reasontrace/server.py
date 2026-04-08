"""
server.py — fastmcp MCP server exposing all 13 reasontrace tools.

Transport: stdio (compatible with Claude Code, Claude Desktop, Cursor).

Start with:
    reasontrace-server

Or via Python:
    python -m reasontrace.server

All 13 tools:
  1.  start_session
  2.  log_reasoning
  3.  log_tool_call
  4.  log_decision
  5.  end_session
  6.  get_drift_score
  7.  explain_drift
  8.  get_baseline
  9.  reset_baseline
  10. list_sessions
  11. get_session
  12. search_sessions
  13. delete_session
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

try:
    from fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "fastmcp is required to run the reasontrace MCP server. "
        "Install it with: pip install fastmcp"
    ) from exc

from reasontrace import storage as _storage
from reasontrace.baseline import (
    update_baseline,
    rebuild_baseline,
    baseline_profile,
    get_confidence,
)
from reasontrace.scorer import DriftScorer, explain_drift as _explain_drift

mcp = FastMCP("reasontrace")
_scorer = DriftScorer()

# In-memory open session registry for the MCP server process
_open_sessions: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helper: now
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tool 1 — start_session
# ---------------------------------------------------------------------------

@mcp.tool()
def start_session(tag: str = "", agent_name: str = "") -> dict[str, Any]:
    """Begin capturing a new agent run.

    Parameters
    ----------
    tag:
        Optional label for this session (e.g. "prod-v2", "experiment-3").
    agent_name:
        Optional name of the agent being traced (e.g. "content-generator").

    Returns
    -------
    dict with ``session_id`` (UUID4 string) and confirmation message.
    """
    session = _storage.new_session(tag=tag, agent_name=agent_name)
    _open_sessions[session["session_id"]] = session
    return {
        "session_id": session["session_id"],
        "message": (
            f"Session {session['session_id']} started. "
            "Use log_reasoning, log_tool_call, and log_decision to capture agent behavior, "
            "then call end_session to compute drift score."
        ),
        "started_at": session["started_at"],
    }


# ---------------------------------------------------------------------------
# Tool 2 — log_reasoning
# ---------------------------------------------------------------------------

@mcp.tool()
def log_reasoning(
    session_id: str,
    content: str,
    step: int | None = None,
) -> dict[str, Any]:
    """Capture a reasoning checkpoint for the active session.

    Parameters
    ----------
    session_id:
        The session ID returned by start_session.
    content:
        Free-form text of what the agent saw, considered, and chose.
        This text is embedded locally for semantic drift scoring.
    step:
        Optional step number within the session.

    Returns
    -------
    Confirmation dict with checkpoint count.
    """
    session = _open_sessions.get(session_id)
    if session is None:
        return {"error": f"Session {session_id!r} not found. Call start_session first."}

    cp: dict[str, Any] = {
        "content": content,
        "timestamp": _now(),
    }
    if step is not None:
        cp["step"] = step

    session["reasoning_checkpoints"].append(cp)
    return {
        "session_id": session_id,
        "checkpoint_count": len(session["reasoning_checkpoints"]),
        "message": "Reasoning checkpoint recorded.",
    }


# ---------------------------------------------------------------------------
# Tool 3 — log_tool_call
# ---------------------------------------------------------------------------

@mcp.tool()
def log_tool_call(
    session_id: str,
    tool_name: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    step: int | None = None,
) -> dict[str, Any]:
    """Record a tool invocation within the active session.

    Parameters
    ----------
    session_id:
        The session ID returned by start_session.
    tool_name:
        Name of the tool called (e.g. "search_web", "query_db").
    inputs:
        Dict of inputs passed to the tool.
    outputs:
        Dict of outputs returned by the tool.
    step:
        Optional step number.

    Returns
    -------
    Confirmation dict with tool call count.
    """
    session = _open_sessions.get(session_id)
    if session is None:
        return {"error": f"Session {session_id!r} not found. Call start_session first."}

    tc: dict[str, Any] = {
        "tool_name": tool_name,
        "inputs": inputs,
        "outputs": outputs,
        "timestamp": _now(),
    }
    if step is not None:
        tc["step"] = step

    session["tool_calls"].append(tc)
    return {
        "session_id": session_id,
        "tool_call_count": len(session["tool_calls"]),
        "message": f"Tool call '{tool_name}' recorded.",
    }


# ---------------------------------------------------------------------------
# Tool 4 — log_decision
# ---------------------------------------------------------------------------

@mcp.tool()
def log_decision(
    session_id: str,
    options_considered: list[str],
    option_chosen: str,
    confidence: float | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Capture a specific decision point within the active session.

    Parameters
    ----------
    session_id:
        The session ID returned by start_session.
    options_considered:
        All options the agent evaluated at this decision point.
    option_chosen:
        The option selected.
    confidence:
        Optional confidence score from 0.0 (uncertain) to 1.0 (certain).
    step:
        Optional step number.

    Returns
    -------
    Confirmation dict with decision count.
    """
    session = _open_sessions.get(session_id)
    if session is None:
        return {"error": f"Session {session_id!r} not found. Call start_session first."}

    decision: dict[str, Any] = {
        "options_considered": options_considered,
        "option_chosen": option_chosen,
        "timestamp": _now(),
    }
    if confidence is not None:
        decision["confidence"] = max(0.0, min(1.0, float(confidence)))
    if step is not None:
        decision["step"] = step

    session["decisions"].append(decision)
    return {
        "session_id": session_id,
        "decision_count": len(session["decisions"]),
        "message": f"Decision '{option_chosen}' recorded.",
    }


# ---------------------------------------------------------------------------
# Tool 5 — end_session
# ---------------------------------------------------------------------------

@mcp.tool()
def end_session(session_id: str) -> dict[str, Any]:
    """Close a session, compute its composite drift score, and persist to disk.

    Scoring begins from session 2. Session 1 initialises the baseline.

    Parameters
    ----------
    session_id:
        The session ID returned by start_session.

    Returns
    -------
    Full drift breakdown including composite_score, severity,
    baseline_confidence, and path to the saved .rtrace file.
    """
    session = _open_sessions.pop(session_id, None)
    if session is None:
        return {"error": f"Session {session_id!r} not found or already ended."}

    session["ended_at"] = _now()

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
            "baseline_confidence_note": (
                "Session 1 — initialising baseline. No score yet. "
                "Scoring begins from session 2."
            ),
            "baseline_sessions_count": 0,
        }

    session["drift"] = drift

    # Persist
    path = _storage.save_session(session)
    drift["session_file"] = str(path)
    session["drift"]["session_file"] = str(path)

    # Update baseline incrementally
    update_baseline(session)
    new_baseline = _storage.load_baseline()
    new_count = new_baseline.get("sessions_count", 0) if new_baseline else 0

    return {
        "session_id": session_id,
        "composite_score": drift.get("composite_score"),
        "tool_pattern_score": drift.get("tool_pattern_score"),
        "semantic_score": drift.get("semantic_score"),
        "complexity_score": drift.get("complexity_score"),
        "severity": drift.get("severity"),
        "baseline_confidence": drift.get("baseline_confidence"),
        "baseline_confidence_note": drift.get("baseline_confidence_note"),
        "baseline_sessions_count": drift.get("baseline_sessions_count"),
        "session_file": str(path),
        "message": (
            f"Session ended. {new_count} session(s) now in baseline."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 6 — get_drift_score
# ---------------------------------------------------------------------------

@mcp.tool()
def get_drift_score(session_id: str) -> dict[str, Any]:
    """Return the full drift breakdown for any past session by ID.

    Parameters
    ----------
    session_id:
        ID of the session to retrieve drift data for.

    Returns
    -------
    Full drift breakdown plus session metadata (tag, agent_name, timestamps).
    """
    session = _storage.load_session(session_id)
    if session is None:
        return {"error": f"Session {session_id!r} not found."}

    drift = session.get("drift") or {}
    return {
        "session_id": session_id,
        "tag": session.get("tag", ""),
        "agent_name": session.get("agent_name", ""),
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
        "composite_score": drift.get("composite_score"),
        "tool_pattern_score": drift.get("tool_pattern_score"),
        "semantic_score": drift.get("semantic_score"),
        "complexity_score": drift.get("complexity_score"),
        "severity": drift.get("severity"),
        "baseline_confidence": drift.get("baseline_confidence"),
        "baseline_confidence_note": drift.get("baseline_confidence_note"),
        "baseline_sessions_count": drift.get("baseline_sessions_count"),
        "reasoning_checkpoint_count": len(session.get("reasoning_checkpoints", [])),
        "tool_call_count": len(session.get("tool_calls", [])),
        "decision_count": len(session.get("decisions", [])),
        "session_file": drift.get("session_file"),
    }


# ---------------------------------------------------------------------------
# Tool 7 — explain_drift
# ---------------------------------------------------------------------------

@mcp.tool()
def explain_drift(session_id: str) -> dict[str, Any]:
    """Return a structured plain-English explanation of drift for a session.

    Identifies which component drifted most, what specifically changed,
    which baseline sessions this session diverged from, and a concrete
    actionable recommendation.

    Parameters
    ----------
    session_id:
        ID of the session to explain.

    Returns
    -------
    dict with: primary_driver, tool_change_detail, semantic_detail,
    complexity_detail, divergent_from, recommendation.
    """
    session = _storage.load_session(session_id)
    if session is None:
        return {"error": f"Session {session_id!r} not found."}

    drift = session.get("drift")
    if drift is None:
        return {"error": f"Session {session_id!r} has no drift data. Was end_session called?"}

    baseline = _storage.load_baseline()
    if baseline is None:
        return {"error": "No baseline available. Run more sessions first."}

    explanation = _explain_drift(session, baseline, drift)
    explanation["session_id"] = session_id
    explanation["composite_score"] = drift.get("composite_score")
    explanation["severity"] = drift.get("severity")
    return explanation


# ---------------------------------------------------------------------------
# Tool 8 — get_baseline
# ---------------------------------------------------------------------------

@mcp.tool()
def get_baseline() -> dict[str, Any]:
    """Return the current baseline profile.

    Returns
    -------
    dict with: sessions_count, confidence, average_step_count,
    average_tool_calls_per_session, most_common_tools,
    average_complexity_score, semantic_centroid_description,
    confidence_note.
    """
    baseline = _storage.load_baseline()
    if baseline is None:
        return {
            "sessions_count": 0,
            "confidence": "initializing",
            "confidence_note": "No sessions recorded yet. Run a session to begin.",
            "average_step_count": 0.0,
            "average_tool_calls_per_session": 0.0,
            "most_common_tools": [],
            "average_complexity_score": 0.0,
            "semantic_centroid_description": "No sessions recorded yet.",
        }
    return baseline_profile(baseline)


# ---------------------------------------------------------------------------
# Tool 9 — reset_baseline
# ---------------------------------------------------------------------------

@mcp.tool()
def reset_baseline(session_ids: list[str] | None = None) -> dict[str, Any]:
    """Reset the baseline entirely or rebuild from specific sessions.

    Parameters
    ----------
    session_ids:
        If provided and non-empty, rebuild baseline from only these sessions
        (in the order given). If empty or not provided, wipe baseline completely.

    Returns
    -------
    Confirmation with updated baseline session count.
    """
    if session_ids:
        sessions: list[dict[str, Any]] = []
        missing: list[str] = []
        for sid in session_ids:
            s = _storage.load_session(sid)
            if s is None:
                missing.append(sid)
            else:
                sessions.append(s)

        if missing:
            return {
                "error": f"Sessions not found: {missing}. Baseline not modified.",
            }

        rebuild_baseline(sessions)
        new_baseline = _storage.load_baseline()
        n = new_baseline.get("sessions_count", 0) if new_baseline else 0
        return {
            "message": f"Baseline rebuilt from {n} session(s).",
            "sessions_count": n,
            "session_ids_used": session_ids,
        }
    else:
        _storage.clear_baseline()
        return {
            "message": "Baseline cleared. Session 1 will re-initialise it.",
            "sessions_count": 0,
        }


# ---------------------------------------------------------------------------
# Tool 10 — list_sessions
# ---------------------------------------------------------------------------

@mcp.tool()
def list_sessions() -> dict[str, Any]:
    """Return all saved sessions with metadata, newest first.

    Returns
    -------
    dict with ``sessions`` (list of session summaries) and ``count``.
    """
    all_sessions = _storage.load_all_sessions()

    summaries: list[dict[str, Any]] = []
    for s in all_sessions:
        drift = s.get("drift") or {}
        summaries.append({
            "session_id": s.get("session_id"),
            "tag": s.get("tag", ""),
            "agent_name": s.get("agent_name", ""),
            "started_at": s.get("started_at"),
            "ended_at": s.get("ended_at"),
            "step_count": max(
                len(s.get("reasoning_checkpoints", [])),
                len(s.get("tool_calls", [])),
            ),
            "tool_call_count": len(s.get("tool_calls", [])),
            "composite_score": drift.get("composite_score"),
            "severity": drift.get("severity", "initializing"),
            "baseline_confidence": drift.get("baseline_confidence", "initializing"),
        })

    return {"sessions": summaries, "count": len(summaries)}


# ---------------------------------------------------------------------------
# Tool 11 — get_session
# ---------------------------------------------------------------------------

@mcp.tool()
def get_session(session_id: str) -> dict[str, Any]:
    """Retrieve a full session by ID including all checkpoints, tool calls, and decisions.

    Parameters
    ----------
    session_id:
        ID of the session to retrieve.

    Returns
    -------
    The complete session dict including reasoning_checkpoints, tool_calls,
    decisions, and drift breakdown.
    """
    session = _storage.load_session(session_id)
    if session is None:
        return {"error": f"Session {session_id!r} not found."}
    return session


# ---------------------------------------------------------------------------
# Tool 12 — search_sessions
# ---------------------------------------------------------------------------

@mcp.tool()
def search_sessions(
    tag: str | None = None,
    agent_name: str | None = None,
    min_drift_score: float | None = None,
    max_drift_score: float | None = None,
    tool_name: str | None = None,
    keyword: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Find sessions matching any combination of filters.

    All parameters are optional and combined with AND logic.

    Parameters
    ----------
    tag:
        Filter by session tag (substring match).
    agent_name:
        Filter by agent name (substring match).
    min_drift_score:
        Minimum composite drift score (inclusive).
    max_drift_score:
        Maximum composite drift score (inclusive).
    tool_name:
        Only sessions that called this tool at least once.
    keyword:
        Appears in any reasoning checkpoint content (case-insensitive).
    date_from:
        ISO-8601 datetime — only sessions started at or after this time.
    date_to:
        ISO-8601 datetime — only sessions started at or before this time.

    Returns
    -------
    dict with ``sessions`` (list of summaries) and ``count``.
    """
    all_sessions = _storage.load_all_sessions()
    matches: list[dict[str, Any]] = []

    # Parse date filters once
    dt_from = _parse_iso(date_from) if date_from else None
    dt_to = _parse_iso(date_to) if date_to else None

    for s in all_sessions:
        drift = s.get("drift") or {}
        composite = drift.get("composite_score")

        # Tag filter
        if tag and tag.lower() not in (s.get("tag") or "").lower():
            continue
        # Agent name filter
        if agent_name and agent_name.lower() not in (s.get("agent_name") or "").lower():
            continue
        # Score filters
        if min_drift_score is not None:
            if composite is None or composite < min_drift_score:
                continue
        if max_drift_score is not None:
            if composite is None or composite > max_drift_score:
                continue
        # Tool name filter
        if tool_name:
            session_tools = {tc.get("tool_name") for tc in s.get("tool_calls", [])}
            if tool_name not in session_tools:
                continue
        # Keyword filter
        if keyword:
            kw = keyword.lower()
            found = any(
                kw in (cp.get("content") or "").lower()
                for cp in s.get("reasoning_checkpoints", [])
            )
            if not found:
                continue
        # Date filters
        if dt_from or dt_to:
            started = _parse_iso(s.get("started_at"))
            if started is None:
                continue
            if dt_from and started < dt_from:
                continue
            if dt_to and started > dt_to:
                continue

        matches.append({
            "session_id": s.get("session_id"),
            "tag": s.get("tag", ""),
            "agent_name": s.get("agent_name", ""),
            "started_at": s.get("started_at"),
            "ended_at": s.get("ended_at"),
            "composite_score": composite,
            "severity": drift.get("severity", "initializing"),
            "tool_call_count": len(s.get("tool_calls", [])),
        })

    return {"sessions": matches, "count": len(matches)}


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string to a timezone-aware datetime, or None on failure."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Tool 13 — delete_session
# ---------------------------------------------------------------------------

@mcp.tool()
def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session by ID and rebuild the baseline without it.

    Removes the .rtrace file and removes this session's contribution from
    the baseline by rebuilding the baseline from all remaining sessions.

    Parameters
    ----------
    session_id:
        ID of the session to delete.

    Returns
    -------
    Confirmation with updated baseline session count.
    """
    session = _storage.load_session(session_id)
    if session is None:
        return {"error": f"Session {session_id!r} not found."}

    # Delete the file
    _storage.delete_session_file(session_id)

    # Rebuild baseline from remaining sessions
    remaining = _storage.load_all_sessions()
    # Sort chronologically (oldest first for correct incremental update)
    remaining.sort(key=lambda s: s.get("started_at") or "")
    rebuild_baseline(remaining)

    new_baseline = _storage.load_baseline()
    new_count = new_baseline.get("sessions_count", 0) if new_baseline else 0

    return {
        "message": f"Session {session_id!r} deleted. Baseline rebuilt from {new_count} remaining session(s).",
        "deleted_session_id": session_id,
        "baseline_sessions_count": new_count,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Start the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
