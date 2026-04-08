"""
baseline.py — Baseline building and incremental updates for drift scoring.

The baseline is a running summary of all past sessions, maintained as a
single JSON file at ~/.reasontrace/baseline.json.  After each session ends
the baseline is updated incrementally using a running mean — no need to
re-read all sessions on every update.

Baseline structure
------------------
{
  "sessions_count": int,
  "tool_centroid": {tool_name: float, ...},    # frequency-weighted centroid
  "tool_vocabulary": [str, ...],               # all tools ever seen
  "semantic_centroid": [float, ...],           # average embedding vector
  "complexity_mean": [float, float, float],    # [steps, reconsiderations, avg_options]
  "complexity_std": [float, float, float],     # running std for z-score
  "complexity_m2": [float, float, float],      # Welford M2 for online variance
  "session_ids": [str, ...],                   # ordered list of contributing sessions
  "most_common_tools": [str, ...],             # top-5 tools by total frequency
  "total_tool_calls": int,
  "total_steps": int,
}
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from reasontrace import storage as _storage
from reasontrace.embedder import embed_session_reasoning, cosine_distance

# ---------------------------------------------------------------------------
# Confidence tier helpers
# ---------------------------------------------------------------------------

def get_confidence(sessions_count: int) -> tuple[str, str]:
    """Return (confidence_label, confidence_note) for a given baseline size."""
    if sessions_count < 2:
        return (
            "initializing",
            "Only 1 session in baseline. Storing session — scoring begins from session 2.",
        )
    if sessions_count < 5:
        return (
            "low",
            f"Based on {sessions_count} sessions. Directional signal — treat as indicative, not definitive.",
        )
    if sessions_count < 10:
        return (
            "medium",
            f"Based on {sessions_count} sessions. Reasonably reliable.",
        )
    return (
        "high",
        f"Based on {sessions_count} sessions. Statistically robust.",
    )


# ---------------------------------------------------------------------------
# Session feature extraction
# ---------------------------------------------------------------------------

def _tool_frequency_vector(session: dict[str, Any]) -> Counter:
    """Return a Counter of tool call frequencies for a session."""
    counter: Counter = Counter()
    for tc in session.get("tool_calls", []):
        name = tc.get("tool_name", "unknown")
        counter[name] += 1
    return counter


def _complexity_vector(session: dict[str, Any]) -> list[float]:
    """Return [step_count, reconsideration_count, avg_options_per_decision]."""
    checkpoints = session.get("reasoning_checkpoints", [])
    tool_calls = session.get("tool_calls", [])
    decisions = session.get("decisions", [])

    step_count = float(max(
        len(checkpoints),
        len(tool_calls),
        max((d.get("step", 0) for d in decisions), default=0),
        max((cp.get("step", 0) for cp in checkpoints), default=0),
    ))

    # Count reconsiderations: decisions where the chosen option does not match
    # the most-frequently considered option, or where confidence < 0.5
    reconsiderations = 0
    for d in decisions:
        options = d.get("options_considered", [])
        confidence = d.get("confidence")
        if confidence is not None and confidence < 0.5:
            reconsiderations += 1
        elif len(options) > 2:
            reconsiderations += 1

    avg_options = 0.0
    if decisions:
        avg_options = sum(
            len(d.get("options_considered", [])) for d in decisions
        ) / len(decisions)

    return [step_count, float(reconsiderations), avg_options]


# ---------------------------------------------------------------------------
# Welford online mean/variance
# ---------------------------------------------------------------------------

def _welford_update(
    n: int, mean: list[float], m2: list[float], x: list[float]
) -> tuple[list[float], list[float]]:
    """One step of Welford's online algorithm.

    Parameters
    ----------
    n:    new count after adding this sample (1-indexed)
    mean: current running mean vector
    m2:   current Welford M2 vector
    x:    new sample vector

    Returns updated (mean, m2).
    """
    new_mean = []
    new_m2 = []
    for i in range(len(x)):
        delta = x[i] - mean[i]
        m = mean[i] + delta / n
        delta2 = x[i] - m
        new_mean.append(m)
        new_m2.append(m2[i] + delta * delta2)
    return new_mean, new_m2


def _welford_std(n: int, m2: list[float]) -> list[float]:
    """Compute population std from Welford M2 accumulators."""
    if n < 2:
        return [1.0] * len(m2)
    return [math.sqrt(v / n) if v / n > 0 else 1.0 for v in m2]


# ---------------------------------------------------------------------------
# Baseline update
# ---------------------------------------------------------------------------

def update_baseline(session: dict[str, Any]) -> dict[str, Any]:
    """Incrementally add a session to the baseline and persist.

    Uses a running mean for tool centroid, semantic centroid, and
    complexity vector — no need to reload all past sessions.

    Parameters
    ----------
    session:
        A completed session dict (post end_session).

    Returns
    -------
    The updated baseline dict.
    """
    baseline = _storage.load_baseline() or _empty_baseline()

    n_old = baseline["sessions_count"]
    n_new = n_old + 1

    # --- Tool pattern ---
    tool_freq = _tool_frequency_vector(session)
    # Update vocabulary
    vocab = set(baseline["tool_vocabulary"]) | set(tool_freq.keys())
    baseline["tool_vocabulary"] = sorted(vocab)

    # Increment total tool calls and update centroid as running mean
    session_tool_total = sum(tool_freq.values()) or 1
    for tool in vocab:
        old_val = baseline["tool_centroid"].get(tool, 0.0)
        new_val = tool_freq.get(tool, 0) / session_tool_total
        baseline["tool_centroid"][tool] = (old_val * n_old + new_val) / n_new

    # most_common_tools: top-5 by centroid weight
    sorted_tools = sorted(
        baseline["tool_centroid"].items(), key=lambda kv: kv[1], reverse=True
    )
    baseline["most_common_tools"] = [t for t, _ in sorted_tools[:5]]
    baseline["total_tool_calls"] = baseline.get("total_tool_calls", 0) + sum(
        tool_freq.values()
    )

    # --- Semantic ---
    session_embedding = embed_session_reasoning(
        session.get("reasoning_checkpoints", [])
    ).tolist()
    old_centroid = np.array(baseline["semantic_centroid"], dtype=np.float32)
    new_emb = np.array(session_embedding, dtype=np.float32)
    updated_centroid = ((old_centroid * n_old) + new_emb) / n_new
    baseline["semantic_centroid"] = updated_centroid.tolist()

    # --- Complexity ---
    cx = _complexity_vector(session)
    baseline["total_steps"] = baseline.get("total_steps", 0) + int(cx[0])
    old_mean = baseline["complexity_mean"]
    old_m2 = baseline["complexity_m2"]
    new_mean, new_m2 = _welford_update(n_new, old_mean, old_m2, cx)
    baseline["complexity_mean"] = new_mean
    baseline["complexity_m2"] = new_m2
    baseline["complexity_std"] = _welford_std(n_new, new_m2)

    # --- Metadata ---
    baseline["sessions_count"] = n_new
    session_ids: list[str] = baseline.get("session_ids", [])
    session_ids.append(session["session_id"])
    baseline["session_ids"] = session_ids

    _storage.save_baseline(baseline)
    return baseline


def rebuild_baseline(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a fresh baseline from a list of session dicts.

    Replaces any existing baseline.  Used by reset_baseline tool.

    Parameters
    ----------
    sessions:
        List of complete session dicts to build the baseline from.
        Must be in chronological order (oldest first is ideal).

    Returns
    -------
    The new baseline dict.
    """
    _storage.clear_baseline()
    baseline = _empty_baseline()
    for session in sessions:
        # Temporarily save baseline after each session to allow
        # update_baseline to read-then-write correctly
        _storage.save_baseline(baseline)
        baseline = update_baseline(session)
    return baseline


def _empty_baseline() -> dict[str, Any]:
    """Return a zero-initialised baseline dict."""
    return {
        "sessions_count": 0,
        "tool_centroid": {},
        "tool_vocabulary": [],
        "semantic_centroid": [0.0] * 384,
        "complexity_mean": [0.0, 0.0, 0.0],
        "complexity_std": [1.0, 1.0, 1.0],
        "complexity_m2": [0.0, 0.0, 0.0],
        "session_ids": [],
        "most_common_tools": [],
        "total_tool_calls": 0,
        "total_steps": 0,
    }


# ---------------------------------------------------------------------------
# Baseline profile summary (for get_baseline tool)
# ---------------------------------------------------------------------------

def baseline_profile(baseline: dict[str, Any]) -> dict[str, Any]:
    """Return a human-readable profile of the current baseline."""
    n = baseline.get("sessions_count", 0)
    confidence, note = get_confidence(n)

    avg_steps = (
        baseline["total_steps"] / n if n > 0 else 0.0
    )
    avg_tool_calls = (
        baseline["total_tool_calls"] / n if n > 0 else 0.0
    )

    # Characterise semantic centroid as a brief description
    centroid = np.array(baseline.get("semantic_centroid", []), dtype=np.float32)
    if np.linalg.norm(centroid) < 1e-6 or n == 0:
        semantic_desc = "No sessions recorded yet."
    elif n < 3:
        semantic_desc = "Too few sessions to characterise reasoning style."
    else:
        semantic_desc = (
            f"Baseline represents {n} session(s) of reasoning. "
            "Use explain_drift to compare any session against this centroid."
        )

    avg_complexity = baseline["complexity_mean"][0] if baseline.get("complexity_mean") else 0.0

    return {
        "sessions_count": n,
        "confidence": confidence,
        "confidence_note": note,
        "average_step_count": round(avg_steps, 2),
        "average_tool_calls_per_session": round(avg_tool_calls, 2),
        "most_common_tools": baseline.get("most_common_tools", []),
        "average_complexity_score": round(avg_complexity, 2),
        "semantic_centroid_description": semantic_desc,
    }
