"""
scorer.py — Composite drift score computation.

Three components, each returning a float in [0.0, 1.0]:

  Component 1 — Tool Pattern Drift (default weight 0.35)
    Represents each session as a normalised tool-frequency vector.
    Computes cosine distance from the baseline centroid.

  Component 2 — Semantic Reasoning Drift (default weight 0.40)
    Embeds all reasoning checkpoint texts locally with sentence-transformers.
    Computes cosine distance from the baseline semantic centroid.

  Component 3 — Decision Complexity Drift (default weight 0.25)
    Complexity vector: [step_count, reconsideration_count, avg_options].
    Normalises each dimension via z-score against baseline distribution.
    Euclidean distance capped at 3 std = 1.0.

Composite = weighted sum, clamped to [0.0, 1.0].
Weights configurable via REASONTRACE_WEIGHTS env var (comma-separated floats).

Implementation notes (see DECISIONS.md for full detail):
- Tool pattern edge case: if session tool vector is all-zeros (no baseline
  vocabulary overlap), score is 1.0 (maximum drift), not 0.0.  [Bug Fix 1]
- Severity boundary: score of exactly 0.8 is "alert", not "critical".
  Critical requires strictly > 0.8.  [Bug Fix 2]
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from reasontrace.embedder import (
    cosine_distance,
    embed_session_reasoning,
)
from reasontrace.baseline import _tool_frequency_vector, _complexity_vector

# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = (0.35, 0.40, 0.25)


def _get_weights() -> tuple[float, float, float]:
    """Return (tool_weight, semantic_weight, complexity_weight).

    Reads REASONTRACE_WEIGHTS env var if set (e.g. '0.35,0.40,0.25').
    Falls back to defaults. Normalises so weights sum to 1.0.
    """
    env = os.environ.get("REASONTRACE_WEIGHTS")
    if env:
        try:
            parts = [float(x.strip()) for x in env.split(",")]
            if len(parts) == 3 and all(w >= 0 for w in parts):
                total = sum(parts)
                if total > 0:
                    return (parts[0] / total, parts[1] / total, parts[2] / total)
        except ValueError:
            pass
    total = sum(_DEFAULT_WEIGHTS)
    return tuple(w / total for w in _DEFAULT_WEIGHTS)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

def severity_label(score: float | None) -> str:
    """Map a composite score to a human-readable severity string.

    Thresholds (per spec):
      < 0.3  → none
      0.3–0.6 → warning
      0.6–0.8 → alert      (includes 0.8)
      > 0.8  → critical
    """
    if score is None:
        return "initializing"
    if score < 0.3:
        return "none"
    if score < 0.6:
        return "warning"
    if score <= 0.8:
        return "alert"
    return "critical"


# ---------------------------------------------------------------------------
# Component 1 — Tool pattern drift
# ---------------------------------------------------------------------------

def _tool_pattern_score(
    session: dict[str, Any], baseline: dict[str, Any]
) -> float:
    """Cosine distance between session tool vector and baseline centroid.

    If the session uses *only* tools not present in the baseline vocabulary
    its projection onto the baseline space is the zero vector — which signals
    maximum drift (1.0).  A session that uses *some* baseline tools and *some*
    new tools gets partial credit via the normalised frequency.
    """
    vocab: list[str] = baseline.get("tool_vocabulary", [])
    if not vocab:
        return 0.0

    session_freq = _tool_frequency_vector(session)
    session_total = sum(session_freq.values()) or 1

    # Projection of session onto baseline vocabulary space
    session_vec = np.array(
        [session_freq.get(t, 0) / session_total for t in vocab], dtype=np.float32
    )
    centroid_vec = np.array(
        [baseline["tool_centroid"].get(t, 0.0) for t in vocab], dtype=np.float32
    )

    session_norm = np.linalg.norm(session_vec)

    # If the session used NO tools from the baseline vocabulary it is
    # maximally drifted — return 1.0 directly.
    if session_norm < 1e-9:
        return 1.0

    return float(np.clip(cosine_distance(session_vec, centroid_vec), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Component 2 — Semantic reasoning drift
# ---------------------------------------------------------------------------

def _semantic_score(
    session: dict[str, Any], baseline: dict[str, Any]
) -> float:
    """Cosine distance between session embedding and baseline semantic centroid."""
    centroid_list: list[float] = baseline.get("semantic_centroid", [])
    if not centroid_list or all(v == 0.0 for v in centroid_list):
        return 0.0

    session_embedding = embed_session_reasoning(
        session.get("reasoning_checkpoints", [])
    )
    centroid = np.array(centroid_list, dtype=np.float32)

    return float(np.clip(cosine_distance(session_embedding, centroid), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Component 3 — Decision complexity drift
# ---------------------------------------------------------------------------

def _complexity_score(
    session: dict[str, Any], baseline: dict[str, Any]
) -> float:
    """Z-score based distance on complexity vector, capped at 1.0."""
    mean: list[float] = baseline.get("complexity_mean", [0.0, 0.0, 0.0])
    std: list[float] = baseline.get("complexity_std", [1.0, 1.0, 1.0])

    cx = _complexity_vector(session)

    # Compute z-scores, clamp each dimension at ±3 std
    z_scores: list[float] = []
    for i in range(3):
        s = std[i] if std[i] > 0 else 1.0
        z = (cx[i] - mean[i]) / s
        z_scores.append(z)

    # Euclidean norm of z-score vector; 3 std ≡ full score of 1.0
    # (distance at 3 std for all 3 dims simultaneously = 3*sqrt(3) ≈ 5.2)
    # Normalise by 3.0 (single-dimension cap) and clamp
    euclidean = math.sqrt(sum(z ** 2 for z in z_scores))
    # Cap at sqrt(3 * 9) = sqrt(27) ≈ 5.196 → 1.0
    max_distance = math.sqrt(3 * 9)
    return float(min(euclidean / max_distance, 1.0))


# ---------------------------------------------------------------------------
# Composite scorer — public entry point
# ---------------------------------------------------------------------------

class DriftScorer:
    """Compute composite drift scores for sessions against a baseline.

    Usage
    -----
    >>> scorer = DriftScorer()
    >>> result = scorer.score(session, baseline)
    >>> result['composite_score']
    0.42
    """

    def score(
        self,
        session: dict[str, Any],
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute full drift breakdown for *session* against *baseline*.

        Parameters
        ----------
        session:
            A complete session dict (post reasoning/tool/decision logging).
        baseline:
            The current baseline dict from storage.

        Returns
        -------
        dict with keys:
            composite_score, tool_pattern_score, semantic_score,
            complexity_score, severity, baseline_confidence,
            baseline_confidence_note, baseline_sessions_count.
        """
        from reasontrace.baseline import get_confidence

        n = baseline.get("sessions_count", 0)
        confidence, confidence_note = get_confidence(n)

        if n < 1:
            # No baseline yet — first session initialises it
            return {
                "composite_score": None,
                "tool_pattern_score": None,
                "semantic_score": None,
                "complexity_score": None,
                "severity": "initializing",
                "baseline_confidence": "initializing",
                "baseline_confidence_note": "Session 1 — initialising baseline. No score yet.",
                "baseline_sessions_count": 0,
            }

        tool_w, sem_w, cx_w = _get_weights()

        tool_score = _tool_pattern_score(session, baseline)
        semantic_score = _semantic_score(session, baseline)
        complexity_score = _complexity_score(session, baseline)

        composite = float(np.clip(
            tool_w * tool_score + sem_w * semantic_score + cx_w * complexity_score,
            0.0,
            1.0,
        ))

        return {
            "composite_score": round(composite, 4),
            "tool_pattern_score": round(tool_score, 4),
            "semantic_score": round(semantic_score, 4),
            "complexity_score": round(complexity_score, 4),
            "severity": severity_label(composite),
            "baseline_confidence": confidence,
            "baseline_confidence_note": confidence_note,
            "baseline_sessions_count": n,
        }


# ---------------------------------------------------------------------------
# Drift explanation helper
# ---------------------------------------------------------------------------

def explain_drift(
    session: dict[str, Any],
    baseline: dict[str, Any],
    drift: dict[str, Any],
) -> dict[str, Any]:
    """Generate a structured plain-English explanation of a session's drift.

    Parameters
    ----------
    session:
        Full session dict.
    baseline:
        Current baseline dict.
    drift:
        The drift result dict returned by DriftScorer.score().

    Returns
    -------
    dict with keys: primary_driver, tool_change_detail, semantic_detail,
        complexity_detail, divergent_from, recommendation.
    """
    tool_score = drift.get("tool_pattern_score") or 0.0
    semantic_score = drift.get("semantic_score") or 0.0
    complexity_score = drift.get("complexity_score") or 0.0
    composite = drift.get("composite_score") or 0.0

    # Determine primary driver
    scores = {
        "tool_pattern": tool_score,
        "semantic_reasoning": semantic_score,
        "decision_complexity": complexity_score,
    }
    primary = max(scores, key=scores.get)  # type: ignore[arg-type]

    # Tool change detail
    session_tools = list(_tool_frequency_vector(session).keys())
    baseline_tools: list[str] = baseline.get("most_common_tools", [])
    new_tools = [t for t in session_tools if t not in baseline_tools]
    missing_tools = [t for t in baseline_tools if t not in session_tools]

    if tool_score > 0.3:
        if new_tools and missing_tools:
            tool_detail = (
                f"Agent switched from [{', '.join(missing_tools)}] to "
                f"[{', '.join(new_tools)}], departing significantly from baseline pattern."
            )
        elif new_tools:
            tool_detail = (
                f"Agent introduced new tools not in baseline: [{', '.join(new_tools)}]."
            )
        elif missing_tools:
            tool_detail = (
                f"Agent stopped using baseline-common tools: [{', '.join(missing_tools)}]."
            )
        else:
            tool_detail = (
                f"Tool usage frequency distribution drifted from baseline "
                f"(cosine distance {tool_score:.2f})."
            )
    else:
        tool_detail = "Tool usage is consistent with baseline."

    # Semantic detail
    if semantic_score > 0.3:
        cps = session.get("reasoning_checkpoints", [])
        sample = cps[0].get("content", "")[:120] if cps else ""
        semantic_detail = (
            f"Reasoning text became semantically distant from baseline "
            f"(cosine distance {semantic_score:.2f}). "
            f'Sample: "{sample}..."' if sample else
            f"Reasoning text became semantically distant (distance {semantic_score:.2f})."
        )
    else:
        semantic_detail = "Reasoning language is semantically consistent with baseline."

    # Complexity detail
    cx = _complexity_vector(session)
    mean = baseline.get("complexity_mean", [0.0, 0.0, 0.0])
    if complexity_score > 0.3:
        complexity_detail = (
            f"Step count changed from baseline average {mean[0]:.1f} to {cx[0]:.0f} "
            f"in this session. Reconsiderations: {cx[1]:.0f} vs baseline avg {mean[1]:.1f}."
        )
    else:
        complexity_detail = "Decision complexity is consistent with baseline."

    # Recommendation
    if composite > 0.8:
        rec = (
            "Critical drift detected. Check whether the system prompt, model version, "
            "or tool availability changed between the last stable session and this one."
        )
    elif composite > 0.6:
        rec = (
            f"Significant {primary.replace('_', ' ')} drift. "
            "Inspect recent sessions for prompt or configuration changes. "
            "Use get_session to review full reasoning checkpoints."
        )
    elif composite > 0.3:
        rec = (
            f"Minor {primary.replace('_', ' ')} drift. "
            "Worth monitoring over next few sessions to confirm trend."
        )
    else:
        rec = "No action needed. Behavior is consistent with baseline."

    # Find most divergent baseline sessions (by session_id list)
    session_ids = baseline.get("session_ids", [])
    # Return the last 3 sessions as likely comparison candidates
    divergent_from = session_ids[-3:] if len(session_ids) >= 3 else session_ids

    return {
        "primary_driver": primary,
        "tool_change_detail": tool_detail,
        "semantic_detail": semantic_detail,
        "complexity_detail": complexity_detail,
        "divergent_from": divergent_from,
        "recommendation": rec,
    }
