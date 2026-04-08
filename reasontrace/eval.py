"""
eval.py — Reproducible evaluation module for reasontrace drift detection.

Run the full evaluation:
    python -m reasontrace.eval

Run a single scenario:
    python -m reasontrace.eval --scenario control
    python -m reasontrace.eval --scenario tool_drift
    python -m reasontrace.eval --scenario semantic_drift
    python -m reasontrace.eval --scenario complexity_drift
    python -m reasontrace.eval --scenario combined_drift

All sessions are generated programmatically with deterministic seeds (seed=42).
No real LLM inference required.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Evaluation bootstrap — use a temporary directory so eval never pollutes
# the user's actual ~/.reasontrace/ baseline.
# ---------------------------------------------------------------------------

_EVAL_TMPDIR: str | None = None


def _setup_eval_env() -> str:
    """Create a temporary storage directory and point REASONTRACE_DIR at it."""
    global _EVAL_TMPDIR
    tmpdir = tempfile.mkdtemp(prefix="reasontrace_eval_")
    _EVAL_TMPDIR = tmpdir
    os.environ["REASONTRACE_DIR"] = tmpdir
    return tmpdir


def _teardown_eval_env() -> None:
    """Clean up the temporary directory."""
    import shutil
    global _EVAL_TMPDIR
    if _EVAL_TMPDIR and Path(_EVAL_TMPDIR).exists():
        shutil.rmtree(_EVAL_TMPDIR, ignore_errors=True)
    if "REASONTRACE_DIR" in os.environ:
        del os.environ["REASONTRACE_DIR"]
    _EVAL_TMPDIR = None


# ---------------------------------------------------------------------------
# Synthetic session factories
# ---------------------------------------------------------------------------

def _make_session(
    tag: str,
    reasoning_texts: list[str],
    tool_sequence: list[str],
    step_count: int,
    reconsiderations: int = 0,
    avg_options: int = 2,
    started_offset_minutes: int = 0,
) -> dict[str, Any]:
    """Build a synthetic session dict ready for scoring."""
    started_at = (
        datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        + timedelta(minutes=started_offset_minutes)
    ).isoformat()
    ended_at = (
        datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        + timedelta(minutes=started_offset_minutes + 5)
    ).isoformat()

    session_id = str(uuid.uuid4())

    reasoning_checkpoints = []
    for i, text in enumerate(reasoning_texts):
        reasoning_checkpoints.append({
            "step": i + 1,
            "content": text,
            "timestamp": started_at,
        })

    tool_calls = []
    for i, tool in enumerate(tool_sequence):
        tool_calls.append({
            "step": i + 1,
            "tool_name": tool,
            "inputs": {"query": f"eval-input-{i}"},
            "outputs": {"result": f"eval-output-{i}"},
            "timestamp": started_at,
        })

    decisions = []
    options_pool = ["option_a", "option_b", "option_c", "option_d"]
    for i in range(reconsiderations):
        decisions.append({
            "step": i + 1,
            "options_considered": options_pool[:avg_options],
            "option_chosen": options_pool[0],
            "confidence": 0.4,  # low confidence → reconsideration
            "timestamp": started_at,
        })
    # Add normal decisions to fill up to step_count
    for i in range(reconsiderations, min(step_count, 3)):
        decisions.append({
            "step": i + 1,
            "options_considered": options_pool[:avg_options],
            "option_chosen": options_pool[0],
            "confidence": 0.9,
            "timestamp": started_at,
        })

    return {
        "session_id": session_id,
        "tag": tag,
        "agent_name": "eval-agent",
        "started_at": started_at,
        "ended_at": ended_at,
        "reasoning_checkpoints": reasoning_checkpoints,
        "tool_calls": tool_calls,
        "decisions": decisions,
        "drift": None,
    }


# ---------------------------------------------------------------------------
# Reasoning text corpora
# ---------------------------------------------------------------------------

_DECISIVE_TEXTS = [
    "Received task. Selected search_web immediately because it retrieves live structured data efficiently.",
    "Search completed. Extracted relevant records using extract_data. Clear path forward.",
    "Data extracted. Applied summarize to condense results into final output. Task complete.",
    "Used search_web as first step — fastest route to authoritative sources for this query type.",
    "extract_data confirmed the search yielded high-quality results. Summarize applied without hesitation.",
    "Chose search_web over query_db: real-time data preferred for this task category.",
    "Step 1: search_web. Step 2: extract_data. Step 3: summarize. Sequence optimal and consistent.",
    "Query identified as time-sensitive. search_web selected. extract_data applied. summarize completed.",
    "Task pattern matches previous runs. Standard tool sequence: search → extract → summarize.",
    "Executed standard three-step pipeline. No deviations from established pattern warranted.",
    "Received task. Identified relevant data sources. search_web initiated immediately.",
    "Results from search_web verified. extract_data applied to filter noise. Summarize completed.",
    "Consistent with baseline approach: search_web, extract_data, summarize — in that order.",
    "Confidence high. search_web optimal. extract_data efficient. summarize delivered output.",
    "Standard task completed via standard pipeline. No anomalies detected.",
]

_HEDGING_TEXTS = [
    "Not entirely sure which approach is best here. Maybe search_web could work, but query_db might also be reasonable.",
    "Uncertain whether to proceed with extract_data or skip directly to summarize. Hard to say.",
    "Could potentially use search_web, though I'm not confident this is the right tool for this task type.",
    "Possibly the right path is search_web, but path B through query_db might also be valid. Unclear.",
    "Attempting extract_data, though it's not obvious this will yield useful results. May need to reconsider.",
    "Several approaches seem plausible but none is clearly optimal. Proceeding tentatively with search_web.",
    "Not confident in the tool selection here. search_web is a guess — other tools might be better suited.",
    "Hard to determine the right sequence. Defaulting to search_web though other paths might be more appropriate.",
    "Uncertain about the task requirements. Choosing search_web provisionally, may revise.",
    "Not sure this is the right approach. The task might benefit from query_db instead of search_web.",
]

_DB_TOOLS = ["query_db", "format_output"]
_SEARCH_TOOLS = ["search_web", "extract_data", "summarize"]
_CONTENT_TOOLS_BASELINE = ["search_web", "extract_data", "draft_content", "review", "publish"]
_CONTENT_TOOLS_DRIFT = ["query_db", "llm_rewrite", "push_api"]


# ---------------------------------------------------------------------------
# Core scenario runner
# ---------------------------------------------------------------------------

def _run_scenario(
    name: str,
    sessions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score all sessions against an incrementally built baseline.

    Returns list of result dicts for each scored session (session 2+).
    """
    # Re-import after env var is set
    from reasontrace import storage as _storage  # noqa: PLC0415
    from reasontrace.baseline import update_baseline  # noqa: PLC0415
    from reasontrace.scorer import DriftScorer  # noqa: PLC0415

    scorer = DriftScorer()
    results: list[dict[str, Any]] = []

    for i, session in enumerate(sessions):
        baseline = _storage.load_baseline()

        if baseline is None or baseline.get("sessions_count", 0) < 1:
            # First session — just initialise baseline, no score
            _storage.save_baseline({
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
            })
            update_baseline(session)
            _storage.save_session(session)
        else:
            drift = scorer.score(session, baseline)
            session["drift"] = drift
            _storage.save_session(session)
            update_baseline(session)
            results.append({
                "session_num": i + 1,
                "session_id": session["session_id"],
                "tool_score": drift.get("tool_pattern_score"),
                "semantic_score": drift.get("semantic_score"),
                "complexity_score": drift.get("complexity_score"),
                "composite": drift.get("composite_score"),
                "severity": drift.get("severity"),
                "confidence": drift.get("baseline_confidence"),
            })

    return results


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(results: list[dict[str, Any]], title: str) -> None:
    print(f"\n{'═' * 76}")
    print(f"  {title}")
    print(f"{'═' * 76}")
    header = (
        f"{'Session':>8}  {'Tool':>8}  {'Semantic':>10}  "
        f"{'Complexity':>12}  {'Composite':>10}  {'Severity':<10}  {'Confidence'}"
    )
    print(header)
    print("-" * 76)
    for r in results:
        def _fmt(v: float | None) -> str:
            return f"{v:.2f}" if v is not None else "  N/A"

        print(
            f"{r['session_num']:>8}  "
            f"{_fmt(r['tool_score']):>8}  "
            f"{_fmt(r['semantic_score']):>10}  "
            f"{_fmt(r['complexity_score']):>12}  "
            f"{_fmt(r['composite']):>10}  "
            f"{r['severity']:<10}  "
            f"{r['confidence']}"
        )
    print()


# ---------------------------------------------------------------------------
# Scenario 1 — Control (no drift)
# ---------------------------------------------------------------------------

def scenario_control() -> list[dict[str, Any]]:
    """15 sessions of identical tool sequence and consistent reasoning."""
    rng = np.random.default_rng(42)
    sessions: list[dict[str, Any]] = []
    for i in range(15):
        # Slightly vary the reasoning text but keep it semantically consistent
        text_idx = rng.integers(0, 10)
        texts = [
            _DECISIVE_TEXTS[text_idx % len(_DECISIVE_TEXTS)],
            _DECISIVE_TEXTS[(text_idx + 1) % len(_DECISIVE_TEXTS)],
            _DECISIVE_TEXTS[(text_idx + 2) % len(_DECISIVE_TEXTS)],
        ]
        sessions.append(_make_session(
            tag="control",
            reasoning_texts=texts,
            tool_sequence=list(_SEARCH_TOOLS),
            step_count=3,
            reconsiderations=0,
            avg_options=2,
            started_offset_minutes=i * 60,
        ))
    return sessions


# ---------------------------------------------------------------------------
# Scenario 2 — Tool pattern drift
# ---------------------------------------------------------------------------

def scenario_tool_drift() -> list[dict[str, Any]]:
    """10 baseline sessions (search pipeline), then 5 with DB pipeline."""
    sessions: list[dict[str, Any]] = []
    # Baseline: same decisive texts + search tools
    for i in range(10):
        sessions.append(_make_session(
            tag="tool_drift",
            reasoning_texts=[
                _DECISIVE_TEXTS[i % len(_DECISIVE_TEXTS)],
                _DECISIVE_TEXTS[(i + 1) % len(_DECISIVE_TEXTS)],
                _DECISIVE_TEXTS[(i + 2) % len(_DECISIVE_TEXTS)],
            ],
            tool_sequence=list(_SEARCH_TOOLS),
            step_count=3,
            reconsiderations=0,
            avg_options=2,
            started_offset_minutes=i * 60,
        ))
    # Drift: same decisive texts, different tools
    for i in range(5):
        sessions.append(_make_session(
            tag="tool_drift",
            reasoning_texts=[
                _DECISIVE_TEXTS[i % len(_DECISIVE_TEXTS)],
                _DECISIVE_TEXTS[(i + 1) % len(_DECISIVE_TEXTS)],
            ],
            tool_sequence=list(_DB_TOOLS),
            step_count=3,
            reconsiderations=0,
            avg_options=2,
            started_offset_minutes=(10 + i) * 60,
        ))
    return sessions


# ---------------------------------------------------------------------------
# Scenario 3 — Semantic reasoning drift
# ---------------------------------------------------------------------------

def scenario_semantic_drift() -> list[dict[str, Any]]:
    """10 baseline sessions (decisive), then 5 with hedging language."""
    sessions: list[dict[str, Any]] = []
    for i in range(10):
        sessions.append(_make_session(
            tag="semantic_drift",
            reasoning_texts=[
                _DECISIVE_TEXTS[i % len(_DECISIVE_TEXTS)],
                _DECISIVE_TEXTS[(i + 1) % len(_DECISIVE_TEXTS)],
                _DECISIVE_TEXTS[(i + 2) % len(_DECISIVE_TEXTS)],
            ],
            tool_sequence=list(_SEARCH_TOOLS),
            step_count=3,
            reconsiderations=0,
            avg_options=2,
            started_offset_minutes=i * 60,
        ))
    for i in range(5):
        sessions.append(_make_session(
            tag="semantic_drift",
            reasoning_texts=[
                _HEDGING_TEXTS[i % len(_HEDGING_TEXTS)],
                _HEDGING_TEXTS[(i + 1) % len(_HEDGING_TEXTS)],
                _HEDGING_TEXTS[(i + 2) % len(_HEDGING_TEXTS)],
            ],
            tool_sequence=list(_SEARCH_TOOLS),
            step_count=3,
            reconsiderations=0,
            avg_options=2,
            started_offset_minutes=(10 + i) * 60,
        ))
    return sessions


# ---------------------------------------------------------------------------
# Scenario 4 — Complexity drift
# ---------------------------------------------------------------------------

def scenario_complexity_drift() -> list[dict[str, Any]]:
    """10 baseline (3-4 steps, no reconsiderations), then 5 (9-12 steps, 2 reconsiderations)."""
    sessions: list[dict[str, Any]] = []
    for i in range(10):
        sessions.append(_make_session(
            tag="complexity_drift",
            reasoning_texts=[
                _DECISIVE_TEXTS[i % len(_DECISIVE_TEXTS)],
                _DECISIVE_TEXTS[(i + 1) % len(_DECISIVE_TEXTS)],
            ],
            tool_sequence=list(_SEARCH_TOOLS),
            step_count=3,
            reconsiderations=0,
            avg_options=2,
            started_offset_minutes=i * 60,
        ))
    for i in range(5):
        step_count = 9 + (i % 4)  # 9–12
        sessions.append(_make_session(
            tag="complexity_drift",
            reasoning_texts=[
                _DECISIVE_TEXTS[i % len(_DECISIVE_TEXTS)],
                _DECISIVE_TEXTS[(i + 2) % len(_DECISIVE_TEXTS)],
            ],
            tool_sequence=list(_SEARCH_TOOLS),
            step_count=step_count,
            reconsiderations=2,
            avg_options=4,
            started_offset_minutes=(10 + i) * 60,
        ))
    return sessions


# ---------------------------------------------------------------------------
# Scenario 5 — Combined drift
# ---------------------------------------------------------------------------

def scenario_combined_drift() -> list[dict[str, Any]]:
    """10 realistic content generation baseline, then 5 with all three drifts."""
    sessions: list[dict[str, Any]] = []
    for i in range(10):
        sessions.append(_make_session(
            tag="combined_drift",
            reasoning_texts=[
                f"Step {j+1}: {_DECISIVE_TEXTS[(i+j) % len(_DECISIVE_TEXTS)]}"
                for j in range(3)
            ],
            tool_sequence=list(_CONTENT_TOOLS_BASELINE),
            step_count=4,
            reconsiderations=0,
            avg_options=2,
            started_offset_minutes=i * 60,
        ))
    for i in range(5):
        sessions.append(_make_session(
            tag="combined_drift",
            reasoning_texts=[
                f"Step {j+1}: {_HEDGING_TEXTS[(i+j) % len(_HEDGING_TEXTS)]}"
                for j in range(3)
            ],
            tool_sequence=list(_CONTENT_TOOLS_DRIFT),
            step_count=10,
            reconsiderations=3,
            avg_options=4,
            started_offset_minutes=(10 + i) * 60,
        ))
    return sessions


# ---------------------------------------------------------------------------
# Verdict printers
# ---------------------------------------------------------------------------

def _verdict_control(results: list[dict[str, Any]]) -> None:
    fp = sum(1 for r in results if (r["composite"] or 0) >= 0.3)
    print(f"  False positive rate: {fp}/{len(results)} sessions ({fp/len(results)*100:.0f}%)")
    verdict = "✅ Consistent behavior correctly scores low throughout." if fp == 0 else "⚠️  Some false positives detected."
    print(f"  Verdict: {verdict}\n")


def _verdict_drift(
    results: list[dict[str, Any]],
    drift_session_nums: range,
    component: str,
    label: str,
) -> None:
    drift_results = [r for r in results if r["session_num"] in drift_session_nums]
    tp = sum(1 for r in drift_results if (r["composite"] or 0) >= 0.3)
    high = [r[component] or 0 for r in drift_results]
    print(f"  True positive rate: {tp}/{len(drift_results)} drift sessions correctly flagged")
    if high:
        print(f"  Average {label} score in drift sessions: {sum(high)/len(high):.3f}")
    verdict = "✅" if tp == len(drift_results) else "⚠️"
    print(f"  Verdict: {verdict} {label} drift {'isolated and detected correctly.' if tp == len(drift_results) else 'partially detected.'}\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

_SCENARIOS = {
    "control": ("Scenario 1 — Control (No Drift)", scenario_control, None),
    "tool_drift": ("Scenario 2 — Tool Pattern Drift", scenario_tool_drift, range(11, 16)),
    "semantic_drift": ("Scenario 3 — Semantic Reasoning Drift", scenario_semantic_drift, range(11, 16)),
    "complexity_drift": ("Scenario 4 — Complexity Drift", scenario_complexity_drift, range(11, 16)),
    "combined_drift": ("Scenario 5 — Combined Drift", scenario_combined_drift, range(11, 16)),
}


def run_scenario(name: str) -> None:
    """Run a single named scenario."""
    if name not in _SCENARIOS:
        print(f"Unknown scenario {name!r}. Choose from: {', '.join(_SCENARIOS.keys())}")
        sys.exit(1)

    title, factory, drift_range = _SCENARIOS[name]
    tmpdir = _setup_eval_env()
    try:
        print(f"\nRunning {title} (storage: {tmpdir})")
        sessions = factory()
        results = _run_scenario(name, sessions)
        _print_table(results, title)

        if name == "control":
            _verdict_control(results)
        elif name == "tool_drift":
            _verdict_drift(results, drift_range, "tool_score", "Tool pattern")
        elif name == "semantic_drift":
            _verdict_drift(results, drift_range, "semantic_score", "Semantic reasoning")
        elif name == "complexity_drift":
            _verdict_drift(results, drift_range, "complexity_score", "Complexity")
        elif name == "combined_drift":
            tp = sum(1 for r in results if r["session_num"] in drift_range and (r["composite"] or 0) >= 0.6)
            print(f"  True positive rate at ALERT level: {tp}/{len(list(drift_range))} sessions")
            print(f"  Verdict: {'✅ All combined drift sessions flagged at alert level.' if tp == len(list(drift_range)) else '⚠️  Some sessions below alert threshold.'}\n")
    finally:
        _teardown_eval_env()


def run_all() -> None:
    """Run all five scenarios sequentially."""
    print("\n" + "═" * 76)
    print("  reasontrace — Full Evaluation Suite")
    print("  Deterministic seed=42 | No real LLM inference required")
    print("═" * 76)

    for name, (title, factory, drift_range) in _SCENARIOS.items():
        tmpdir = _setup_eval_env()
        try:
            print(f"\nRunning {title}...")
            sessions = factory()
            results = _run_scenario(name, sessions)
            _print_table(results, title)

            if name == "control":
                _verdict_control(results)
            elif name == "tool_drift":
                _verdict_drift(results, drift_range, "tool_score", "Tool pattern")
            elif name == "semantic_drift":
                _verdict_drift(results, drift_range, "semantic_score", "Semantic reasoning")
            elif name == "complexity_drift":
                _verdict_drift(results, drift_range, "complexity_score", "Complexity")
            elif name == "combined_drift":
                tp = sum(1 for r in results if r["session_num"] in drift_range and (r["composite"] or 0) >= 0.6)
                print(f"  True positive rate at ALERT level: {tp}/{len(list(drift_range))} sessions")
                print(f"  Verdict: {'✅' if tp == len(list(drift_range)) else '⚠️'} Combined drift evaluation complete.\n")
        finally:
            _teardown_eval_env()

    print("═" * 76)
    print("  Evaluation complete. See EVALUATION.md for full analysis.")
    print("═" * 76 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="reasontrace reproducible evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Scenarios:",
            "  control         15 sessions, no drift (false positive rate test)",
            "  tool_drift      10 baseline + 5 tool-pattern drift sessions",
            "  semantic_drift  10 baseline + 5 semantic reasoning drift sessions",
            "  complexity_drift 10 baseline + 5 complexity drift sessions",
            "  combined_drift  10 baseline + 5 all-three-components drift sessions",
        ]),
    )
    parser.add_argument(
        "--scenario",
        choices=list(_SCENARIOS.keys()),
        default=None,
        help="Run only this scenario (default: run all)",
    )
    args = parser.parse_args()

    if args.scenario:
        run_scenario(args.scenario)
    else:
        run_all()
