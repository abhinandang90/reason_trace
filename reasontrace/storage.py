"""
storage.py — Persistent storage for reasontrace sessions.

Manages the ~/.reasontrace/ directory (or REASONTRACE_DIR env override).
Each session is stored as a human-readable .rtrace JSON file.
All file writes are atomic: write to .tmp then os.replace().
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def get_storage_dir() -> Path:
    """Return the storage directory, creating it if necessary.

    Respects the REASONTRACE_DIR environment variable. Defaults to
    ~/.reasontrace/.
    """
    env_dir = os.environ.get("REASONTRACE_DIR")
    if env_dir:
        storage_dir = Path(env_dir).expanduser().resolve()
    else:
        storage_dir = Path.home() / ".reasontrace"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def get_baseline_path() -> Path:
    """Return path to the baseline JSON file."""
    return get_storage_dir() / "baseline.json"


def get_session_path(session_id: str) -> Path:
    """Return path for a given session's .rtrace file."""
    return get_storage_dir() / f"{session_id}.rtrace"


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as JSON to *path* atomically via a .tmp rename."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        # Clean up tmp if os.replace raised
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Session I/O
# ---------------------------------------------------------------------------

def save_session(session: dict[str, Any]) -> Path:
    """Persist a session dict to disk as <session_id>.rtrace.

    Returns the path to the written file.
    """
    session_id = session["session_id"]
    path = get_session_path(session_id)
    _atomic_write(path, session)
    return path


def load_session(session_id: str) -> dict[str, Any] | None:
    """Load and return a session dict by ID, or None if not found."""
    path = get_session_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def delete_session_file(session_id: str) -> bool:
    """Delete the .rtrace file for a session. Returns True if deleted."""
    path = get_session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def list_session_files() -> list[Path]:
    """Return all .rtrace files in the storage directory, newest first."""
    storage_dir = get_storage_dir()
    files = sorted(
        storage_dir.glob("*.rtrace"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


def load_all_sessions() -> list[dict[str, Any]]:
    """Load every stored session, skipping any that are malformed."""
    sessions: list[dict[str, Any]] = []
    for path in list_session_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return sessions


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def save_baseline(baseline: dict[str, Any]) -> None:
    """Persist the baseline dict atomically."""
    _atomic_write(get_baseline_path(), baseline)


def load_baseline() -> dict[str, Any] | None:
    """Load and return the baseline dict, or None if it does not exist."""
    path = get_baseline_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clear_baseline() -> None:
    """Delete the baseline file entirely."""
    path = get_baseline_path()
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# In-memory session builder
# ---------------------------------------------------------------------------

def new_session(tag: str | None = None, agent_name: str | None = None) -> dict[str, Any]:
    """Create a fresh in-memory session dict."""
    return {
        "session_id": str(uuid.uuid4()),
        "tag": tag or "",
        "agent_name": agent_name or "",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "reasoning_checkpoints": [],
        "tool_calls": [],
        "decisions": [],
        "drift": None,
    }


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
