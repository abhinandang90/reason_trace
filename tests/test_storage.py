"""
tests/test_storage.py — Unit tests for reasontrace.storage

Tests cover:
- Storage directory resolution (env var override)
- Atomic writes
- Session create / save / load / delete
- Baseline save / load / clear
- list_session_files ordering
- new_session structure
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

import reasontrace.storage as storage


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Point REASONTRACE_DIR at a temp directory for every test."""
    monkeypatch.setenv("REASONTRACE_DIR", str(tmp_path))
    # Clear the module-level cache that Path.home() might have created
    yield tmp_path


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

class TestGetStorageDir:
    def test_uses_env_var(self, tmp_path):
        assert storage.get_storage_dir() == tmp_path

    def test_creates_directory(self, monkeypatch, tmp_path):
        new_dir = tmp_path / "nested" / "path"
        monkeypatch.setenv("REASONTRACE_DIR", str(new_dir))
        result = storage.get_storage_dir()
        assert result.exists()
        assert result.is_dir()


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

class TestSessionCRUD:
    def test_new_session_structure(self):
        s = storage.new_session(tag="test", agent_name="bot")
        assert "session_id" in s
        assert s["tag"] == "test"
        assert s["agent_name"] == "bot"
        assert s["reasoning_checkpoints"] == []
        assert s["tool_calls"] == []
        assert s["decisions"] == []
        assert s["drift"] is None
        assert s["started_at"] is not None
        assert s["ended_at"] is None

    def test_save_and_load_roundtrip(self):
        s = storage.new_session(tag="roundtrip")
        s["reasoning_checkpoints"].append({"content": "hello", "step": 1})
        path = storage.save_session(s)
        assert path.exists()
        assert path.suffix == ".rtrace"

        loaded = storage.load_session(s["session_id"])
        assert loaded is not None
        assert loaded["session_id"] == s["session_id"]
        assert loaded["tag"] == "roundtrip"
        assert loaded["reasoning_checkpoints"][0]["content"] == "hello"

    def test_load_nonexistent_returns_none(self):
        result = storage.load_session("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_delete_session_file(self):
        s = storage.new_session()
        storage.save_session(s)
        assert storage.delete_session_file(s["session_id"]) is True
        assert storage.load_session(s["session_id"]) is None

    def test_delete_nonexistent_returns_false(self):
        assert storage.delete_session_file("nonexistent-id") is False

    def test_atomic_write_no_partial_file(self, tmp_path):
        """The .tmp file must not remain after a successful write."""
        s = storage.new_session()
        storage.save_session(s)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], "Temporary files should be cleaned up"


# ---------------------------------------------------------------------------
# list_session_files
# ---------------------------------------------------------------------------

class TestListSessionFiles:
    def test_returns_newest_first(self):
        import time
        s1 = storage.new_session(tag="first")
        storage.save_session(s1)
        time.sleep(0.05)
        s2 = storage.new_session(tag="second")
        storage.save_session(s2)

        files = storage.list_session_files()
        assert len(files) == 2
        # Second session (newer) should be first
        assert s2["session_id"] in files[0].name

    def test_empty_directory_returns_empty_list(self):
        assert storage.list_session_files() == []


# ---------------------------------------------------------------------------
# load_all_sessions
# ---------------------------------------------------------------------------

class TestLoadAllSessions:
    def test_loads_all(self):
        sessions = [storage.new_session(tag=f"s{i}") for i in range(3)]
        for s in sessions:
            storage.save_session(s)
        loaded = storage.load_all_sessions()
        assert len(loaded) == 3

    def test_skips_malformed_files(self, tmp_path):
        bad = tmp_path / "bad-session.rtrace"
        bad.write_text("not json", encoding="utf-8")
        s = storage.new_session()
        storage.save_session(s)
        loaded = storage.load_all_sessions()
        assert len(loaded) == 1  # only the valid one


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

class TestBaselineCRUD:
    def test_save_and_load(self):
        bl = {"sessions_count": 5, "tool_centroid": {"search_web": 0.8}}
        storage.save_baseline(bl)
        loaded = storage.load_baseline()
        assert loaded is not None
        assert loaded["sessions_count"] == 5
        assert loaded["tool_centroid"]["search_web"] == pytest.approx(0.8)

    def test_load_nonexistent_returns_none(self):
        assert storage.load_baseline() is None

    def test_clear_baseline(self):
        storage.save_baseline({"sessions_count": 1})
        storage.clear_baseline()
        assert storage.load_baseline() is None

    def test_atomic_baseline_write(self, tmp_path):
        storage.save_baseline({"sessions_count": 1})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# now_iso
# ---------------------------------------------------------------------------

class TestNowIso:
    def test_returns_string(self):
        ts = storage.now_iso()
        assert isinstance(ts, str)
        assert "T" in ts  # ISO 8601 format

    def test_returns_utc(self):
        ts = storage.now_iso()
        assert ts.endswith("+00:00") or ts.endswith("Z")
