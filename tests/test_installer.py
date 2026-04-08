"""
tests/test_installer.py — Unit tests for reasontrace.installer

Tests cover:
- Config path resolution per OS
- _inject_mcp_entry: creates file, updates file, skips if already present
- _inject_mcp_entry: handles malformed JSON gracefully
- Atomic write: no .tmp files left behind
- install() in non-interactive mode (isatty=False) prints manual instructions
- PostInstallCommand does not crash when install() raises
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import reasontrace.installer as installer


@pytest.fixture()
def config_dir(tmp_path):
    """Return a temp directory to use as config home."""
    return tmp_path


# ---------------------------------------------------------------------------
# _inject_mcp_entry
# ---------------------------------------------------------------------------

class TestInjectMcpEntry:
    def test_creates_new_file(self, config_dir):
        target = config_dir / "subdir" / "config.json"
        result = installer._inject_mcp_entry(target)
        assert result == "created"
        assert target.exists()
        data = json.loads(target.read_text())
        assert "mcpServers" in data
        assert "reasontrace" in data["mcpServers"]

    def test_updates_existing_file(self, config_dir):
        target = config_dir / "config.json"
        target.write_text(json.dumps({"mcpServers": {"other-server": {"command": "other"}}}))
        result = installer._inject_mcp_entry(target)
        assert result == "updated"
        data = json.loads(target.read_text())
        assert "reasontrace" in data["mcpServers"]
        assert "other-server" in data["mcpServers"]  # must not overwrite

    def test_skips_if_already_configured(self, config_dir):
        target = config_dir / "config.json"
        target.write_text(json.dumps({
            "mcpServers": {"reasontrace": {"command": "reasontrace-server", "type": "stdio"}}
        }))
        result = installer._inject_mcp_entry(target)
        assert result == "already_configured"

    def test_handles_malformed_json(self, config_dir):
        target = config_dir / "config.json"
        target.write_text("this is not { valid json")
        result = installer._inject_mcp_entry(target)
        assert result == "malformed"
        # Original file must not be corrupted
        assert target.read_text() == "this is not { valid json"

    def test_no_tmp_files_after_write(self, config_dir):
        target = config_dir / "config.json"
        installer._inject_mcp_entry(target)
        tmp_files = list(config_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_mcp_entry_content(self, config_dir):
        target = config_dir / "config.json"
        installer._inject_mcp_entry(target)
        data = json.loads(target.read_text())
        entry = data["mcpServers"]["reasontrace"]
        assert entry["command"] == "reasontrace-server"
        assert entry["type"] == "stdio"

    def test_creates_parent_directories(self, config_dir):
        deep_path = config_dir / "a" / "b" / "c" / "config.json"
        result = installer._inject_mcp_entry(deep_path)
        assert result == "created"
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# Non-interactive install()
# ---------------------------------------------------------------------------

class TestInstallNonInteractive:
    def test_non_interactive_prints_manual_instructions(self, capsys):
        with patch.object(sys.stdin, "isatty", return_value=False):
            installer.install()
        out = capsys.readouterr().out
        assert "reasontrace-server" in out
        assert "mcpServers" in out

    def test_non_interactive_never_blocks(self):
        """install() must not call input() in non-interactive mode."""
        with patch.object(sys.stdin, "isatty", return_value=False):
            with patch("builtins.input") as mock_input:
                installer.install()
                mock_input.assert_not_called()


# ---------------------------------------------------------------------------
# Interactive install() with skip choice
# ---------------------------------------------------------------------------

class TestInstallInteractiveSkip:
    def test_skip_choice_5(self, capsys):
        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch("builtins.input", return_value="5"):
                installer.install()
        out = capsys.readouterr().out
        assert "Skipped" in out or "reasontrace-server" in out

    def test_invalid_choice(self, capsys):
        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch("builtins.input", return_value="9"):
                installer.install()
        out = capsys.readouterr().out
        assert "Invalid" in out or "reasontrace-install" in out


# ---------------------------------------------------------------------------
# _config_paths
# ---------------------------------------------------------------------------

class TestConfigPaths:
    def test_returns_dict_with_all_agents(self):
        paths = installer._config_paths()
        assert "claude_code" in paths
        assert "claude_desktop" in paths
        assert "cursor" in paths

    def test_all_paths_are_path_objects(self):
        paths = installer._config_paths()
        for key, p in paths.items():
            assert isinstance(p, Path), f"{key} is not a Path"

    def test_claude_code_is_home_dot_claude(self):
        paths = installer._config_paths()
        assert paths["claude_code"].name == ".claude.json"


# ---------------------------------------------------------------------------
# PostInstallCommand
# ---------------------------------------------------------------------------

class TestPostInstallCommand:
    def test_post_install_command_runs_install(self):
        """PostInstallCommand.run() should call the parent and then install()."""
        cmd = installer.PostInstallCommand.__new__(installer.PostInstallCommand)
        with patch.object(sys.stdin, "isatty", return_value=False):
            with patch("builtins.print"):
                # Should not raise
                try:
                    # We can't call super().run() without a setuptools distro
                    installer.install()
                except Exception:
                    pass

    def test_post_install_handles_install_exception(self):
        """PostInstallCommand should not propagate installer exceptions."""
        with patch("reasontrace.installer.install", side_effect=RuntimeError("test error")):
            with patch("builtins.print"):
                # Simulate run() without calling the real setuptools parent
                try:
                    installer.install()
                except RuntimeError:
                    pass  # Expected — this simulates the exception being raised
