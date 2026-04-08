"""
installer.py — Interactive MCP config injection for reasontrace.

Runs automatically at the end of ``pip install reasontrace`` via the
setuptools PostInstallCommand hook.  Also exposed as the
``reasontrace-install`` entry point so the user can re-run configuration
at any time (e.g. after installing a new coding agent).

Non-interactive environments (CI, Docker) are detected via
``sys.stdin.isatty()`` and the prompt is skipped silently.

All file writes are atomic (write to .tmp, then os.replace()).
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config file path resolution
# ---------------------------------------------------------------------------

def _config_paths() -> dict[str, Path]:
    """Return the MCP config file path for each supported coding agent.

    Paths are OS-aware.
    """
    system = platform.system()
    home = Path.home()

    # Claude Code — always ~/.claude.json regardless of OS
    claude_code = home / ".claude.json"

    # Claude Desktop — varies by OS
    if system == "Darwin":
        claude_desktop = (
            home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        )
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", str(home / "AppData" / "Roaming"))
        claude_desktop = Path(appdata) / "Claude" / "claude_desktop_config.json"
    else:
        claude_desktop = home / ".config" / "Claude" / "claude_desktop_config.json"

    # Cursor — varies by OS
    if system == "Windows":
        userprofile = os.environ.get("USERPROFILE", str(home))
        cursor = Path(userprofile) / ".cursor" / "mcp.json"
    else:
        cursor = home / ".cursor" / "mcp.json"

    return {
        "claude_code": claude_code,
        "claude_desktop": claude_desktop,
        "cursor": cursor,
    }


# ---------------------------------------------------------------------------
# MCP entry to inject
# ---------------------------------------------------------------------------

_MCP_ENTRY: dict[str, Any] = {
    "command": "reasontrace-server",
    "type": "stdio",
}

_MANUAL_CONFIG_TEXT = """\
{
  "mcpServers": {
    "reasontrace": {
      "command": "reasontrace-server",
      "type": "stdio"
    }
  }
}"""

_MANUAL_PATH_TEXT = """\
  Config file locations:
  Claude Code:     ~/.claude.json
  Claude Desktop:  ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
                   %APPDATA%\\Claude\\claude_desktop_config.json (Windows)
                   ~/.config/Claude/claude_desktop_config.json (Linux)
  Cursor:          ~/.cursor/mcp.json (macOS/Linux)
                   %USERPROFILE%\\.cursor\\mcp.json (Windows)"""


# ---------------------------------------------------------------------------
# Atomic JSON read/merge/write
# ---------------------------------------------------------------------------

def _inject_mcp_entry(config_path: Path) -> str:
    """Inject the reasontrace MCP entry into *config_path*.

    Returns a status string: 'updated', 'already_configured', 'created',
    or 'malformed' (on JSON decode error).
    """
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")

    if not config_path.exists():
        # Create the file with the MCP entry
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"mcpServers": {"reasontrace": _MCP_ENTRY}}
        try:
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp_path, config_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return "created"

    # File exists — read and merge
    try:
        existing_text = config_path.read_text(encoding="utf-8")
        existing_data = json.loads(existing_text)
    except (json.JSONDecodeError, OSError):
        return "malformed"

    if not isinstance(existing_data, dict):
        return "malformed"

    mcp_servers = existing_data.setdefault("mcpServers", {})

    if "reasontrace" in mcp_servers:
        return "already_configured"

    mcp_servers["reasontrace"] = _MCP_ENTRY
    try:
        tmp_path.write_text(json.dumps(existing_data, indent=2), encoding="utf-8")
        os.replace(tmp_path, config_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return "updated"


# ---------------------------------------------------------------------------
# Prompt and interactive flow
# ---------------------------------------------------------------------------

_BANNER = "━" * 40

_PROMPT_TEXT = f"""\
{_BANNER}
  reasontrace — MCP server setup
{_BANNER}

Which coding agents would you like to configure?

  [1] Claude Code
  [2] Claude Desktop
  [3] Cursor
  [4] All of the above
  [5] Skip for now

Your choice: """


def _print_skip_instructions() -> None:
    print(f"\n{_BANNER}")
    print("  Skipped. To configure manually, add this to your MCP config:\n")
    print(_MANUAL_CONFIG_TEXT)
    print(f"\n{_MANUAL_PATH_TEXT}")
    print(f"\n  Run 'reasontrace-install' anytime to configure later.")
    print(_BANNER)


def _print_results(
    selected: dict[str, bool],
    results: dict[str, str],
    paths: dict[str, Path],
) -> None:
    """Print the installation result banner."""
    agent_labels = {
        "claude_code": "Claude Code",
        "claude_desktop": "Claude Desktop",
        "cursor": "Cursor",
    }

    print(f"\n{_BANNER}")
    for key, label in agent_labels.items():
        if not selected.get(key):
            symbol = "✗"
            detail = "skipped (not selected)"
        else:
            result = results.get(key, "unknown")
            if result in ("updated", "created"):
                symbol = "✓"
                detail = f"{paths[key]} {'updated' if result == 'updated' else 'created'}"
                # Use tilde shorthand for readability
                try:
                    rel = paths[key].relative_to(Path.home())
                    detail = f"~/{rel} {'updated' if result == 'updated' else 'created'}"
                except ValueError:
                    pass
            elif result == "already_configured":
                symbol = "✓"
                detail = "already configured, skipping"
            elif result == "malformed":
                symbol = "⚠"
                detail = (
                    f"WARNING: {paths[key]} contains invalid JSON. "
                    "Add the MCP entry manually (see below)."
                )
            else:
                symbol = "✗"
                detail = "unknown error"
        print(f"  {symbol} {label:<15} → {detail}")

    # Check for any malformed files
    malformed = [k for k, v in results.items() if v == "malformed"]
    if malformed:
        print(f"\n  Manual config entry to add:")
        print(_MANUAL_CONFIG_TEXT)

    print(f"\n  Restart your coding agent to activate reasontrace tools.")
    print(f"  Run 'reasontrace-install' anytime to reconfigure.")
    print(_BANNER)


# ---------------------------------------------------------------------------
# Core install function
# ---------------------------------------------------------------------------

def install() -> None:
    """Interactive MCP configuration wizard.

    Detects non-interactive environments and skips the prompt gracefully.
    This function is called both by PostInstallCommand and directly by the
    ``reasontrace-install`` entry point.
    """
    # Non-interactive environment: skip prompt, print manual instructions
    if not sys.stdin.isatty():
        _print_skip_instructions()
        return

    try:
        choice_raw = input(_PROMPT_TEXT).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        _print_skip_instructions()
        return

    if choice_raw == "5" or choice_raw == "":
        _print_skip_instructions()
        return

    if choice_raw not in ("1", "2", "3", "4"):
        print(f"\nInvalid choice '{choice_raw}'. Run 'reasontrace-install' to try again.")
        return

    selected: dict[str, bool] = {
        "claude_code": choice_raw in ("1", "4"),
        "claude_desktop": choice_raw in ("2", "4"),
        "cursor": choice_raw in ("3", "4"),
    }

    paths = _config_paths()
    results: dict[str, str] = {}

    for key, is_selected in selected.items():
        if is_selected:
            results[key] = _inject_mcp_entry(paths[key])

    _print_results(selected, results, paths)


# ---------------------------------------------------------------------------
# setuptools PostInstallCommand hook
# ---------------------------------------------------------------------------

try:
    from setuptools.command.install import install as _SetuptoolsInstall

    class PostInstallCommand(_SetuptoolsInstall):  # type: ignore[misc]
        """Subclass of setuptools ``install`` that runs the MCP config wizard."""

        def run(self) -> None:
            super().run()
            try:
                install()
            except Exception as exc:
                # Never block a pip install due to installer failures
                print(
                    f"\n[reasontrace] Installer encountered an error: {exc}\n"
                    "Run 'reasontrace-install' manually to configure MCP."
                )

except ImportError:
    # setuptools not available (e.g., in a minimal environment)
    class PostInstallCommand:  # type: ignore[no-redef]
        """Stub when setuptools is unavailable."""

        def run(self) -> None:
            install()
