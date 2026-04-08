"""
reasontrace — Local-first reasoning drift detection for AI agents.

Automatically builds a behavioral baseline from your agent's reasoning
history and scores every new run against it, detecting when your agent
starts reasoning differently before you notice it in the outputs.

Public API
----------
Tracer       — decorator, SDK hook, and manual checkpoint interface
DriftScorer  — compute composite drift scores against a baseline
install      — interactive MCP config injection (also runs at pip install)
"""

from reasontrace.tracer import Tracer
from reasontrace.scorer import DriftScorer
from reasontrace.installer import install

__all__ = ["Tracer", "DriftScorer", "install"]
__version__ = "0.1.0"
