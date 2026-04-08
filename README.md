# reasontrace

**Local-first reasoning drift detection for AI agents, served as an MCP server.**

Every observability platform wants you to send your agent's reasoning to their cloud and open their dashboard. reasontrace does neither. It installs locally, runs locally, stores everything on your machine, and surfaces all tools inside Claude Code or Cursor — where you already work. And unlike manual trace comparison tools, it automatically tells you when your agent starts reasoning differently — before you see it in the outputs.

---

## Why reasontrace is categorically different

| Tool | What it does | What's wrong with it (for this use case) |
|------|-------------|------------------------------------------|
| LangSmith, Langfuse, AgentOps, Braintrust | Cloud-hosted agent observability | Requires account, dashboard, data leaving your machine |
| Agent-Diff benchmark paper | Diffs environment state between two manually selected runs | Manual comparison, state-level not reasoning-level, no automatic detection |
| OpenTelemetry / MLflow tracing | Infrastructure-level distributed tracing | Team-oriented, complex setup, not a single pip install |
| **reasontrace** | Automatic reasoning drift detection, MCP-native, fully local | — |

reasontrace automatically builds a behavioral baseline from your agent's reasoning history and scores every new run against it. It detects when your agent starts reasoning differently — in its tool choices, its semantic thinking style, and its decision complexity — before you notice it in the outputs.

No cloud. No account. No dashboard. No manual run comparison. `pip install reasontrace` does everything.

---

## Install

```bash
pip install reasontrace
```

At the end of installation you will see an interactive prompt:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  reasontrace — MCP server setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Which coding agents would you like to configure?

  [1] Claude Code
  [2] Claude Desktop
  [3] Cursor
  [4] All of the above
  [5] Skip for now

Your choice: _
```

Choose your coding agent(s) and reasontrace injects its MCP server config automatically. No JSON editing required. Restart your coding agent and all 13 reasontrace tools are available immediately.

To reconfigure at any time:

```bash
reasontrace-install
```

---

## Quick start — Python SDK

```python
from reasontrace import Tracer

tracer = Tracer()

@tracer.watch
def choose_tool(context: dict, options: list[str]) -> str:
    # your agent logic here
    return options[0]

# Start a traced session
session_id = tracer.start(tag="prod-run-42", agent_name="content-generator")

# Manually log reasoning checkpoints
tracer.log_reasoning(
    "Received task. Decided to use search_web because it retrieves live data efficiently.",
    session_id=session_id,
    step=1,
)

# @tracer.watch captures function calls automatically
selected = choose_tool({"task": "summarize AI trends"}, ["search_web", "query_db"])

# Log a tool call
tracer.log_tool_call(
    tool_name="search_web",
    inputs={"query": "AI trends 2026"},
    outputs={"results": ["..."]},
    session_id=session_id,
    step=2,
)

# Log a decision point
tracer.log_decision(
    options_considered=["search_web", "query_db", "use_cache"],
    option_chosen="search_web",
    confidence=0.85,
    session_id=session_id,
    step=2,
)

# End session — scores drift, saves .rtrace file, updates baseline
result = tracer.end(session_id)

print(result["drift"]["composite_score"])   # e.g. 0.42
print(result["drift"]["severity"])          # "warning"
print(result["drift"]["baseline_confidence"])  # "medium"
```

---

## Quick start — MCP tools (Claude Code / Cursor)

Once installed, all 13 tools are available inside your coding agent. A typical session:

```
# Start a session
start_session(tag="experiment-v3", agent_name="content-bot")
→ { "session_id": "550e8400-...", "started_at": "2026-04-06T10:00:00Z" }

# Log what the agent saw, considered, and chose
log_reasoning(session_id="550e...", content="Decided to use search_web for live data.", step=1)
log_tool_call(session_id="550e...", tool_name="search_web", inputs={"query": "AI"}, outputs={"results": [...]}, step=1)
log_decision(session_id="550e...", options_considered=["search_web", "query_db"], option_chosen="search_web", confidence=0.9)

# End session and get drift score
end_session(session_id="550e...")
→ {
    "composite_score": 0.42,
    "tool_pattern_score": 0.31,
    "semantic_score": 0.51,
    "complexity_score": 0.38,
    "severity": "warning",
    "baseline_confidence": "medium",
    "baseline_confidence_note": "Based on 6 sessions. Reasonably reliable.",
    "baseline_sessions_count": 6,
    "session_file": "~/.reasontrace/550e8400-....rtrace"
  }
```

---

## Composite drift score — how it works

Every session is scored 0.0 to 1.0 against a baseline built from all past sessions. Scoring begins from session 2. Session 1 initialises the baseline.

**Three components, each [0.0, 1.0]:**

**Component 1 — Tool Pattern Drift (weight: 0.35)**
Represents each session as a normalised tool-frequency vector. Computes cosine distance from the baseline centroid. Captures which tools were chosen, how often, and in what mix.

**Component 2 — Semantic Reasoning Drift (weight: 0.40)**
Embeds all reasoning checkpoint texts using `sentence-transformers/all-MiniLM-L6-v2` locally. Averages embeddings per session. Computes cosine distance from the baseline embedding centroid. No API calls — all local inference. Captures whether the agent's thinking is semantically consistent with its historical pattern.

**Component 3 — Decision Complexity Drift (weight: 0.25)**
Computes a complexity vector per session: [step count, reconsideration count, average options considered per decision]. Normalises against baseline distribution using z-scores, capped at 3 standard deviations = 1.0. Captures whether the agent is becoming more or less decisive and efficient.

**Composite = weighted average, clamped to [0.0, 1.0].**

**Example interpretation:**

```
Session 14 drift result:
  composite_score: 0.71          ← alert level
  tool_pattern_score: 0.08       ← tools unchanged
  semantic_score: 0.72           ← reasoning language shifted significantly
  complexity_score: 0.11         ← decision complexity unchanged

  primary_driver: semantic_reasoning
  interpretation: Agent's reasoning vocabulary shifted. Previously decisive
                  language ("selected X because...") replaced with hedging
                  language ("not sure whether X or Y might be better").
  recommendation: Check if system prompt changed between last stable session
                  and session 14. Specifically the instruction style.
```

---

## Drift severity thresholds

| Score | Severity | Meaning |
|-------|----------|---------|
| < 0.3 | **none** | Behavior consistent with baseline |
| 0.3–0.6 | **warning** | Reasoning drifting, worth investigating |
| 0.6–0.8 | **alert** | Significant behavioral change detected |
| > 0.8 | **critical** | Agent reasoning has fundamentally shifted |

---

## Confidence tiers — no hard session minimum

Scoring begins from session 2. There is no arbitrary minimum that blocks value. Every score response includes a `baseline_confidence` field:

| Baseline sessions | Confidence | Meaning |
|-------------------|------------|---------|
| 1 | `initializing` | No score yet — session stored only |
| 2–4 | `low` | Directional signal — treat as indicative, not definitive |
| 5–9 | `medium` | Reasonably reliable |
| 10+ | `high` | Statistically robust |

This means reasontrace is useful from day one inside Cursor or Claude Code, even if you only run 2–3 sessions on a task.

---

## All 13 MCP tools

### `start_session`
Begin capturing a new agent run.

```
Input:  tag (optional), agent_name (optional)
Output: { session_id, started_at, message }
```

### `log_reasoning`
Capture a reasoning checkpoint. The content text is embedded locally for semantic drift scoring.

```
Input:  session_id, content (free-form text), step (optional)
Output: { session_id, checkpoint_count, message }
```

### `log_tool_call`
Record a tool invocation.

```
Input:  session_id, tool_name, inputs (dict), outputs (dict), step (optional)
Output: { session_id, tool_call_count, message }
```

### `log_decision`
Capture a decision point with all options considered.

```
Input:  session_id, options_considered (list), option_chosen, confidence (0.0–1.0, optional), step (optional)
Output: { session_id, decision_count, message }
```

### `end_session`
Close the session. Compute composite drift score. Persist as `.rtrace` JSON. Update baseline.

```
Input:  session_id
Output: {
  composite_score, tool_pattern_score, semantic_score, complexity_score,
  severity, baseline_confidence, baseline_confidence_note,
  baseline_sessions_count, session_file, message
}
```

### `get_drift_score`
Return the full drift breakdown for any past session by ID.

```
Input:  session_id
Output: Full drift breakdown + session metadata (tag, timestamps, checkpoint counts)
```

### `explain_drift`
The killer feature. Structured plain-English explanation of drift for a session.

```
Input:  session_id
Output: {
  primary_driver,        ← which component drifted most
  tool_change_detail,    ← "agent switched from search_web to query_db"
  semantic_detail,       ← "reasoning shifted from decisive to hedging language"
  complexity_detail,     ← "step count increased from baseline avg 3.2 to 9"
  divergent_from,        ← which baseline sessions this diverged from most
  recommendation         ← "check if system prompt changed between session X and this one"
}
```

**Example output:**

```json
{
  "session_id": "550e8400-...",
  "composite_score": 0.71,
  "severity": "alert",
  "primary_driver": "semantic_reasoning",
  "tool_change_detail": "Tool usage is consistent with baseline.",
  "semantic_detail": "Reasoning text became semantically distant from baseline (cosine distance 0.72). Sample: 'Not entirely sure which approach is best here. Maybe search_web could work, though query_db might also be reasonable...'",
  "complexity_detail": "Decision complexity is consistent with baseline.",
  "divergent_from": ["session-id-8", "session-id-9", "session-id-10"],
  "recommendation": "Significant semantic reasoning drift. Inspect recent sessions for prompt or configuration changes. Use get_session to review full reasoning checkpoints."
}
```

### `get_baseline`
Return the current baseline profile.

```
Output: {
  sessions_count, confidence, average_step_count,
  average_tool_calls_per_session, most_common_tools,
  average_complexity_score, semantic_centroid_description, confidence_note
}
```

### `reset_baseline`
Reset baseline entirely or rebuild from a specified list of session IDs.

```
Input:  session_ids (list, optional) — if empty, wipe baseline; if provided, rebuild from those sessions
Output: { message, sessions_count, session_ids_used }
```

### `list_sessions`
Return all saved sessions with metadata, newest first.

```
Output: {
  sessions: [{ session_id, tag, agent_name, started_at, ended_at,
               composite_score, severity, baseline_confidence }],
  count
}
```

### `get_session`
Retrieve a full session by ID — all reasoning checkpoints, tool calls, decisions, and drift breakdown.

```
Input:  session_id
Output: Complete .rtrace session dict
```

### `search_sessions`
Find sessions matching any combination of filters (AND logic).

```
Input:  tag, agent_name, min_drift_score, max_drift_score, tool_name,
        keyword (in reasoning text), date_from, date_to — all optional
Output: { sessions: [...], count }
```

**Example:** Find all sessions that used `query_db` with drift > 0.5:

```
search_sessions(tool_name="query_db", min_drift_score=0.5)
```

### `delete_session`
Delete a session by ID. Removes the `.rtrace` file. Rebuilds baseline from remaining sessions.

```
Input:  session_id
Output: { message, deleted_session_id, baseline_sessions_count }
```

---

## The baseline — how it builds itself

Session 1 → baseline initialised, no drift score.
Session 2 → first drift score computed (confidence: `low`).
Session N → baseline updated incrementally via running mean after every session.

The baseline is stored at `~/.reasontrace/baseline.json` and updated after every `end_session` call using Welford's online algorithm — no need to reload all past sessions. Every `.rtrace` session file is human-readable JSON.

**Resetting the baseline:**

```
# Wipe entirely — next session re-initialises
reset_baseline()

# Rebuild from specific sessions — useful after cleaning out bad sessions
reset_baseline(session_ids=["session-id-1", "session-id-2", "session-id-5"])
```

---

## Anthropic SDK hook

Automatically capture tool calls from an Anthropic API client:

```python
import anthropic
from reasontrace import Tracer

client = anthropic.Anthropic()
tracer = Tracer()

session_id = tracer.start(tag="sdk-session")

# Hook the client — all tool_use blocks are auto-logged
tracer.attach_anthropic(client, session_id)

# Normal API calls — tool use is captured transparently
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    tools=[...],
    messages=[...],
)

tracer.end(session_id)
```

---

## Manual MCP config (if you skipped the prompt)

Add this to your coding agent's config file:

```json
{
  "mcpServers": {
    "reasontrace": {
      "command": "reasontrace-server",
      "type": "stdio"
    }
  }
}
```

Config file locations:

| Agent | Location |
|-------|----------|
| Claude Code | `~/.claude.json` |
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Claude Desktop (Linux) | `~/.config/Claude/claude_desktop_config.json` |
| Cursor (macOS/Linux) | `~/.cursor/mcp.json` |
| Cursor (Windows) | `%USERPROFILE%\.cursor\mcp.json` |

Run `reasontrace-install` anytime to configure interactively.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONTRACE_DIR` | `~/.reasontrace/` | Storage directory for `.rtrace` files and `baseline.json` |
| `REASONTRACE_WEIGHTS` | `0.35,0.40,0.25` | Comma-separated weights for [tool_pattern, semantic, complexity] components. Automatically normalised to sum to 1.0. |

**Example — emphasise semantic drift:**

```bash
REASONTRACE_WEIGHTS=0.2,0.6,0.2 reasontrace-server
```

---

## Storage format

Each session is stored as a human-readable `.rtrace` JSON file in `~/.reasontrace/`:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "tag": "prod-v2",
  "agent_name": "content-generator",
  "started_at": "2026-04-06T10:00:00Z",
  "ended_at": "2026-04-06T10:04:32Z",
  "reasoning_checkpoints": [
    {
      "step": 1,
      "content": "Received task. Decided to use search_web first to gather context.",
      "timestamp": "2026-04-06T10:00:12Z"
    }
  ],
  "tool_calls": [
    {
      "step": 1,
      "tool_name": "search_web",
      "inputs": {"query": "AI trends 2026"},
      "outputs": {"results": ["..."]},
      "timestamp": "2026-04-06T10:00:15Z"
    }
  ],
  "decisions": [
    {
      "step": 2,
      "options_considered": ["search_web", "query_db", "use_cache"],
      "option_chosen": "search_web",
      "confidence": 0.85,
      "timestamp": "2026-04-06T10:00:14Z"
    }
  ],
  "drift": {
    "composite_score": 0.42,
    "tool_pattern_score": 0.31,
    "semantic_score": 0.51,
    "complexity_score": 0.38,
    "severity": "warning",
    "baseline_confidence": "medium",
    "baseline_confidence_note": "Based on 6 sessions. Reasonably reliable.",
    "baseline_sessions_count": 6
  }
}
```

---

## Running tests

```bash
pip install pytest
pytest tests/
```

---

## Evaluation

Does drift detection actually work? See [EVALUATION.md](EVALUATION.md) for a full reproducible evaluation across five controlled scenarios — control (no drift), tool pattern drift, semantic drift, complexity drift, and combined drift.

Run it yourself:

```bash
python -m reasontrace.eval
```

No real LLM inference required — all sessions are generated deterministically with seed=42.

---

## Design principles

- **Local-first** — zero cloud, zero account, all data on disk as human-readable JSON
- **MCP-native** — Claude Code or Cursor IS the interface, no browser dashboard needed
- **pip install does everything** — interactive prompt configures MCP automatically
- **Useful from session 1** — no hard session gate, confidence tiers communicate reliability honestly
- **No embedded LLM calls** — sentence-transformers runs locally, no API key needed
- **Framework agnostic** — works with raw Anthropic SDK, LangChain, LangGraph, anything
- **Automatic drift detection** — no manual run comparison, baseline builds itself over time
- **Non-interactive safe** — installer detects CI/Docker and skips prompt gracefully
- **Atomic writes** — all config and storage file writes use temp-file-then-rename pattern

---

## License

MIT
