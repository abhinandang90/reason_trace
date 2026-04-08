# reasontrace — Live Lab (Browser Demo)

A standalone, interactive browser demo of the reasontrace drift scoring
engine. No install, no server, no Python required.

Open `index.html` in any modern browser and it works immediately.

---

## What this is

reasontrace detects when an AI agent starts reasoning differently — before
you notice it in the outputs. It scores every agent session against a
baseline built from past sessions, using three components: tool pattern
drift, semantic reasoning drift, and decision complexity drift.

This demo lets you watch that scoring happen live, session by session,
with five pre-built scenarios that each isolate a different type of drift.

**The installable Python package and MCP server live on the `main` branch.**
This demo is a companion visualization — the full reasontrace package is
what you actually use inside Claude Code or Cursor.

---

## How to run

```
open index.html
```

That's it. No dependencies to install. The demo runs entirely in the
browser using vanilla JavaScript and Chart.js (loaded from CDN).

---

## What's inside

| File | What it does |
|------|-------------|
| `index.html` | Layout — header, sidebar, chart area, detail panel |
| `reasontrace-core.js` | JS port of `scorer.py` + `baseline.py` — the full drift engine |
| `app.js` | UI logic, Chart.js integration, scenario playback |
| `style.css` | Dark theme, Inter font, severity colour system |

---

## The five scenarios

Each scenario runs 10–12 sessions with animated playback. Click any
session in the log or on the chart to inspect its full drift breakdown.

**No Drift** — 12 consistent sessions of a content research agent. All
composite scores stay below 0.3. Demonstrates the false positive rate.

**Tool Pattern Drift** — 6 baseline sessions using a web-search pipeline
(`search_web → extract_data → summarize`), then 6 drift sessions where the
agent switches to an internal database pipeline (`query_db → transform_data
→ format_output`). Tool component spikes; semantic and complexity stay flat.

**Semantic Drift** — 6 confident, decisive baseline sessions, then 6 where
the agent becomes uncertain and hedging. Same tools and step count. Semantic
component spikes; others stay flat.

**Complexity Drift** — 6 fast, direct baseline sessions (3–4 steps), then
6 where the agent enters decision paralysis (10+ steps, multiple
reconsiderations). Complexity component spikes; others stay flat.

**Full Drift** — 5 baseline sessions of a recommendation agent, then 7
sessions where all three components spike simultaneously — new toolset,
hedging reasoning, and far more steps. Composite score reaches alert level.

---

## How the JS scoring engine works

`reasontrace-core.js` is a faithful port of the Python scorer with one
deliberate simplification: semantic embeddings use a lightweight
keyword-based simulation rather than `sentence-transformers`. The engine
detects hedging words (`perhaps`, `maybe`, `uncertain`, `might`, etc.)
versus confident words (`clearly`, `decided`, `optimal`, etc.) and uses a
hash-seeded pseudo-random function for the remaining embedding dimensions.

This is sufficient to demonstrate semantic drift in a browser without any
ML dependencies. The production Python package uses real local embeddings
via `sentence-transformers/all-MiniLM-L6-v2`.

Everything else — tool pattern cosine distance, complexity z-scores,
Welford online variance for the baseline, incremental centroid updates,
confidence tiers, severity boundaries — is implemented identically to the
Python package, including both bug fixes documented in `DECISIONS.md`:

- Zero vocabulary overlap → tool score 1.0 (not 0.0)
- Severity boundary: score of exactly 0.8 is `alert`, not `critical`

---

## Relationship to the main package

| | Live Lab (this branch) | Python package (`main`) |
|--|----------------------|------------------------|
| Runs in | Browser | Terminal / Claude Code / Cursor |
| Embeddings | Keyword simulation | `sentence-transformers` (local) |
| Storage | In-memory only | `~/.reasontrace/*.rtrace` files |
| MCP tools | None | 13 tools via fastmcp |
| Install | Open HTML file | `pip install reasontrace` |

---

## Branch

This demo lives on the `demo/live-lab` branch.
The installable package is on `main`.
