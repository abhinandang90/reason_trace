# reasontrace — Key Implementation Decisions

This file records important implementation decisions made during the build,
including bugs caught during verification and architectural choices that
deviate from or clarify the original spec.

---

## Packaging Fix 1 — Wrong Build Backend

**File:** `pyproject.toml`

**Problem:** The original `build-backend` was set to
`"setuptools.backends.legacy:build"`, which does not exist in setuptools
and causes an immediate build failure when running `python -m build`.

**Fix:** Changed to the correct value: `"setuptools.build_meta"`.

This is the standard setuptools PEP 517 build backend and is required for
`python -m build`, `pip wheel .`, and PyPI uploads via twine to work.

---

## Packaging Fix 2 — Removed PostInstallCommand Hook

**File:** `pyproject.toml`, `reasontrace/installer.py`

**Problem:** The original `pyproject.toml` included:

```toml
[tool.setuptools.cmdclass]
install = "reasontrace.installer:PostInstallCommand"
```

This attempts to override setuptools' `install` command to run the
interactive MCP config wizard at the end of `pip install`. It is broken
in all modern packaging environments for two reasons:

1. **Build isolation:** `pip` runs builds in an isolated virtual environment
   (PEP 517/518). The `reasontrace` package is not installed in that
   environment, so `from reasontrace.installer import PostInstallCommand`
   raises `ModuleNotFoundError` during the build step.

2. **Deprecation:** The `install` command is a legacy setuptools concept.
   Modern pip uses the wheel installer directly and does not invoke
   `setup.py install` or any `cmdclass` hooks.

**Fix:** Removed the `[tool.setuptools.cmdclass]` section entirely.

**How the installer still works:** The `reasontrace-install` entry point
is preserved. After `pip install reasontrace`, the user runs:

```bash
reasontrace-install
```

This calls `reasontrace.installer:install` directly — the same interactive
MCP config wizard — with no dependency on setuptools hooks. This is the
correct, modern approach.

---

## Packaging Fix 3 — README Content-Type for PyPI

**File:** `pyproject.toml`

**Problem:** The original `readme = "README.md"` string form does not tell
PyPI that the file is Markdown, causing the README to render as plain text
on the PyPI project page rather than formatted Markdown.

**Fix:** Changed to the explicit table form:

```toml
readme = {file = "README.md", content-type = "text/markdown"}
```

PyPI requires the `content-type` field to render Markdown correctly.

---

## Bug Fix 1 — Tool Pattern Scorer: Zero Vocabulary Overlap

**File:** `reasontrace/scorer.py`
**Function:** tool pattern drift computation

**Problem:** When a drifted session used zero tools from the baseline
vocabulary — i.e. entirely new tools never seen before — the cosine
distance computation returned 0.0 (no drift) because the session vector
was all zeros and cosine similarity of a zero vector is undefined,
defaulting to no distance.

**Fix:** Added an explicit check before cosine distance computation. If
the session tool frequency vector is all zeros (no overlap with baseline
vocabulary), return 1.0 (maximum drift) immediately. A session that uses
entirely different tools than anything in the baseline is by definition
maximally drifted on the tool pattern dimension.

**Implication for users:** If your agent completely switches tool sets
between baseline and new sessions, the tool pattern score will correctly
report 1.0 rather than silently reporting 0.0.

---

## Bug Fix 2 — Severity Boundary at 0.8

**File:** `reasontrace/scorer.py`
**Function:** severity label assignment

**Problem:** The original implementation used `>= 0.8 → critical`, but
the spec defines `> 0.8 → critical`, meaning a score of exactly 0.8
should be `alert`, not `critical`.

**Fix:** Changed boundary condition from `>= 0.8` to `> 0.8`.

**Correct severity mapping:**
- score < 0.3 → `"none"`
- 0.3 <= score < 0.6 → `"warning"`
- 0.6 <= score <= 0.8 → `"alert"`
- score > 0.8 → `"critical"`

**Implication for users:** A score of exactly 0.8 is reported as `alert`
not `critical`. This is the correct boundary per spec.

---

## Architectural Decision — Lazy Embedder Loading

**File:** `reasontrace/embedder.py`, `reasontrace/server.py`

**Decision:** `sentence-transformers` is not imported at module load time.
The embedder is loaded lazily — only when the MCP server actually runs
and a session ending triggers drift scoring.

**Reason:** `sentence-transformers` is a heavyweight dependency that
downloads a model on first use. Importing it eagerly would make the
test suite require the full model to be installed and downloaded, making
tests slow and environment-dependent.

**Implementation:** `embedder.py` wraps the import in a function-level
`import` statement inside `_get_model()`. The model is instantiated once
and cached as a module-level singleton after first load. This means the
first `end_session` call after install will trigger a one-time model
download (~90 MB), and all subsequent calls use the cached model.

---

## Architectural Decision — Deterministic Embedding Stub in Tests

**File:** `tests/conftest.py`

**Decision:** The test suite uses a deterministic embedding stub that
replaces the real `sentence-transformers` embedder for all tests. The
stub generates reproducible fake embeddings from input text using a
seeded hash function, without requiring `sentence-transformers` to be
installed.

**Reason:** Makes the full test suite runnable in any environment without
downloading the embedding model. Tests remain fast, deterministic, and
dependency-light. The real embedder is exercised only in integration tests
and the eval module.

**Implementation:** `conftest.py` patches `embed_texts` and
`embed_session_reasoning` in `reasontrace.embedder`, `reasontrace.baseline`,
and `reasontrace.scorer` at the start of every test using a pytest autouse
fixture. The stub uses `numpy.random.default_rng(hash(text) & 0xFFFFFFFF)`
so identical texts always produce identical unit-normalised vectors of
shape `(384,)`, and different texts produce different vectors — exactly
the contract the real embedder provides.

**Coverage:** All 6 test files run cleanly with the stub. The stub
produces vectors with the correct shape and normalisation so all scorer
logic is exercised correctly. The `sentence-transformers` import is never
triggered during `pytest` runs.

---

## Severity Thresholds — Full Reference

For clarity, the complete severity threshold mapping as implemented:

| Score Range         | Severity       | Meaning                                   |
|---------------------|----------------|-------------------------------------------|
| First session       | initializing   | No baseline yet, session stored only      |
| score < 0.3         | none           | Behavior consistent with baseline         |
| 0.3 ≤ score < 0.6   | warning        | Reasoning drifting, worth investigating   |
| 0.6 ≤ score ≤ 0.8   | alert          | Significant behavioral change detected    |
| score > 0.8         | critical       | Agent reasoning has fundamentally shifted |
