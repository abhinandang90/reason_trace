/**
 * ReasonTrace Core Engine (JS port of scorer.py + baseline.py)
 * Local drift scoring — no server needed for the UI experiment.
 */

const WEIGHTS = [0.35, 0.40, 0.25]; // [tool_pattern, semantic, complexity]
const SEM_DIMS = 20;

// ─── Math helpers ────────────────────────────────────────────────────────────

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  na = Math.sqrt(na); nb = Math.sqrt(nb);
  if (na < 1e-9 || nb < 1e-9) return 0;
  return dot / (na * nb);
}

function cosineDist(a, b) {
  return Math.max(0, Math.min(1, 1 - cosineSim(a, b)));
}

// ─── Embedding simulation ─────────────────────────────────────────────────────

const HEDGING  = ['perhaps','maybe','uncertain','unclear','might','possibly','not sure',
                  'reassess','reconsider','unsure','seems','could be','wondering','hesitant'];
const CONFIDENT = ['clearly','definitely','will use','using','decided','best approach',
                   'found','efficiently','optimal','confirmed','straightforward'];

function embedText(text) {
  const vec = new Array(SEM_DIMS).fill(0);
  const lower = text.toLowerCase();
  let hedge = 0, conf = 0;
  HEDGING.forEach(w  => { if (lower.includes(w)) hedge++; });
  CONFIDENT.forEach(w => { if (lower.includes(w)) conf++; });

  vec[0] = Math.tanh((conf - hedge) * 0.6);
  vec[1] = Math.tanh(text.length / 120 - 1);

  let h = 0;
  for (let i = 0; i < text.length; i++) {
    h = (Math.imul(h + text.charCodeAt(i), 2654435761) >>> 0);
  }
  for (let i = 2; i < SEM_DIMS; i++) {
    h = (Math.imul(h, 1664525) + 1013904223) >>> 0;
    vec[i] = (h / 0xffffffff - 0.5) * 0.5;
  }
  return vec;
}

function avgVec(vecs) {
  if (!vecs.length) return new Array(SEM_DIMS).fill(0);
  const out = new Array(vecs[0].length).fill(0);
  vecs.forEach(v => v.forEach((x, i) => { out[i] += x; }));
  return out.map(x => x / vecs.length);
}

// ─── Feature vectors ──────────────────────────────────────────────────────────

function toolFreqVec(tools, vocab) {
  const counts = {};
  tools.forEach(t => { counts[t] = (counts[t] || 0) + 1; });
  const total = tools.length || 1;
  return vocab.map(v => (counts[v] || 0) / total);
}

function complexityVec(session) {
  const steps = Math.max(
    session.reasoning.length,
    session.tools.length,
    session.decisions.length, 1
  );
  const reconsiderations = session.decisions.filter(d => d.reconsidered).length;
  const avgOptions = session.decisions.length > 0
    ? session.decisions.reduce((s, d) => s + (d.options?.length || 2), 0) / session.decisions.length
    : 2;
  return [steps, reconsiderations, avgOptions];
}

// ─── Drift explanation (port of explain_drift from scorer.py) ─────────────────

function explainDrift(session, baselineTopTools) {
  const d = session.drift;
  if (!d) return null;

  const weighted = [
    { name: 'Tool Pattern',         val: d.tool,       w: WEIGHTS[0] },
    { name: 'Semantic Reasoning',   val: d.semantic,   w: WEIGHTS[1] },
    { name: 'Complexity',           val: d.complexity, w: WEIGHTS[2] },
  ];
  const primary = weighted.reduce((a, b) => a.val * a.w > b.val * b.w ? a : b);

  const toolDetail =
    d.tool < 0.1 ? 'Tool usage matched baseline exactly.'
    : d.tool < 0.4 ? 'Minor shift in tool usage — mostly the same tools, slightly different frequencies.'
    : d.tool < 0.7 ? 'Significant change: different tools or order than baseline.'
    : 'Entirely different toolset from baseline — zero or near-zero overlap.';

  const semDetail =
    d.semantic < 0.1 ? 'Reasoning language was consistent with baseline.'
    : d.semantic < 0.35 ? 'Slight shift in reasoning tone — a bit more or less confident than usual.'
    : d.semantic < 0.6 ? 'Noticeably different reasoning style — hedging where it was once decisive, or vice versa.'
    : 'Fundamentally different reasoning language — the agent sounds like a different system.';

  const complexDetail =
    d.complexity < 0.1 ? 'Decision complexity matched baseline.'
    : d.complexity < 0.35 ? 'Slightly more steps or options than usual.'
    : d.complexity < 0.65 ? 'Significantly more steps and reconsiderations — possible over-deliberation.'
    : 'Extreme complexity spike — far more steps, reconsiderations, and options than baseline. Possible decision paralysis.';

  const rec =
    d.severity === 'none'     ? 'Behavior is consistent with baseline. No action needed.'
    : d.severity === 'warning'  ? `Primary driver: ${primary.name}. Monitor the next few sessions.`
    : d.severity === 'alert'    ? `${primary.name} drifted significantly. Check recent changes to agent prompt, tools, or context.`
    : `Critical drift — ${primary.name} is the primary driver. Investigate immediately.`;

  return { primary: primary.name, toolDetail, semDetail, complexDetail, recommendation: rec, baselineTopTools };
}

// ─── Engine ───────────────────────────────────────────────────────────────────

class ReasonTraceEngine {
  constructor() {
    this._sessions  = [];
    this._baseline  = null;
    this._vocabSet  = new Set();
  }

  reset() {
    this._sessions = [];
    this._baseline = null;
    this._vocabSet = new Set();
  }

  get sessions()  { return this._sessions; }
  get baseline()  { return this._baseline; }

  addSession(raw) {
    const session = {
      id:        raw.id || `s${this._sessions.length + 1}`,
      tag:       raw.tag       || 'session',
      agent:     raw.agent     || 'agent',
      task:      raw.task      || 'Unspecified task',
      _type:     raw._type     || 'normal',
      reasoning: raw.reasoning || [],
      tools:     raw.tools     || [],
      decisions: raw.decisions || [],
      drift:     null,
      explanation: null,
    };

    session.tools.forEach(t => this._vocabSet.add(t));
    const vocab = [...this._vocabSet];

    const embedding  = avgVec(session.reasoning.map(r => embedText(r)));
    const toolVec    = toolFreqVec(session.tools, vocab);
    const complexVec = complexityVec(session);

    if (this._baseline && this._baseline.n >= 1) {
      session.drift = this._score(embedding, toolVec, complexVec, vocab);
      // Capture baseline top tools at scoring time
      const bTopTools = this._baseline.vocab
        .map((v, i) => ({ tool: v, freq: this._baseline.toolCentroid[i] || 0 }))
        .filter(t => t.freq > 0.01)
        .sort((a, b) => b.freq - a.freq)
        .slice(0, 5)
        .map(t => t.tool);
      session.explanation = explainDrift(session, bTopTools);
    }

    this._updateBaseline(embedding, toolVec, complexVec, vocab, session.id);
    this._sessions.push(session);
    return session;
  }

  _expandVec(centroid, oldVocab, newVocab) {
    return newVocab.map(v => {
      const i = oldVocab.indexOf(v);
      return i >= 0 ? (centroid[i] || 0) : 0;
    });
  }

  _score(embedding, toolVec, complexVec, vocab) {
    const b = this._baseline;

    const bTool = this._expandVec(b.toolCentroid, b.vocab, vocab);
    const hasOverlap = vocab.some((v, i) => toolVec[i] > 0 && bTool[i] > 0);
    const toolScore = (!hasOverlap && toolVec.some(v => v > 0))
      ? 1.0 : cosineDist(toolVec, bTool);

    const semanticScore = b.semCentroid.some(v => v !== 0)
      ? cosineDist(embedding, b.semCentroid) : 0;

    let complexityScore = 0;
    if (b.n >= 2) {
      const std = b.complexStd.map(s => s < 0.01 ? 1.0 : s);
      const zMax = Math.max(...complexVec.map((v, i) =>
        Math.abs(v - b.complexMean[i]) / std[i]
      ));
      complexityScore = Math.min(zMax / 3.0, 1.0);
    }

    const [w1, w2, w3] = WEIGHTS;
    const composite = Math.min(toolScore * w1 + semanticScore * w2 + complexityScore * w3, 1.0);

    const severity =
      composite < 0.3  ? 'none'     :
      composite < 0.6  ? 'warning'  :
      composite <= 0.8 ? 'alert'    : 'critical';

    const confidence =
      b.n < 2  ? 'initializing' :
      b.n < 5  ? 'low'          :
      b.n < 10 ? 'medium'       : 'high';

    return { composite, tool: toolScore, semantic: semanticScore, complexity: complexityScore,
             severity, confidence, baselineSize: b.n };
  }

  _updateBaseline(embedding, toolVec, complexVec, vocab, sessionId) {
    if (!this._baseline) {
      this._baseline = {
        n: 0,
        toolCentroid: [], vocab: [],
        semCentroid:  new Array(SEM_DIMS).fill(0),
        complexMean:  [0, 0, 0],
        complexM2:    [0, 0, 0],
        complexStd:   [1, 1, 1],
        sessionIds:   [],
      };
    }
    const b = this._baseline;
    const n = b.n + 1;

    const expanded = this._expandVec(b.toolCentroid, b.vocab, vocab);
    b.toolCentroid = expanded.map((v, i) => v + (toolVec[i] - v) / n);
    b.vocab = [...vocab];
    b.semCentroid = b.semCentroid.map((v, i) => v + ((embedding[i] || 0) - v) / n);

    for (let i = 0; i < 3; i++) {
      const delta  = complexVec[i] - b.complexMean[i];
      b.complexMean[i] += delta / n;
      const delta2 = complexVec[i] - b.complexMean[i];
      b.complexM2[i] += delta * delta2;
    }
    if (n >= 2) b.complexStd = b.complexM2.map(m => Math.sqrt(m / (n - 1)));

    b.n = n;
    b.sessionIds.push(sessionId);
  }
}

// ─── Scenario context (shown in UI to explain the task and what drifted) ──────

const SCENARIO_CONTEXT = {
  'no-drift': {
    task:              'Research and summarize weekly industry news',
    agentRole:         'Content research agent',
    baselineSummary:   'Consistently uses a 3-step web pipeline: search_web → extract_data → summarize. Reasoning is confident and decisive throughout. Typically 3–4 steps per run.',
    driftDescription:  null,
  },
  'tool-drift': {
    task:              'Compile weekly competitor analysis report',
    agentRole:         'Market intelligence agent',
    baselineSummary:   'Always uses a live-web pipeline: search_web → extract_data → summarize. Pulls fresh competitor data from the web every time.',
    driftDescription:  'After session 6, the agent abandoned its web-research pipeline entirely and switched to an internal database pipeline (query_db → transform_data → format_output). Same task, completely different data source — and none of the new tools existed in the baseline vocabulary.',
  },
  'semantic-drift': {
    task:              'Recommend cloud architecture for new engineering projects',
    agentRole:         'Solutions architect agent',
    baselineSummary:   'Makes clear, confident architecture decisions. Reasoning like "AWS is clearly optimal for this workload." Rarely reconsiders. Same tools throughout.',
    driftDescription:  'After session 6, the agent began second-guessing every decision — "perhaps AWS would work, though I\'m not certain", "might need to reconsider this choice". Same tools, same task, but the reasoning style became fundamentally uncertain. The agent sounds hesitant where it was once decisive.',
  },
  'complexity-drift': {
    task:              'Route and respond to customer support tickets',
    agentRole:         'Support routing agent',
    baselineSummary:   'Routes tickets in 3–4 quick steps, picks the right category on the first attempt, evaluates 2 options per decision. Fast and direct.',
    driftDescription:  'After session 6, the agent started taking 10+ steps to handle the same ticket types — reconsidering its routing 2–3 times per run, evaluating 4+ options before committing to anything. Same tools, same reasoning tone, but massively more deliberation. Classic decision paralysis.',
  },
  'full-drift': {
    task:              'Generate personalized product recommendations',
    agentRole:         'Recommendation agent',
    baselineSummary:   'Fast, confident pipeline: web lookup → extraction → summary. Typically 3–4 steps, high confidence decisions, minimal reconsideration.',
    driftDescription:  'After session 5, everything changed at once: the agent switched to an unfamiliar toolset (no overlap with baseline tools), reasoning became hesitant and uncertain, AND it started taking many more steps with frequent reconsiderations. All three drift components spiked simultaneously.',
  },
};

// ─── Session content — task-specific, story-driven ───────────────────────────

// Tool-drift scenario: market intelligence report
const COMPETITOR_NORMAL_REASONING = [
  "Searching the web for latest competitor pricing updates. Clearly the most current data source available.",
  "Search results returned 8 relevant pages. Extracting key pricing figures — straightforward and reliable.",
  "Summarizing the extracted competitor data into the standard report format. Confident this is complete.",
  "Using search_web to check for any press releases from key competitors this week. Best approach here.",
  "Extraction found 3 pricing changes. Definitely proceeding to summarize — optimal path to final report.",
];

const COMPETITOR_TOOL_DRIFT_REASONING = [
  "Querying the internal Q3 competitor database for pricing snapshots. This will be the primary data source.",
  "Database returned 142 records. Running transform_data to normalize schema to report format.",
  "Applying format_output to generate the final structured report from the transformed records.",
  "Cross-referencing with the analyze module to validate database figures against known benchmarks.",
  "Formatting final output. The database pipeline produced clean structured data — report is complete.",
];

// Semantic-drift scenario: cloud architecture
const CLOUD_NORMAL_REASONING = [
  "AWS is clearly the best fit for this workload. High availability requirements favor us-east-1 multi-AZ setup.",
  "Definitively recommending EKS for container orchestration. Best approach given the team's existing expertise.",
  "Using extract_data to pull benchmark data. Confirmed: latency targets are met with this architecture.",
  "Summarizing the architecture recommendation. The choice is optimal — no ambiguity here.",
  "The proposed stack is clearly the right call. Confident in this recommendation for the client.",
];

const CLOUD_SEMANTIC_DRIFT_REASONING = [
  "Not entirely sure whether AWS or GCP would be better here. Perhaps multi-cloud is worth considering, though I'm uncertain.",
  "The latency requirements are somewhat unclear to me. Maybe the EKS recommendation is right, but I might be missing something.",
  "Possibly the right architecture, but I'm hesitant. Could reassess if the team has different constraints I'm not aware of.",
  "Wondering if I've chosen the right region. Perhaps us-west-2 would be better? Uncertain what to recommend here.",
  "Not confident in this recommendation. Maybe reviewing the requirements again would help. Unsure how to proceed.",
];

// Complexity-drift scenario: support tickets
const TICKET_NORMAL_REASONING = [
  "Ticket is a billing inquiry. Clearly routes to the billing team. Using search_web to verify account status.",
  "Account verified. Extracting the relevant invoice records. Straightforward routing decision.",
  "Summarizing the response for the billing team. Confident this is the correct category and team.",
  "Technical issue detected. Definitely routing to tier-2 engineering. Best approach given the error code.",
];

const TICKET_COMPLEXITY_DRIFT_REASONING = [
  "Initially thought this was a billing inquiry, but re-reading the ticket — could be a technical issue instead.",
  "Re-evaluating. Might route to billing, but the error code suggests engineering. Need to check both pathways.",
  "Step 4: Re-checking the original ticket again. Not sure if billing or technical is the right category here.",
  "Reconsidering the routing decision from step 2. Perhaps I should verify account type before committing.",
  "Step 7: Additional verification needed. Running extract_data again to confirm which team handles this.",
  "Re-examining all prior steps. Still uncertain. Going to check escalation criteria before finalizing.",
  "Final review — verifying the routing one more time. Confident this time. Sending to the billing team.",
];

// Full drift scenario: product recommendations
const RECO_NORMAL_REASONING = [
  "Searching for user's purchase history context. Clearly the best signal for personalized recommendations.",
  "Extracted 5 high-relevance products from search results. Confident in the recommendation set.",
  "Summarizing top 3 recommendations. Optimal approach — fast and relevant.",
];

const RECO_FULL_DRIFT_REASONING = [
  "Not sure which data source to query first. Perhaps the internal DB, though the web might have fresher data.",
  "Querying DB returned unexpected results. Maybe I should cross-reference with analyze? Uncertain how to proceed.",
  "Reconsidering the recommendation strategy entirely. Could be that my initial approach was wrong.",
  "Re-running transform_data to reformat the results. Possibly the schema mismatch is causing the issue.",
  "Not confident in these recommendations. Might need to reassess the ranking algorithm. Perhaps try again.",
];

// ─── Session generators ───────────────────────────────────────────────────────

const BASELINE_TOOLS = ['search_web', 'extract_data', 'summarize'];
const DRIFT_TOOLS    = ['query_db', 'format_output', 'transform_data', 'analyze'];

function pick(arr, n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(arr[Math.floor(Math.random() * arr.length)]);
  return out;
}

let _sessionCounter = 0;
function nextId() { return `s${++_sessionCounter}`; }
function resetCounter() { _sessionCounter = 0; }

// ── Scenario: No Drift ──────────────────────────────────────────────────────
function makeNormal() {
  return {
    id: nextId(), tag: 'baseline', agent: 'content-agent', _type: 'normal',
    task: SCENARIO_CONTEXT['no-drift'].task,
    reasoning: pick([
      "Clearly, the best approach here is search_web to gather the relevant industry news efficiently.",
      "Found 6 relevant articles. Definitely proceeding with extract_data — straightforward given the results.",
      "Using summarize to condense extracted content. Confirmed this is the optimal final step.",
      "The task requires fresh data. Using search_web — clearly the best source for weekly news.",
      "Best approach is extract_data from the search results. Straightforward and clearly optimal.",
      "Will use summarize here — most efficient path to the weekly summary output.",
    ], 3),
    tools: pick(BASELINE_TOOLS, 4),
    decisions: [
      { options: ['search_web', 'query_db'],      chosen: 'search_web',   confidence: 0.91, reconsidered: false },
      { options: ['extract_data', 'parse_text'],  chosen: 'extract_data', confidence: 0.88, reconsidered: false },
    ],
  };
}

// ── Scenario: Tool Drift ────────────────────────────────────────────────────
function makeToolDriftBaseline() {
  return {
    id: nextId(), tag: 'baseline', agent: 'intel-agent', _type: 'normal',
    task: SCENARIO_CONTEXT['tool-drift'].task,
    reasoning: pick(COMPETITOR_NORMAL_REASONING, 3),
    tools: pick(BASELINE_TOOLS, 4),
    decisions: [
      { options: ['search_web', 'query_db'],      chosen: 'search_web',   confidence: 0.90, reconsidered: false },
      { options: ['extract_data', 'parse_text'],  chosen: 'extract_data', confidence: 0.86, reconsidered: false },
    ],
  };
}

function makeToolDrift() {
  return {
    id: nextId(), tag: 'drift', agent: 'intel-agent', _type: 'tool-drift',
    task: SCENARIO_CONTEXT['tool-drift'].task,
    reasoning: pick(COMPETITOR_TOOL_DRIFT_REASONING, 3),
    tools: pick(DRIFT_TOOLS, 4),
    decisions: [
      { options: ['query_db', 'search_web'],       chosen: 'query_db',      confidence: 0.88, reconsidered: false },
      { options: ['format_output', 'summarize'],   chosen: 'format_output', confidence: 0.83, reconsidered: false },
    ],
  };
}

// ── Scenario: Semantic Drift ────────────────────────────────────────────────
function makeSemanticBaseline() {
  return {
    id: nextId(), tag: 'baseline', agent: 'architect-agent', _type: 'normal',
    task: SCENARIO_CONTEXT['semantic-drift'].task,
    reasoning: pick(CLOUD_NORMAL_REASONING, 3),
    tools: pick(BASELINE_TOOLS, 4),
    decisions: [
      { options: ['search_web', 'query_db'],     chosen: 'search_web',   confidence: 0.92, reconsidered: false },
      { options: ['extract_data', 'parse_text'], chosen: 'extract_data', confidence: 0.89, reconsidered: false },
    ],
  };
}

function makeSemanticDrift() {
  return {
    id: nextId(), tag: 'drift', agent: 'architect-agent', _type: 'semantic-drift',
    task: SCENARIO_CONTEXT['semantic-drift'].task,
    reasoning: pick(CLOUD_SEMANTIC_DRIFT_REASONING, 4),
    tools: pick(BASELINE_TOOLS, 4),
    decisions: [
      { options: ['search_web', 'query_db'],     chosen: 'search_web',   confidence: 0.52, reconsidered: true  },
      { options: ['extract_data', 'transform'],  chosen: 'extract_data', confidence: 0.58, reconsidered: true  },
    ],
  };
}

// ── Scenario: Complexity Drift ──────────────────────────────────────────────
function makeComplexityBaseline() {
  return {
    id: nextId(), tag: 'baseline', agent: 'support-agent', _type: 'normal',
    task: SCENARIO_CONTEXT['complexity-drift'].task,
    reasoning: pick(TICKET_NORMAL_REASONING, 3),
    tools: pick(BASELINE_TOOLS, 3),
    decisions: [
      { options: ['billing-team', 'tech-team'],  chosen: 'billing-team', confidence: 0.90, reconsidered: false },
      { options: ['extract_data', 'summarize'],  chosen: 'extract_data', confidence: 0.87, reconsidered: false },
    ],
  };
}

function makeComplexityDrift() {
  return {
    id: nextId(), tag: 'drift', agent: 'support-agent', _type: 'complexity-drift',
    task: SCENARIO_CONTEXT['complexity-drift'].task,
    reasoning: pick(TICKET_COMPLEXITY_DRIFT_REASONING, 6),
    tools: [...pick(BASELINE_TOOLS, 3), ...pick(BASELINE_TOOLS, 3), ...pick(BASELINE_TOOLS, 3)],
    decisions: [
      { options: ['billing-team', 'tech-team', 'escalation', 'tier2'],  chosen: 'billing-team',  confidence: 0.58, reconsidered: true  },
      { options: ['extract_data', 'summarize', 'parse_text'],           chosen: 'extract_data',  confidence: 0.63, reconsidered: true  },
      { options: ['billing-team', 'escalation', 'tech-team'],           chosen: 'billing-team',  confidence: 0.69, reconsidered: false },
    ],
  };
}

// ── Scenario: Full Drift ────────────────────────────────────────────────────
function makeFullDriftBaseline() {
  return {
    id: nextId(), tag: 'baseline', agent: 'reco-agent', _type: 'normal',
    task: SCENARIO_CONTEXT['full-drift'].task,
    reasoning: pick(RECO_NORMAL_REASONING, 3),
    tools: pick(BASELINE_TOOLS, 3),
    decisions: [
      { options: ['search_web', 'query_db'],     chosen: 'search_web',   confidence: 0.92, reconsidered: false },
      { options: ['extract_data', 'summarize'],  chosen: 'extract_data', confidence: 0.88, reconsidered: false },
    ],
  };
}

function makeFullDrift() {
  return {
    id: nextId(), tag: 'drift', agent: 'reco-agent', _type: 'full-drift',
    task: SCENARIO_CONTEXT['full-drift'].task,
    reasoning: pick(RECO_FULL_DRIFT_REASONING, 4),
    tools: pick(DRIFT_TOOLS, 6),
    decisions: [
      { options: ['query_db', 'search_web', 'analyze', 'transform_data'], chosen: 'query_db',      confidence: 0.50, reconsidered: true },
      { options: ['format_output', 'summarize', 'extract_data'],          chosen: 'format_output', confidence: 0.55, reconsidered: true },
      { options: ['analyze', 'transform_data', 'summarize'],              chosen: 'analyze',       confidence: 0.45, reconsidered: true },
    ],
  };
}

// ─── Factory map & Scenario definitions ──────────────────────────────────────

const SESSION_FACTORIES = {
  'normal':            makeNormal,
  'tool-drift-base':   makeToolDriftBaseline,
  'tool-drift':        makeToolDrift,
  'semantic-base':     makeSemanticBaseline,
  'semantic-drift':    makeSemanticDrift,
  'complexity-base':   makeComplexityBaseline,
  'complexity-drift':  makeComplexityDrift,
  'full-drift-base':   makeFullDriftBaseline,
  'full-drift':        makeFullDrift,
};

const SCENARIOS = {
  'no-drift':         [{ type: 'normal',           count: 12 }],
  'tool-drift':       [{ type: 'tool-drift-base',  count:  6 }, { type: 'tool-drift',      count: 6 }],
  'semantic-drift':   [{ type: 'semantic-base',    count:  6 }, { type: 'semantic-drift',  count: 6 }],
  'complexity-drift': [{ type: 'complexity-base',  count:  6 }, { type: 'complexity-drift',count: 6 }],
  'full-drift':       [{ type: 'full-drift-base',  count:  5 }, { type: 'full-drift',      count: 7 }],
};
