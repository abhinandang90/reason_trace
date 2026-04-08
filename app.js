/**
 * ReasonTrace Live Lab — UI + Chart Logic
 */

const engine = new ReasonTraceEngine();
let chart = null;
let selectedSession = null;
let activeTab = 'reasoning';
let isRunning = false;
let currentScenarioKey = null;

// ─── Chart ────────────────────────────────────────────────────────────────────

const backgroundBandsPlugin = {
  id: 'backgroundBands',
  beforeDraw(ch) {
    const { ctx, chartArea, scales } = ch;
    if (!chartArea) return;
    const { left, right } = chartArea;
    const y = scales.y;
    [
      { from: 0,   to: 0.3, color: 'rgba(34,197,94,0.06)'  },
      { from: 0.3, to: 0.6, color: 'rgba(245,158,11,0.07)' },
      { from: 0.6, to: 0.8, color: 'rgba(249,115,22,0.07)' },
      { from: 0.8, to: 1.0, color: 'rgba(239,68,68,0.08)'  },
    ].forEach(({ from, to, color }) => {
      const yT = y.getPixelForValue(to);
      const yB = y.getPixelForValue(from);
      ctx.save(); ctx.fillStyle = color;
      ctx.fillRect(left, yT, right - left, yB - yT);
      ctx.restore();
    });
  }
};

function initChart() {
  const ctx = document.getElementById('drift-chart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    plugins: [backgroundBandsPlugin],
    data: {
      labels: [],
      datasets: [
        { label: 'Composite', data: [], borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.1)',
          borderWidth: 2.5, pointRadius: 5, pointHoverRadius: 7, tension: 0.3,
          pointBackgroundColor: [], pointBorderColor: [] },
        { label: 'Tool',      data: [], borderColor: '#06b6d4', borderWidth: 1.5,
          borderDash: [5,4],  pointRadius: 3, tension: 0.3, pointBackgroundColor: '#06b6d4' },
        { label: 'Semantic',  data: [], borderColor: '#a78bfa', borderWidth: 1.5,
          borderDash: [5,4],  pointRadius: 3, tension: 0.3, pointBackgroundColor: '#a78bfa' },
        { label: 'Complexity',data: [], borderColor: '#f472b6', borderWidth: 1.5,
          borderDash: [5,4],  pointRadius: 3, tension: 0.3, pointBackgroundColor: '#f472b6' },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 350 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e2235', borderColor: '#2e3450', borderWidth: 1,
          titleColor: '#e2e8f0', bodyColor: '#7c87a0', padding: 10,
          callbacks: {
            title: items => {
              const s = engine.sessions[items[0].index];
              return s ? `Session ${items[0].index + 1} · ${s.task}` : `Session ${items[0].index + 1}`;
            },
            label: item => {
              const v = item.parsed.y;
              const labels = ['Composite','Tool','Semantic','Complexity'];
              return v == null ? `  ${labels[item.datasetIndex]}: baseline` :
                                 `  ${labels[item.datasetIndex]}: ${v.toFixed(3)}`;
            }
          }
        }
      },
      scales: {
        x: { grid: { color: 'rgba(46,52,80,0.5)' },
             ticks: { color: '#7c87a0', font: { size: 10 } },
             title: { display: true, text: 'Session #', color: '#7c87a0', font: { size: 10 } } },
        y: { min: 0, max: 1,
             grid: { color: 'rgba(46,52,80,0.5)' },
             ticks: { color: '#7c87a0', font: { size: 10 }, callback: v => v.toFixed(1) },
             title: { display: true, text: 'Drift Score', color: '#7c87a0', font: { size: 10 } } }
      },
      onClick(_, elements) {
        if (!elements.length) return;
        const s = engine.sessions[elements[0].index];
        if (s) selectSession(s);
      }
    }
  });
}

const SEV_COLOR = { none:'#22c55e', warning:'#f59e0b', alert:'#f97316', critical:'#ef4444' };

function updateChart() {
  const ss = engine.sessions;
  chart.data.labels    = ss.map((_, i) => i + 1);
  chart.data.datasets[0].data = ss.map(s => s.drift?.composite ?? null);
  chart.data.datasets[1].data = ss.map(s => s.drift?.tool      ?? null);
  chart.data.datasets[2].data = ss.map(s => s.drift?.semantic  ?? null);
  chart.data.datasets[3].data = ss.map(s => s.drift?.complexity?? null);
  const pts = ss.map(s => s.drift ? SEV_COLOR[s.drift.severity] : '#6366f1');
  chart.data.datasets[0].pointBackgroundColor = pts;
  chart.data.datasets[0].pointBorderColor     = pts;
  chart.update();
}

// ─── Session log ──────────────────────────────────────────────────────────────

const TYPE_LABEL = {
  'normal':           'Baseline',
  'tool-drift':       'Tool Drift',
  'semantic-drift':   'Semantic Drift',
  'complexity-drift': 'Complexity Drift',
  'full-drift':       'Full Drift',
};

function renderLog() {
  const log  = document.getElementById('session-log');
  const ss   = engine.sessions;
  if (!ss.length) {
    log.innerHTML = `<div class="log-empty">No sessions yet.<br>Run a scenario or add sessions manually.</div>`;
    return;
  }
  log.innerHTML = ss.map((s, i) => {
    const sel  = selectedSession?.id === s.id;
    const isN  = i === ss.length - 1;
    const type = TYPE_LABEL[s._type] || s._type;
    if (!s.drift) return `
      <div class="log-item${sel?' selected':''}${isN?' new':''}" data-id="${s.id}">
        <span class="session-num">${i+1}</span>
        <span class="session-tag">${type}</span>
        <span class="baseline-badge">baseline</span>
      </div>`;
    return `
      <div class="log-item${sel?' selected':''}${isN?' new':''}" data-id="${s.id}">
        <span class="session-num">${i+1}</span>
        <span class="session-tag">${type}</span>
        <span class="severity-dot" style="background:${SEV_COLOR[s.drift.severity]}"></span>
        <span class="score-text">${s.drift.composite.toFixed(3)}</span>
      </div>`;
  }).join('');
  log.querySelectorAll('.log-item').forEach(el =>
    el.addEventListener('click', () => {
      const s = engine.sessions.find(s => s.id === el.dataset.id);
      if (s) selectSession(s);
    })
  );
  log.scrollTop = log.scrollHeight;
}

// ─── Header ───────────────────────────────────────────────────────────────────

function updateHeader() {
  const n = engine.sessions.length;
  document.getElementById('session-count').textContent = `${n} session${n===1?'':'s'}`;
  const conf = !engine.baseline ? 'initializing'
    : engine.baseline.n < 2  ? 'initializing'
    : engine.baseline.n < 5  ? 'low'
    : engine.baseline.n < 10 ? 'medium' : 'high';
  document.getElementById('confidence-label').textContent = `Baseline: ${conf}`;
  const badge = document.getElementById('confidence-badge');
  badge.className = `confidence-badge ${conf === 'initializing' ? '' : conf}`;
}

// ─── Detail panel ─────────────────────────────────────────────────────────────

function selectSession(session) {
  selectedSession = session;
  renderLog();
  renderDetail(session);
}

function renderDetail(s) {
  document.getElementById('detail-empty').style.display   = 'none';
  document.getElementById('detail-content').style.display = 'block';

  renderContextStrip(s);
  renderScores(s);
  renderExplanation(s);
  renderTabContent(s, activeTab);
}

// ── Context strip: task + agent + what was done ──────────────────────────────

function renderContextStrip(s) {
  const strip = document.getElementById('context-strip');
  const ctx   = currentScenarioKey ? SCENARIO_CONTEXT[currentScenarioKey] : null;
  const typeLabel = TYPE_LABEL[s._type] || s._type;
  const sevBadge  = s.drift
    ? `<span class="severity-badge ${s.drift.severity}">${s.drift.severity}</span>`
    : `<span class="severity-badge baseline">baseline</span>`;

  strip.innerHTML = `
    <div class="context-meta">
      <span class="context-task-label">Task</span>
      <span class="context-task">${esc(s.task)}</span>
    </div>
    <div class="context-meta">
      <span class="context-task-label">Agent</span>
      <span class="context-task">${esc(s.agent)}</span>
    </div>
    <div class="context-meta">
      <span class="context-task-label">Session type</span>
      ${sevBadge}
      <span class="context-type-label">${typeLabel}</span>
    </div>
    ${ctx ? `
    <div class="context-baseline-box">
      <span class="ctx-label">Normal behavior:</span> ${esc(ctx.baselineSummary)}
    </div>` : ''}`;
}

// ── Drift scores card ────────────────────────────────────────────────────────

function renderScores(s) {
  const panel = document.getElementById('drift-scores-panel');
  if (!s.drift) {
    panel.innerHTML = `
      <div class="drift-card-title">Drift Scores <span class="severity-badge baseline">Baseline</span></div>
      <div style="color:var(--muted);font-size:11px;line-height:1.8;margin-top:8px">
        Session <strong style="color:var(--text)">${engine.sessions.indexOf(s)+1}</strong> — establishes the baseline.<br>
        Drift scoring begins from session 2 onward as the baseline accumulates.
      </div>
      <div class="confidence-note">Baseline size: ${engine.baseline?.n ?? 0} session(s)</div>`;
    return;
  }
  const d = s.drift;
  const bar = (val, col) => `
    <div class="score-bar-bg">
      <div class="score-bar-fill" style="width:${(val*100).toFixed(1)}%;background:${col}"></div>
    </div>`;
  panel.innerHTML = `
    <div class="drift-card-title">Drift Scores <span class="severity-badge ${d.severity}">${d.severity}</span></div>
    <div class="composite-score" style="color:${SEV_COLOR[d.severity]}">${d.composite.toFixed(3)}</div>
    <div style="color:var(--muted);font-size:10px;margin-bottom:12px">composite drift score</div>
    <div class="score-row">
      <div class="score-row-header">
        <span class="score-label">Tool Pattern <span style="color:var(--muted);font-size:9px">(weight 35%)</span></span>
        <span class="score-value" style="color:var(--c-tool)">${d.tool.toFixed(3)}</span>
      </div>${bar(d.tool,'var(--c-tool)')}
    </div>
    <div class="score-row">
      <div class="score-row-header">
        <span class="score-label">Semantic Reasoning <span style="color:var(--muted);font-size:9px">(weight 40%)</span></span>
        <span class="score-value" style="color:var(--c-sem)">${d.semantic.toFixed(3)}</span>
      </div>${bar(d.semantic,'var(--c-sem)')}
    </div>
    <div class="score-row">
      <div class="score-row-header">
        <span class="score-label">Complexity <span style="color:var(--muted);font-size:9px">(weight 25%)</span></span>
        <span class="score-value" style="color:var(--c-complex)">${d.complexity.toFixed(3)}</span>
      </div>${bar(d.complexity,'var(--c-complex)')}
    </div>
    <div class="confidence-note">
      Confidence: <strong style="color:var(--text)">${d.confidence}</strong> &nbsp;·&nbsp;
      Baseline: <strong style="color:var(--text)">${d.baselineSize}</strong> sessions
    </div>`;
}

// ── What-drifted explanation ─────────────────────────────────────────────────

function renderExplanation(s) {
  const box = document.getElementById('explain-box');
  if (!s.drift || !s.explanation) { box.style.display = 'none'; return; }
  box.style.display = 'block';
  const ex   = s.explanation;
  const ctx  = currentScenarioKey ? SCENARIO_CONTEXT[currentScenarioKey] : null;
  const d    = s.drift;

  const statusIcon = v => v < 0.15 ? '✓ consistent' : v < 0.4 ? '~ slightly changed' : '✗ drifted';
  const statusColor = v => v < 0.15 ? 'var(--c-none)' : v < 0.4 ? 'var(--c-warn)' : 'var(--c-critical)';

  const baseToolList = ex.baselineTopTools?.length
    ? ex.baselineTopTools.join(' → ')
    : 'not yet established';

  const sessionToolList = [...new Set(s.tools)].join(' → ');

  box.innerHTML = `
    <div class="explain-header">
      <span class="explain-title">What Drifted</span>
      <span class="explain-driver">Primary driver: <strong style="color:var(--text)">${ex.primary}</strong></span>
    </div>

    ${ctx?.driftDescription ? `
    <div class="explain-story">
      ${esc(ctx.driftDescription)}
    </div>` : ''}

    <div class="explain-components">
      <div class="explain-component">
        <div class="explain-comp-header">
          <span class="comp-name" style="color:var(--c-tool)">Tool Pattern</span>
          <span class="comp-status" style="color:${statusColor(d.tool)}">${statusIcon(d.tool)}</span>
        </div>
        <div class="comp-detail">${esc(ex.toolDetail)}</div>
        <div class="comp-compare">
          <span class="comp-compare-label">Baseline tools:</span>
          <span class="comp-tools baseline-tools">${esc(baseToolList)}</span>
        </div>
        <div class="comp-compare">
          <span class="comp-compare-label">This session:</span>
          <span class="comp-tools ${d.tool > 0.4 ? 'drift-tools' : 'baseline-tools'}">${esc(sessionToolList)}</span>
        </div>
      </div>

      <div class="explain-component">
        <div class="explain-comp-header">
          <span class="comp-name" style="color:var(--c-sem)">Semantic Reasoning</span>
          <span class="comp-status" style="color:${statusColor(d.semantic)}">${statusIcon(d.semantic)}</span>
        </div>
        <div class="comp-detail">${esc(ex.semDetail)}</div>
      </div>

      <div class="explain-component">
        <div class="explain-comp-header">
          <span class="comp-name" style="color:var(--c-complex)">Complexity</span>
          <span class="comp-status" style="color:${statusColor(d.complexity)}">${statusIcon(d.complexity)}</span>
        </div>
        <div class="comp-detail">${esc(ex.complexDetail)}</div>
      </div>
    </div>

    <div class="explain-recommendation">
      <span class="rec-icon">→</span> ${esc(ex.recommendation)}
    </div>`;
}

// ── Content tabs ────────────────────────────────────────────────────────────

function renderTabContent(s, tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab-${tab}`)?.classList.add('active');

  if (tab === 'reasoning') {
    const el = document.getElementById('tab-reasoning');
    el.innerHTML = s.reasoning.length
      ? s.reasoning.map((r, i) => `
          <div class="checkpoint-item">
            <div class="checkpoint-num">Step ${i+1}</div>
            ${esc(r)}
          </div>`).join('')
      : `<span style="color:var(--muted);font-size:11px">No reasoning checkpoints.</span>`;
  }

  if (tab === 'tools') {
    const el = document.getElementById('tab-tools');
    const baselineSet = new Set(engine.baseline?.vocab.filter((v,i) =>
      (engine.baseline?.toolCentroid[i] || 0) > 0.05
    ) || []);

    if (!s.tools.length) {
      el.innerHTML = `<span style="color:var(--muted);font-size:11px">No tool calls.</span>`;
    } else {
      const chips = s.tools.map((t, i) => {
        const isNew = baselineSet.size > 0 && !baselineSet.has(t);
        return (i > 0 ? `<span class="tool-arrow">→</span>` : '') +
          `<span class="tool-chip${isNew ? ' drift-tool' : ''}">${esc(t)}${isNew ? ' <span class="new-tag">new</span>' : ''}</span>`;
      }).join('');

      const baselineToolStr = baselineSet.size
        ? [...baselineSet].join(' → ')
        : 'not yet established';

      el.innerHTML = `
        ${baselineSet.size > 0 ? `
        <div class="tool-compare-row">
          <span class="tool-compare-label">Baseline pipeline:</span>
          <div class="tool-flow">${[...baselineSet].map(t => `<span class="tool-chip">${esc(t)}</span>`).join('<span class="tool-arrow">→</span>')}</div>
        </div>
        <div class="tool-compare-divider">This session:</div>` : ''}
        <div class="tool-flow">${chips}</div>`;
    }
  }

  if (tab === 'decisions') {
    const el = document.getElementById('tab-decisions');
    el.innerHTML = s.decisions.length
      ? s.decisions.map((d, i) => `
          <div class="decision-item">
            <div>Chose: <span class="decision-chosen">${esc(d.chosen)}</span>
              ${d.reconsidered ? '<span class="decision-reconsidered">⚠ reconsidered</span>' : ''}
            </div>
            <div class="decision-options">Options considered: ${d.options.map(esc).join(', ')}</div>
            <div class="decision-conf">Confidence: ${(d.confidence*100).toFixed(0)}%</div>
          </div>`).join('')
      : `<span style="color:var(--muted);font-size:11px">No decisions logged.</span>`;
  }
}

// ─── Scenarios ────────────────────────────────────────────────────────────────

async function runScenario(name) {
  if (isRunning) return;
  isRunning = true;
  currentScenarioKey = name;
  resetAll(false);

  const steps = SCENARIOS[name];
  const total = steps.reduce((s, x) => s + x.count, 0);
  let done = 0;

  const progress = document.getElementById('scenario-progress');
  const fill      = document.getElementById('progress-fill');
  const plabel    = document.getElementById('progress-label-text');
  progress.classList.add('visible');
  setScenarioBtnsDisabled(true, name);

  for (const { type, count } of steps) {
    for (let i = 0; i < count; i++) {
      await delay(350);
      addSession(type);
      done++;
      fill.style.width = `${(done/total)*100}%`;
      plabel.textContent = `Session ${done}/${total} — ${TYPE_LABEL[type] || type}`;
    }
  }

  progress.classList.remove('visible');
  setScenarioBtnsDisabled(false, null);
  document.getElementById('btn-add').disabled = false;
  isRunning = false;
}

function setScenarioBtnsDisabled(disabled, activeKey) {
  document.querySelectorAll('.scenario-btn').forEach(b => {
    b.disabled = disabled;
    b.classList.toggle('active', b.dataset.scenario === activeKey);
  });
  document.getElementById('btn-add').disabled = disabled;
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Session add ──────────────────────────────────────────────────────────────

function addSession(type) {
  const factory = SESSION_FACTORIES[type];
  if (!factory) return;
  const session = engine.addSession(factory());
  updateChart();
  renderLog();
  updateHeader();
  selectSession(session);
}

// ─── Reset ────────────────────────────────────────────────────────────────────

function resetAll(render = true) {
  engine.reset();
  resetCounter();
  selectedSession   = null;
  currentScenarioKey = null;
  if (chart) {
    chart.data.labels = [];
    chart.data.datasets.forEach(d => { d.data = []; });
    chart.update();
  }
  if (render) {
    renderLog();
    updateHeader();
    document.getElementById('detail-empty').style.display   = '';
    document.getElementById('detail-content').style.display = 'none';
    document.getElementById('explain-box').style.display    = 'none';
  }
}

// ─── Scenario buttons ─────────────────────────────────────────────────────────

const SCENARIO_META = [
  { id:'no-drift',         label:'No Drift',            dot:'none',     badge:'12',  hint:'Flat baseline — all sessions consistent' },
  { id:'tool-drift',       label:'Tool Pattern Drift',  dot:'orange',   badge:'6+6', hint:'Agent switches data source mid-stream' },
  { id:'semantic-drift',   label:'Semantic Drift',      dot:'warn',     badge:'6+6', hint:'Agent becomes uncertain and hedging' },
  { id:'complexity-drift', label:'Complexity Drift',    dot:'pink',     badge:'6+6', hint:'Agent develops decision paralysis' },
  { id:'full-drift',       label:'Full Drift',          dot:'critical', badge:'5+7', hint:'Everything changes at once' },
];

function initScenarioButtons() {
  const c = document.getElementById('scenario-buttons');
  c.innerHTML = SCENARIO_META.map(m => `
    <button class="scenario-btn" data-scenario="${m.id}" title="${esc(m.hint)}">
      <span class="dot ${m.dot}"></span>
      <span class="scenario-label">${esc(m.label)}</span>
      <span class="badge">${m.badge}</span>
    </button>`).join('');
  c.querySelectorAll('.scenario-btn').forEach(btn =>
    btn.addEventListener('click', () => runScenario(btn.dataset.scenario))
  );
}

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initChart();
  initScenarioButtons();
  updateHeader();
  renderLog();

  document.getElementById('btn-add').addEventListener('click', () => {
    const type = document.getElementById('session-type').value;
    currentScenarioKey = (() => {
      if (type === 'normal')            return 'no-drift';
      if (type === 'tool-drift')        return 'tool-drift';
      if (type === 'semantic-drift')    return 'semantic-drift';
      if (type === 'complexity-drift')  return 'complexity-drift';
      if (type === 'full-drift')        return 'full-drift';
      return null;
    })();
    addSession(type);
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    resetAll(true);
    initScenarioButtons();
  });

  document.getElementById('content-tabs').addEventListener('click', e => {
    const btn = e.target.closest('.tab-btn');
    if (!btn || !selectedSession) return;
    renderTabContent(selectedSession, btn.dataset.tab);
  });
});
