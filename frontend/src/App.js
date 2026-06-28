import { useState } from "react";
import "./App.css";

// ── Constants ─────────────────────────────────────────────────────────────────

const PRIORITY_CONFIG = {
  Critical: { color: "var(--critical)", bg: "var(--critical-bg)", dot: "🔴" },
  High:     { color: "var(--high)",     bg: "var(--high-bg)",     dot: "🟠" },
  Medium:   { color: "var(--medium)",   bg: "var(--medium-bg)",   dot: "🟡" },
  Low:      { color: "var(--low)",      bg: "var(--low-bg)",      dot: "🟢" },
};

const CATEGORY_ICONS = {
  "Billing":         "💳",
  "Technical Issue": "🔧",
  "Account Access":  "🔑",
  "Feature Request": "💡",
  "Refund":          "↩️",
  "Onboarding":      "🚀",
  "Performance":     "⚡",
  "Data Privacy":    "🔒",
};

const DEMO_TICKETS = [
  { id: "TKT-001", text: "My card was charged twice this month and I need a refund urgently." },
  { id: "TKT-002", text: "The app keeps crashing every time I try to export a report." },
  { id: "TKT-003", text: "I forgot my password and the reset link isn't working at all." },
  { id: "TKT-004", text: "Would love to see a dark mode option added to the interface." },
  { id: "TKT-005", text: "I need all my personal data deleted immediately per GDPR right." },
  { id: "TKT-006", text: "Platform has been incredibly slow today — 3 minute load times." },
  { id: "TKT-007", text: "Just signed up but I'm not sure how to add my team members." },
  { id: "TKT-008", text: "I want a refund for the annual plan I just purchased yesterday." },
];

// ── Priority Badge ────────────────────────────────────────────────────────────

function PriorityBadge({ priority }) {
  const cfg = PRIORITY_CONFIG[priority] || {};
  return (
    <span className="priority-badge" style={{ color: cfg.color, background: cfg.bg }}>
      {priority}
    </span>
  );
}

// ── Confidence Bar ────────────────────────────────────────────────────────────

function ConfBar({ score, max }) {
  const pct = Math.round(score * 100);
  const barPct = max > 0 ? (score / max) * 100 : pct;
  return (
    <div className="conf-bar-row">
      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ width: `${barPct}%` }} />
      </div>
      <span className="conf-bar-pct">{pct}%</span>
    </div>
  );
}

// ── Single Classify Result ────────────────────────────────────────────────────

function ClassifyResult({ result }) {
  const [expanded, setExpanded] = useState(false);
  const maxScore = result.all_scores[0]?.score || 1;

  return (
    <div className="result-card">
      <div className="result-top">
        <div className="result-category">
          <span className="cat-icon">{CATEGORY_ICONS[result.label] || "📋"}</span>
          <div>
            <div className="cat-label-sm">Category</div>
            <div className="cat-name">{result.label}</div>
          </div>
        </div>
        <div className="result-meta">
          <PriorityBadge priority={result.priority} />
          <span className="conf-pill">{Math.round(result.confidence * 100)}% confidence</span>
        </div>
      </div>

      <button className="breakdown-toggle" onClick={() => setExpanded(!expanded)}>
        {expanded ? "Hide" : "Show"} all category scores
        <span className="toggle-arrow">{expanded ? "▲" : "▼"}</span>
      </button>

      {expanded && (
        <div className="breakdown">
          {result.all_scores.map((s) => (
            <div key={s.category} className={`breakdown-row ${s.category === result.category ? "breakdown-row--top" : ""}`}>
              <span className="breakdown-icon">{CATEGORY_ICONS[s.label] || "📋"}</span>
              <span className="breakdown-name">{s.label}</span>
              <ConfBar score={s.score} max={maxScore} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Batch Queue Row ───────────────────────────────────────────────────────────

function QueueRow({ ticket, index }) {
  const delay = `${index * 60}ms`;
  return (
    <div className="queue-row" style={{ animationDelay: delay }}>
      <span className="queue-id">{ticket.id}</span>
      <span className="queue-text">{ticket.text.slice(0, 70)}{ticket.text.length > 70 ? "…" : ""}</span>
      <span className="queue-cat">
        {CATEGORY_ICONS[ticket.label] || "📋"} {ticket.label}
      </span>
      <PriorityBadge priority={ticket.priority} />
      <span className="queue-conf">{Math.round(ticket.confidence * 100)}%</span>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState("single"); // "single" | "batch"

  // Single
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Batch
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  async function handleClassify() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Server error");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleBatchDemo() {
    setBatchLoading(true);
    setBatchResults(null);
    try {
      const res = await fetch("/classify/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickets: DEMO_TICKETS }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Server error");
      setBatchResults(data.results);
    } catch (e) {
      setBatchResults([]);
    } finally {
      setBatchLoading(false);
    }
  }

  const priorityCounts = batchResults
    ? ["Critical", "High", "Medium", "Low"].map((p) => ({
        priority: p,
        count: batchResults.filter((r) => r.priority === p).length,
      }))
    : [];

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-mark">◈</span>
            <span className="logo-text">TicketSight</span>
            <span className="logo-tag">NLP Classifier</span>
          </div>
          <nav className="header-links">
            <a href="https://github.com/shrutimishra816/nlp-ticket-classifier" target="_blank" rel="noreferrer">GitHub</a>
          </nav>
        </div>
      </header>

      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-eyebrow">TF-IDF · Logistic Regression · 8 Categories</div>
          <h1 className="hero-title">Route tickets instantly.<br /><em>No analyst required.</em></h1>
          <p className="hero-sub">
            89% accuracy across 10,000+ support tickets. Every ticket gets a category,
            a priority level, and a confidence score — in milliseconds.
          </p>
          <div className="hero-stats">
            <div className="stat"><span className="stat-val">89%</span><span className="stat-lbl">Accuracy</span></div>
            <div className="stat-div" />
            <div className="stat"><span className="stat-val">8</span><span className="stat-lbl">Categories</span></div>
            <div className="stat-div" />
            <div className="stat"><span className="stat-val">60%</span><span className="stat-lbl">Faster Triage</span></div>
            <div className="stat-div" />
            <div className="stat"><span className="stat-val">$30K</span><span className="stat-lbl">Annual Saving</span></div>
          </div>
        </div>
      </section>

      {/* ── Tabs ── */}
      <div className="tabs-bar">
        <div className="tabs-inner">
          <button className={`tab ${tab === "single" ? "tab--active" : ""}`} onClick={() => setTab("single")}>
            Single Ticket
          </button>
          <button className={`tab ${tab === "batch" ? "tab--active" : ""}`} onClick={() => setTab("batch")}>
            Batch Queue
          </button>
        </div>
      </div>

      {/* ── Single ── */}
      {tab === "single" && (
        <section className="section">
          <div className="single-inner">

            {/* Input */}
            <div className="card">
              <div className="card-header">
                <span className="card-step">01</span>
                <h2>Paste Ticket Text</h2>
              </div>
              <textarea
                className="ticket-input"
                rows={5}
                placeholder="e.g. My card was charged twice this month and I need this fixed urgently…"
                value={text}
                onChange={(e) => { setText(e.target.value); setResult(null); setError(null); }}
              />
              <div className="input-actions">
                <button
                  className="btn-primary"
                  onClick={handleClassify}
                  disabled={loading || !text.trim()}
                >
                  {loading ? (
                    <><span className="spinner" /> Classifying…</>
                  ) : "Classify Ticket"}
                </button>
                <button className="btn-ghost" onClick={() => {
                  setText(DEMO_TICKETS[Math.floor(Math.random() * DEMO_TICKETS.length)].text);
                  setResult(null);
                }}>
                  Load Example
                </button>
                {text && (
                  <button className="btn-ghost" onClick={() => { setText(""); setResult(null); setError(null); }}>
                    Clear
                  </button>
                )}
              </div>
            </div>

            {/* Result */}
            <div className="card">
              <div className="card-header">
                <span className="card-step">02</span>
                <h2>Classification</h2>
              </div>

              {!result && !error && !loading && (
                <div className="empty-state">
                  <div className="empty-icon">◈</div>
                  <p>Paste a support ticket and click <strong>Classify Ticket</strong> to see the category, priority, and confidence breakdown.</p>
                </div>
              )}

              {loading && (
                <div className="empty-state">
                  <div className="loading-dots">
                    <span /><span /><span />
                  </div>
                  <p>Classifying…</p>
                </div>
              )}

              {error && (
                <div className="error-state">
                  <span>⚠</span> {error}
                </div>
              )}

              {result && <ClassifyResult result={result} />}
            </div>

          </div>
        </section>
      )}

      {/* ── Batch ── */}
      {tab === "batch" && (
        <section className="section">
          <div className="batch-inner">

            <div className="card">
              <div className="card-header">
                <span className="card-step">01</span>
                <h2>Ops Dashboard — Batch Triage</h2>
                <button
                  className="btn-primary btn-sm"
                  onClick={handleBatchDemo}
                  disabled={batchLoading}
                  style={{ marginLeft: "auto" }}
                >
                  {batchLoading ? <><span className="spinner" /> Running…</> : "Run Demo Batch"}
                </button>
              </div>

              {!batchResults && !batchLoading && (
                <div className="empty-state">
                  <div className="empty-icon">📋</div>
                  <p>Click <strong>Run Demo Batch</strong> to classify 8 sample tickets simultaneously and see the priority queue.</p>
                </div>
              )}

              {batchLoading && (
                <div className="empty-state">
                  <div className="loading-dots"><span /><span /><span /></div>
                  <p>Classifying batch…</p>
                </div>
              )}

              {batchResults && batchResults.length > 0 && (
                <>
                  {/* Priority summary */}
                  <div className="priority-summary">
                    {priorityCounts.map(({ priority, count }) => (
                      <div key={priority} className="priority-tile" style={{
                        borderTop: `3px solid ${PRIORITY_CONFIG[priority].color}`,
                        background: PRIORITY_CONFIG[priority].bg,
                      }}>
                        <span className="priority-tile-count" style={{ color: PRIORITY_CONFIG[priority].color }}>{count}</span>
                        <span className="priority-tile-label">{priority}</span>
                      </div>
                    ))}
                  </div>

                  {/* Queue header */}
                  <div className="queue-header">
                    <span>ID</span>
                    <span>Ticket</span>
                    <span>Category</span>
                    <span>Priority</span>
                    <span>Confidence</span>
                  </div>

                  {/* Queue rows sorted by priority */}
                  {["Critical", "High", "Medium", "Low"].flatMap((p) =>
                    batchResults
                      .filter((r) => r.priority === p)
                      .map((r, i) => <QueueRow key={r.id} ticket={r} index={i} />)
                  )}
                </>
              )}
            </div>
          </div>
        </section>
      )}

      {/* ── How It Works ── */}
      <section className="how-section">
        <div className="how-inner">
          <h2 className="how-title">How it works</h2>
          <div className="pipeline">
            {[
              { n: "1", label: "Raw Ticket", desc: "Free-text from CRM or email" },
              { n: "2", label: "TF-IDF", desc: "Bigrams, 15k features, sublinear TF" },
              { n: "3", label: "Logistic Regression", desc: "C=5, lbfgs, 89% accuracy" },
              { n: "4", label: "Priority Tag", desc: "Critical / High / Medium / Low" },
              { n: "5", label: "Airtable Export", desc: "CSV ready for ops team" },
            ].map((s, i, arr) => (
              <div key={s.n} className="pipeline-item">
                <div className="pipe-step">
                  <div className="pipe-num">{s.n}</div>
                  <div>
                    <div className="pipe-label">{s.label}</div>
                    <div className="pipe-desc">{s.desc}</div>
                  </div>
                </div>
                {i < arr.length - 1 && <div className="pipe-arrow">→</div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="footer">
        <div className="footer-inner">
          <span>TicketSight · NLP Support Classifier</span>
          <span>TF-IDF · Logistic Regression · Flask · React</span>
        </div>
      </footer>

    </div>
  );
}
