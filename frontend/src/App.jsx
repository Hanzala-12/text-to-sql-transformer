import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

function useBackendStatus() {
  const [status, setStatus] = useState({ loading: true, ready: false, error: null });
  const pollRef = useRef(null);

  const check = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      const data = await res.json();
      setStatus({ loading: false, ...data });
      return data.ready;
    } catch {
      setStatus({ loading: false, ready: false, error: "Cannot reach the backend at " + API_BASE });
      return false;
    }
  };

  useEffect(() => {
    check();
    pollRef.current = setInterval(async () => {
      const ready = await check();
      if (ready) clearInterval(pollRef.current);
    }, 4000);
    return () => clearInterval(pollRef.current);
  }, []);

  return status;
}

function StatusBadge({ status }) {
  if (status.loading) {
    return <span className="badge badge-pending">checking backend…</span>;
  }
  if (status.ready) {
    return (
      <span className="badge badge-ready">
        model ready · epoch {status.checkpoint_epoch} · dev loss{" "}
        {Number(status.checkpoint_dev_loss).toFixed(3)}
      </span>
    );
  }
  return <span className="badge badge-error">backend not ready</span>;
}

function ColumnChips({ columns }) {
  if (columns.length === 0) return null;
  return (
    <div className="chip-row">
      {columns.map((c, i) => (
        <span className="chip" key={i}>
          <span className="chip-index">c{i}</span>
          {c}
        </span>
      ))}
    </div>
  );
}

export default function App() {
  const status = useBackendStatus();

  const [question, setQuestion] = useState("What is Terrence Ross' nationality?");
  const [columnsRaw, setColumnsRaw] = useState(
    "Player, No., Nationality, Position, Years in Toronto, School/Club Team"
  );
  const [method, setMethod] = useState("beam");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [requestError, setRequestError] = useState(null);

  const columns = columnsRaw
    .split(",")
    .map((c) => c.trim())
    .filter(Boolean);

  const canSubmit = status.ready && question.trim() && columns.length > 0 && !loading;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setRequestError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, columns, method }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed.");
      setResult(data);
    } catch (err) {
      setRequestError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Text-to-SQL</h1>
          <p className="subtitle">
            Encoder-decoder Transformer, trained from scratch on WikiSQL
          </p>
        </div>
        <StatusBadge status={status} />
      </header>

      {!status.ready && !status.loading && (
        <div className="notice notice-error">{status.error}</div>
      )}

      <main className="layout">
        <form className="card" onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What is Terrence Ross' nationality?"
            />
          </div>

          <div className="field">
            <label htmlFor="columns">Table columns (comma-separated)</label>
            <input
              id="columns"
              type="text"
              value={columnsRaw}
              onChange={(e) => setColumnsRaw(e.target.value)}
              placeholder="Player, No., Nationality, Position"
            />
            <ColumnChips columns={columns} />
          </div>

          <div className="field">
            <span className="field-label">Decoding</span>
            <div className="segmented">
              {["beam", "greedy"].map((m) => (
                <button
                  type="button"
                  key={m}
                  className={"segmented-option" + (method === m ? " active" : "")}
                  onClick={() => setMethod(m)}
                >
                  {m === "beam" ? "Beam search (4)" : "Greedy"}
                </button>
              ))}
            </div>
          </div>

          <button type="submit" className="submit-btn" disabled={!canSubmit}>
            {loading ? "Generating…" : "Generate SQL"}
          </button>
        </form>

        <section className="card result-card">
          <h2>Result</h2>

          {!result && !requestError && !loading && (
            <p className="placeholder">Run a query to see the generated SQL here.</p>
          )}

          {loading && <p className="placeholder">Decoding…</p>}

          {requestError && <div className="notice notice-error">{requestError}</div>}

          {result && result.ok && (
            <>
              <pre className="sql-block">{result.sql}</pre>
              <details className="raw-details">
                <summary>Raw model output</summary>
                <code className="raw-code">{result.tokenized}</code>
              </details>
            </>
          )}

          {result && !result.ok && (
            <div className="notice notice-warn">
              <strong>Could not parse a valid query.</strong>
              <p>{result.error}</p>
              <details className="raw-details">
                <summary>Raw model output</summary>
                <code className="raw-code">{result.tokenized}</code>
              </details>
            </div>
          )}
        </section>
      </main>

      <footer className="page-footer">
        Generative AI - Assignment 02 · Text-to-SQL Transformer
      </footer>
    </div>
  );
}
