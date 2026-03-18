import React, { useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

// ── tiny helpers ──────────────────────────────────────────────────────────────
const B64Img = ({ b64, alt }) =>
  b64 ? (
    <img
      src={`data:image/png;base64,${b64}`}
      alt={alt}
      style={{ maxWidth: "100%", borderRadius: 8, margin: "8px 0" }}
    />
  ) : null;

const Card = ({ title, children }) => (
  <div className="card">
    <h3 className="card-title">{title}</h3>
    {children}
  </div>
);

const MetricBadge = ({ label, value }) => (
  <span className="badge">
    <strong>{label}:</strong> {value}
  </span>
);

// ── tab components ────────────────────────────────────────────────────────────
function UploadTab({ onUpload }) {
  const [file, setFile] = useState(null);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail);
      const data = await res.json();
      setInfo(data);
      onUpload(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="📤 Upload Dataset">
      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
        className="file-input"
      />
      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className="btn btn-primary"
      >
        {loading ? "Uploading…" : "Upload CSV"}
      </button>
      {error && <p className="error">{error}</p>}
      {info && (
        <div className="info-box">
          <p>
            ✅ <strong>{info.filename}</strong> — {info.rows} rows × {info.columns} columns
          </p>
          <p className="col-list">Columns: {info.column_names.join(", ")}</p>
        </div>
      )}
    </Card>
  );
}

function EDATab() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/eda`);
      if (!res.ok) throw new Error((await res.json()).detail);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="🧪 Exploratory Data Analysis">
      <button onClick={run} disabled={loading} className="btn btn-primary">
        {loading ? "Analysing…" : "Run EDA"}
      </button>
      <a
        href={`${API}/eda/report`}
        target="_blank"
        rel="noreferrer"
        className="btn btn-secondary"
        style={{ marginLeft: 8 }}
      >
        📄 Download PDF Report
      </a>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <B64Img b64={result.missing_plot} alt="Missing Values" />
          <B64Img b64={result.correlation_plot} alt="Correlation Heatmap" />
          <div className="dist-grid">
            {Object.entries(result.distribution_plots || {}).map(([col, b64]) => (
              <div key={col}>
                <p className="dist-label">{col}</p>
                <B64Img b64={b64} alt={col} />
              </div>
            ))}
          </div>
        </>
      )}
    </Card>
  );
}

function MLTab() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/ml`);
      if (!res.ok) throw new Error((await res.json()).detail);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="🤖 Churn Prediction (ML)">
      <button onClick={run} disabled={loading} className="btn btn-primary">
        {loading ? "Training models…" : "Run ML Prediction"}
      </button>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <p className="best-model">
            🏆 Best Model: <strong>{result.best_model}</strong>
          </p>
          <table className="metrics-table">
            <thead>
              <tr>
                {["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"].map(
                  (h) => (
                    <th key={h}>{h}</th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {result.metrics.map((row) => (
                <tr key={row.Model}>
                  {["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"].map(
                    (k) => (
                      <td key={k}>{row[k]}</td>
                    )
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          <B64Img b64={result.comparison_plot} alt="Model Comparison" />
          <B64Img b64={result.roc_plot} alt="ROC Curves" />
        </>
      )}
    </Card>
  );
}

function SurvivalTab() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/survival`);
      if (!res.ok) throw new Error((await res.json()).detail);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="⏳ Survival Analysis">
      <button onClick={run} disabled={loading} className="btn btn-primary">
        {loading ? "Running…" : "Run Survival Analysis"}
      </button>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <p>
            Duration col: <code>{result.duration_col}</code> | Event col:{" "}
            <code>{result.event_col}</code>
          </p>
          <B64Img b64={result.km_plot} alt="Kaplan-Meier" />
          <B64Img b64={result.cox_plot} alt="Cox PH Hazard Ratios" />
          <h4>High-Risk Customers (Top 10)</h4>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Customer</th>
                <th>Churn Risk</th>
                <th>Survival Score</th>
              </tr>
            </thead>
            <tbody>
              {(result.high_risk_customers || []).map((r, i) => (
                <tr key={i}>
                  <td>{r.customer_id ?? i}</td>
                  <td>{r.churn_risk}</td>
                  <td>{r.survival_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Card>
  );
}

function CausalTab() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/causal`);
      if (!res.ok) throw new Error((await res.json()).detail);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="🔎 Causal Inference">
      <button onClick={run} disabled={loading} className="btn btn-primary">
        {loading ? "Running…" : "Run Causal Inference"}
      </button>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <div className="info-box">
            <MetricBadge label="Treatment" value={result.treatment} />
            <MetricBadge label="Outcome" value={result.outcome} />
            <MetricBadge
              label="Causal Estimate"
              value={
                result.causal_estimate !== null
                  ? result.causal_estimate.toFixed(4)
                  : "N/A"
              }
            />
            <MetricBadge label="Method" value={result.method} />
          </div>
          <B64Img b64={result.graph_plot} alt="Causal Graph" />
        </>
      )}
    </Card>
  );
}

function RLTab() {
  const [logs, setLogs] = useState([]);
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState(null);

  const [age, setAge] = useState(35);
  const [fee, setFee] = useState(80);
  const [activity, setActivity] = useState(50);
  const [recommendation, setRecommendation] = useState(null);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState(null);

  const trainAgent = async () => {
    setTraining(true);
    setTrainError(null);
    setLogs([]);
    try {
      const res = await fetch(`${API}/rl/train`, { method: "POST" });
      if (!res.ok) throw new Error((await res.json()).detail);
      const data = await res.json();
      setLogs(data.logs);
    } catch (e) {
      setTrainError(e.message);
    } finally {
      setTraining(false);
    }
  };

  const getRecommendation = async () => {
    setRecLoading(true);
    setRecError(null);
    setRecommendation(null);
    try {
      const res = await fetch(`${API}/rl/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ age: Number(age), fee: Number(fee), activity: Number(activity) }),
      });
      if (!res.ok) throw new Error((await res.json()).detail);
      const data = await res.json();
      setRecommendation(data.recommendation);
    } catch (e) {
      setRecError(e.message);
    } finally {
      setRecLoading(false);
    }
  };

  return (
    <Card title="🎮 Action Recommendation (RL)">
      <h4>Step 1 — Train the Agent</h4>
      <button onClick={trainAgent} disabled={training} className="btn btn-primary">
        {training ? "Training…" : "Train RL Agent (300 episodes)"}
      </button>
      {trainError && <p className="error">{trainError}</p>}
      {logs.length > 0 && (
        <div className="log-box">
          {logs.map((l, i) => (
            <p key={i} className="log-line">
              {l}
            </p>
          ))}
        </div>
      )}

      <h4 style={{ marginTop: 24 }}>Step 2 — Get a Recommendation</h4>
      <div className="form-grid">
        <label>
          Age
          <input
            type="number"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            className="form-input"
          />
        </label>
        <label>
          Monthly Fee ($)
          <input
            type="number"
            value={fee}
            onChange={(e) => setFee(e.target.value)}
            className="form-input"
          />
        </label>
        <label>
          Activity Score (0-100)
          <input
            type="number"
            value={activity}
            onChange={(e) => setActivity(e.target.value)}
            className="form-input"
          />
        </label>
      </div>
      <button
        onClick={getRecommendation}
        disabled={recLoading}
        className="btn btn-accent"
      >
        {recLoading ? "Getting recommendation…" : "Get Action Recommendation"}
      </button>
      {recError && <p className="error">{recError}</p>}
      {recommendation && (
        <div className="recommendation-box">
          🎯 Recommended Action: <strong>{recommendation}</strong>
        </div>
      )}
    </Card>
  );
}

// ── main App ──────────────────────────────────────────────────────────────────
const TABS = ["Upload", "EDA", "ML Prediction", "Survival", "Causal", "RL Agent"];

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [datasetInfo, setDatasetInfo] = useState(null);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🧠 ChurnSense AI</h1>
        <p className="tagline">Upload. Understand. Act.</p>
        {datasetInfo && (
          <p className="dataset-pill">
            📊 {datasetInfo.filename} — {datasetInfo.rows} rows × {datasetInfo.columns} cols
          </p>
        )}
      </header>

      <nav className="tab-nav">
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setActiveTab(i)}
            className={`tab-btn ${activeTab === i ? "active" : ""}`}
          >
            {t}
          </button>
        ))}
      </nav>

      <main className="main-content">
        {activeTab === 0 && <UploadTab onUpload={setDatasetInfo} />}
        {activeTab === 1 && <EDATab />}
        {activeTab === 2 && <MLTab />}
        {activeTab === 3 && <SurvivalTab />}
        {activeTab === 4 && <CausalTab />}
        {activeTab === 5 && <RLTab />}
      </main>
    </div>
  );
}
