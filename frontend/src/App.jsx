import { useState, useEffect } from "react";
import { connect, disconnect } from "./api/socket";
import CalibrationMap from "./components/CalibrationMap";
import RewardCurve from "./components/RewardCurve";
import FailureLog from "./components/FailureLog";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [nodes, setNodes] = useState([]);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [failures, setFailures] = useState([]);
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [zoneCounts, setZoneCounts] = useState({ zone_c: 0, zone_b: 0, green: 0 });

  useEffect(() => {
    fetch(`${API_BASE}/api/state`)
      .then((r) => r.json())
      .then((data) => {
        const cm = data?.calibration_map || {};
        setNodes(cm.nodes || []);
        setRewardHistory(data?.reward_history || []);
        setStep(data?.step || 0);
        setZoneCounts({
          zone_c: cm.zone_c_count || 0,
          zone_b: cm.zone_b_count || 0,
          green: cm.green_count || 0,
        });
      })
      .catch(() => {});

    fetch(`${API_BASE}/api/failures`)
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setFailures(data);
      })
      .catch(() => {});

    connect((data) => {
      if (!data) return;
      if (data.calibration_map?.nodes) setNodes(data.calibration_map.nodes);
      if (data.reward !== undefined && data.step !== undefined) {
        setRewardHistory((prev) => [...prev, { step: data.step, reward: data.reward }]);
      }
      if (data.failure) setFailures((prev) => [data.failure, ...prev].slice(0, 20));
      if (data.step !== undefined) setStep(data.step);
      if (data.zone_counts) setZoneCounts(data.zone_counts);
    });

    return () => disconnect();
  }, []);

  const handleRun = async () => {
    setRunning(true);
    try {
      await fetch(`${API_BASE}/api/run-steps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n: 1 }),
      });
    } finally {
      setRunning(false);
    }
  };

  const handleReset = async () => {
    try {
      await fetch(`${API_BASE}/api/reset`, { method: "POST" });
    } catch (e) {
      /* noop */
    }
    setRewardHistory([]);
    setFailures([]);
    setStep(0);
    setZoneCounts({ zone_c: 0, zone_b: 0, green: 0 });
  };

  return (
    <div
      style={{
        fontFamily: "system-ui, sans-serif",
        padding: "16px",
        maxWidth: "1200px",
        margin: "0 auto",
      }}
    >
      <div style={{ marginBottom: "16px", display: "flex", alignItems: "center", gap: "12px" }}>
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: 600, margin: 0 }}>EvoAI Lab</h1>
          <p style={{ fontSize: "13px", color: "#666", margin: "4px 0 0 0" }}>
            Step {step} &middot; Zone C: {zoneCounts.zone_c} &middot; Zone B: {zoneCounts.zone_b} &middot; Green: {zoneCounts.green}
          </p>
          <p style={{ fontSize: "12px", color: "#999", margin: "4px 0 0 0", fontStyle: "italic" }}>
            Most AI training asks if the model is right or wrong. We train for something harder: is it right about being right?
          </p>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: "8px" }}>
          <button
            onClick={handleRun}
            disabled={running}
            style={{
              padding: "8px 16px",
              fontSize: "13px",
              cursor: running ? "not-allowed" : "pointer",
              background: running ? "#ccc" : "#1D9E75",
              color: "#fff",
              border: "none",
              borderRadius: "6px",
            }}
          >
            {running ? "Running..." : "Run step"}
          </button>
          <button
            onClick={handleReset}
            style={{
              padding: "8px 16px",
              fontSize: "13px",
              cursor: "pointer",
              background: "transparent",
              border: "1px solid #ccc",
              borderRadius: "6px",
            }}
          >
            Reset
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
        <div style={{ gridRow: "span 2" }}>
          <CalibrationMap nodes={nodes} />
        </div>
        <div>
          <RewardCurve history={rewardHistory} />
        </div>
        <div>
          <FailureLog failures={failures} />
        </div>
      </div>
    </div>
  );
}
