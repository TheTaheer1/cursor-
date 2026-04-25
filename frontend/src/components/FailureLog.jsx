export default function FailureLog({ failures = [] }) {
  const safe = Array.isArray(failures) ? failures : [];
  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: "10px", overflow: "hidden", background: "#fff" }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid #e5e7eb" }}>
        <span style={{ fontSize: "14px", fontWeight: 600 }}>What the model learned</span>
        <span style={{ float: "right", fontSize: "12px", color: "#999" }}>{safe.length} moments</span>
      </div>
      <div style={{ maxHeight: "200px", overflowY: "auto", padding: "8px" }}>
        {safe.length === 0 && (
          <p style={{ fontSize: "13px", color: "#999", padding: "12px" }}>
            No failures yet. Run steps to see learning moments.
          </p>
        )}
        {safe.map((f, i) => {
          const failure = f?.failure_answer || "";
          const correction = f?.correction || "";
          return (
            <div
              key={i}
              style={{
                borderLeft: "3px solid #1D9E75",
                padding: "8px 10px",
                marginBottom: "8px",
                background: "#f9fafb",
                borderRadius: "0 6px 6px 0",
              }}
            >
              <div
                style={{
                  fontSize: "11px",
                  fontWeight: 600,
                  color: "#666",
                  marginBottom: "4px",
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                }}
              >
                {(f?.topic || "general")} &middot; {(f?.question_type || "?")} &middot; {(f?.difficulty_tier || "?")}
              </div>
              <div style={{ fontSize: "12px", color: "#A32D2D", marginBottom: "4px" }}>
                Believed: <em>{failure.slice(0, 120)}{failure.length > 120 ? "..." : ""}</em>
              </div>
              <div style={{ fontSize: "12px", color: "#3B6D11" }}>
                Learned: {correction.slice(0, 160)}{correction.length > 160 ? "..." : ""}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
