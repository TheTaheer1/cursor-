import { useMemo } from "react";

const ZONE_COLORS = {
  zone_c: { bg: "#FCEBEB", border: "#A32D2D", dot: "#E24B4A", label: "#791F1F" },
  zone_b: { bg: "#FAEEDA", border: "#BA7517", dot: "#EF9F27", label: "#633806" },
  green:  { bg: "#EAF3DE", border: "#3B6D11", dot: "#639922", label: "#27500A" },
};

export default function CalibrationMap({ nodes = [] }) {
  const sorted = useMemo(() => {
    const safe = Array.isArray(nodes) ? nodes : [];
    return [...safe].sort((a, b) => {
      const order = { zone_c: 0, zone_b: 1, green: 2 };
      return (order[a?.zone] ?? 3) - (order[b?.zone] ?? 3);
    });
  }, [nodes]);

  const counts = useMemo(() => {
    const safe = Array.isArray(nodes) ? nodes : [];
    return {
      zone_c: safe.filter((n) => n?.zone === "zone_c").length,
      zone_b: safe.filter((n) => n?.zone === "zone_b").length,
      green:  safe.filter((n) => n?.zone === "green").length,
    };
  }, [nodes]);

  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: "10px", overflow: "hidden", background: "#fff" }}>
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #e5e7eb",
          display: "flex",
          gap: "12px",
          alignItems: "center",
        }}
      >
        <span style={{ fontSize: "14px", fontWeight: 600 }}>Calibration map</span>
        <span style={{ fontSize: "12px", color: "#E24B4A", marginLeft: "auto" }}>Zone C: {counts.zone_c}</span>
        <span style={{ fontSize: "12px", color: "#BA7517" }}>Zone B: {counts.zone_b}</span>
        <span style={{ fontSize: "12px", color: "#3B6D11" }}>Green: {counts.green}</span>
      </div>
      <div
        style={{
          padding: "12px",
          display: "flex",
          flexWrap: "wrap",
          gap: "8px",
          maxHeight: "420px",
          overflowY: "auto",
        }}
      >
        {sorted.length === 0 && (
          <p style={{ fontSize: "13px", color: "#999", padding: "20px" }}>
            No nodes yet. Run a step to begin.
          </p>
        )}
        {sorted.map((node) => {
          const colors = ZONE_COLORS[node?.zone] || ZONE_COLORS.green;
          const conf = node?.confidence_avg || 0;
          const visits = node?.visit_count || 0;
          return (
            <div
              key={node.key}
              title={`${node.key}\nVisits: ${visits}\nAvg confidence: ${conf.toFixed(2)}`}
              style={{
                background: colors.bg,
                border: `1px solid ${colors.border}`,
                borderRadius: "8px",
                padding: "8px 10px",
                minWidth: "120px",
                cursor: "default",
                transition: "transform 0.2s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.03)")}
              onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                <div
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    background: colors.dot,
                    flexShrink: 0,
                  }}
                />
                <span style={{ fontSize: "11px", fontWeight: 600, color: colors.label }}>{node.topic}</span>
              </div>
              <div style={{ fontSize: "10px", color: colors.label, opacity: 0.8 }}>{node.question_type}</div>
              <div style={{ fontSize: "10px", color: colors.label, opacity: 0.7 }}>{node.difficulty_tier}</div>
              {visits > 0 && (
                <div style={{ fontSize: "10px", color: colors.label, marginTop: "4px", opacity: 0.6 }}>
                  {visits} visit{visits !== 1 ? "s" : ""}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
