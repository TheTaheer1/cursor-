import { useMemo } from "react";
import {
  Bar,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

export default function RewardCurve({ history = [] }) {
  const data = useMemo(() => {
    const safe = Array.isArray(history) ? history : [];
    return safe.map((h, i) => ({
      step: h?.step ?? i,
      reward: parseFloat(((h?.reward ?? 0)).toFixed(3)),
    }));
  }, [history]);

  const mean = useMemo(() => {
    if (!data.length) return 0;
    return parseFloat((data.reduce((s, d) => s + d.reward, 0) / data.length).toFixed(3));
  }, [data]);

  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: "10px", overflow: "hidden", background: "#fff" }}>
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #e5e7eb",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <span style={{ fontSize: "14px", fontWeight: 600 }}>Reward curve</span>
        <span
          style={{
            marginLeft: "auto",
            fontSize: "12px",
            color: mean >= 0 ? "#3B6D11" : "#A32D2D",
          }}
        >
          Mean: {mean}
        </span>
      </div>
      <div style={{ padding: "12px" }}>
        {data.length === 0 ? (
          <p style={{ fontSize: "13px", color: "#999", padding: "20px 0" }}>No reward data yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={180}>
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="step"
                tick={{ fontSize: 10 }}
                label={{ value: "step", position: "insideBottomRight", offset: 0, fontSize: 10 }}
              />
              <YAxis tick={{ fontSize: 10 }} domain={[-1, 1]} />
              <Tooltip formatter={(v) => [Number(v).toFixed(3), "reward"]} />
              <ReferenceLine y={0} stroke="#999" strokeWidth={0.8} />
              <Bar
                dataKey="reward"
                fill="#1D9E75"
                isAnimationActive={false}
                shape={(props) => {
                  const { x, y, width, height, value } = props;
                  const color = (value ?? 0) >= 0 ? "#1D9E75" : "#E24B4A";
                  const rectY = (value ?? 0) >= 0 ? y : y + height;
                  const rectH = Math.abs(height || 0);
                  return <rect x={x} y={rectY} width={width} height={rectH} fill={color} opacity={0.85} />;
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
