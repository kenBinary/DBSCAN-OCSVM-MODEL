import { Data } from "@/types/chart";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = {
  "Normal Connections": "#4ade80",
  "Anomalous Connections": "#f87171",
};

interface NetworkTrafficChartProps {
  data: Data[];
}

export function NetworkTrafficChart({ data }: NetworkTrafficChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart width={400} height={400}>
        <Pie
          dataKey="value"
          isAnimationActive={false}
          data={data}
          cx="50%"
          cy="50%"
          outerRadius={80}
          fill="#8884d8"
          label
        >
          {data.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={COLORS[entry.name as keyof typeof COLORS] || "#8884d8"}
            />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </ResponsiveContainer>
  );
}
