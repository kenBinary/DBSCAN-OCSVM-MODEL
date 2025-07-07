import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

interface data {
  name: string;
  value: number;
}

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#A28BFE"];

interface NetworkAttackCategoryChartProps {
  data: data[];
}
export function NetworkAttackCategoryChart({
  data,
}: NetworkAttackCategoryChartProps) {
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
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </ResponsiveContainer>
  );
}
