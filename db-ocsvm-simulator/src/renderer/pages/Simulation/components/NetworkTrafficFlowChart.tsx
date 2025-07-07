import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TrafficFlowDataPoint } from "@/types/simulationResults";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const CustomYAxisTick = (props: any) => {
  const { x, y, payload } = props;
  return (
    <g transform={`translate(${x},${y})`}>
      <text x={0} y={0} dy={1} textAnchor="end" fill="White">
        {`${payload.value}`}
      </text>
    </g>
  );
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const CustomXAxisTick = (props: any) => {
  const { x, y, payload } = props;
  return (
    <g transform={`translate(${x},${y})`}>
      <text x={0} y={0} dy={20} textAnchor="end" fill="White">
        {payload.value}
      </text>
    </g>
  );
};

interface NetworkTrafficFlowChartProps {
  data: TrafficFlowDataPoint[];
}
export function NetworkTrafficFlowChart({
  data,
}: NetworkTrafficFlowChartProps) {
  return (
    <div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={data}
          margin={{ top: 5, right: 30, left: 30, bottom: 5 }}
        >
          <XAxis dataKey="name" tick={<CustomXAxisTick />} />
          <YAxis tick={<CustomYAxisTick />} />
          <CartesianGrid stroke="" />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="Anomalies"
            stroke="red"
            strokeWidth={3}
          />
          <Line
            type="monotone"
            dataKey="Normal"
            stroke="green"
            strokeWidth={3}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
