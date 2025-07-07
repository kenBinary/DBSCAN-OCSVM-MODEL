import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleCheck } from "@fortawesome/free-solid-svg-icons";
import clsx from "clsx";

function metricDifference(startValue: number, finalValue: number) {
  const diff = ((finalValue - startValue) / startValue) * 100;

  if (diff > 0) {
    return `+${diff.toFixed(2)}%`;
  }
  if (diff < 0) {
    return `${diff.toFixed(2)}%`;
  }
  return "No change";
}

interface OverallModelPerformanceProps {
  baseMetrics: Metrics;
  referenceMetrics: Metrics;
}
export function OverallModelPerformance({
  baseMetrics,
  referenceMetrics,
}: OverallModelPerformanceProps) {
  return (
    <div className="grid-cols-2 grid gap-4 text-text-light font-medium text-xl px-3 py-4">
      <MetricCard
        metricType="Accuracy"
        value={baseMetrics.accuracy}
        valueChange={metricDifference(
          parseInt(referenceMetrics.accuracy),
          parseInt(baseMetrics.accuracy)
        )}
      ></MetricCard>

      <MetricCard
        metricType="Precision"
        value={baseMetrics.precision}
        valueChange={metricDifference(
          parseInt(referenceMetrics.precision),
          parseInt(baseMetrics.precision)
        )}
      ></MetricCard>

      <MetricCard
        metricType="Recall"
        value={baseMetrics.recall}
        valueChange={metricDifference(
          parseInt(referenceMetrics.recall),
          parseInt(baseMetrics.recall)
        )}
      ></MetricCard>

      <MetricCard
        metricType="F1 Score"
        value={baseMetrics.f1_score}
        valueChange={metricDifference(
          parseInt(referenceMetrics.f1_score),
          parseInt(baseMetrics.f1_score)
        )}
      ></MetricCard>
    </div>
  );
}

interface MetricCardProps {
  metricType: "Accuracy" | "Precision" | "Recall" | "F1 Score";
  value: string;
  valueChange: string;
}
function MetricCard({ metricType, value, valueChange }: MetricCardProps) {
  let metricDescription = "Overall correctness";

  if (metricType === "Precision") {
    metricDescription = "Correct anomaly predictions";
  } else if (metricType === "Recall") {
    metricDescription = "Found all actual anomalies";
  } else if (metricType === "F1 Score") {
    metricDescription = "Balances precision and recall";
  } else if (metricType === "Accuracy") {
    metricDescription = "Overall correctness";
  }

  return (
    <div className="bg-table-table-header rounded-lg p-4">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <FontAwesomeIcon icon={faCircleCheck} />
          <p>{metricType}</p>
        </div>
        <div
          className={clsx("badge badge-md text-white tracking-wide", {
            "badge-success": valueChange && valueChange.startsWith("+"),
            "badge-error": valueChange && valueChange.startsWith("-"),
            "badge-neutral": valueChange === "No change",
          })}
        >
          {valueChange}
        </div>
      </div>

      <div>
        <p className="text-4xl font-bold mt-4 ">{value}%</p>
        <progress
          className="progress progress-primary w-full h-2.5 bg-primary-700"
          value={value}
          max="100"
        ></progress>
      </div>

      <div className="text-sm text-primary-200">{metricDescription}</div>
    </div>
  );
}
