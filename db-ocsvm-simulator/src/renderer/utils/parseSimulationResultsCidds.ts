import { Data } from "@/types/chart";
import { ParsedSimulationResultCidds } from "@/types/simulationResultsCidds";

export function parseDetectionRatesBinary(
  evaluationResult: ModelEvaluationResultCidds
): Data[] {
  let normalCount = 0;
  let anomalyCount = 0;
  Object.entries(evaluationResult.detection_rates).forEach(([key, value]) => {
    if (key === "benign") {
      /*
      the correctly_detected are records classified as anomalous,
      therefore we subtract the total count of benign records
      */
      normalCount = value.count - value.correctly_detected;
    } else {
      anomalyCount += value.correctly_detected;
    }
  });

  const detectionRateBinaryData: Data[] = [
    { name: "Normal Connections", value: normalCount },
    { name: "Anomalous Connections", value: anomalyCount },
  ];

  return detectionRateBinaryData;
}

export function parseDetectionRatesMulti(
  evaluationResult: ModelEvaluationResultCidds
): Data[] {
  const detectionRatesMulti: Data[] = [];

  Object.entries(evaluationResult.detection_rates).forEach(([key, value]) => {
    if (key !== "benign" && value.correctly_detected > 0) {
      const detectionRateBinary: Data = {
        name: key,
        value: value.correctly_detected,
      };
      detectionRatesMulti.push(detectionRateBinary);
    }
  });

  return detectionRatesMulti;
}

export function parseSimulationResultsCidds(
  simulationResults: SimulationCiddsResponse | null
) {
  if (!simulationResults) {
    console.error("Simulation results are null or undefined");
    return null;
  }

  const baseModelResults = simulationResults.baseModelResults;
  const proposedModelResults = simulationResults.proposedModelResults;

  if (!baseModelResults || !proposedModelResults) {
    console.error("Base or Proposed model results are null or undefined");
    return null;
  }

  const parsedResults: ParsedSimulationResultCidds = {
    y_true: baseModelResults.prediction_result.y_true,
    baseModel: {
      binaryClassification: parseDetectionRatesBinary(baseModelResults),
      multiClassification: parseDetectionRatesMulti(baseModelResults),
      y_pred: baseModelResults.prediction_result.y_pred,
    },
    proposedModel: {
      binaryClassification: parseDetectionRatesBinary(proposedModelResults),
      multiClassification: parseDetectionRatesMulti(proposedModelResults),
      y_pred: proposedModelResults.prediction_result.y_pred,
    },
  };

  return parsedResults;
}
