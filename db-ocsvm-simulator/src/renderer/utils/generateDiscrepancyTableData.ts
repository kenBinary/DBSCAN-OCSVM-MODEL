import { ModelPrediction } from "@/types/discrepancyTable";
import { ParsedSimulationResultCidds } from "@/types/simulationResultsCidds";

function isNormalOrAnomalyString(value: number): string {
  return value === 1 ? "✔️ normal" : "❗ anomaly";
}

export function generateDiscrepancyTableData(
  parsedResults: ParsedSimulationResultCidds
): ModelPrediction[] {
  return parsedResults.y_true.map((y_true, index) => {
    const basePrediction = parsedResults.baseModel.y_pred[index];
    const proposedPrediction = parsedResults.proposedModel.y_pred[index];

    if (basePrediction === proposedPrediction && basePrediction === y_true) {
      return {
        basePrediction: isNormalOrAnomalyString(basePrediction),
        proposedPrediction: isNormalOrAnomalyString(proposedPrediction),
        status: "Both Models Correctly Classified",
      };
    } else if (basePrediction !== y_true && proposedPrediction === y_true) {
      return {
        basePrediction: isNormalOrAnomalyString(basePrediction),
        proposedPrediction: isNormalOrAnomalyString(proposedPrediction),
        status: "Base Model Misclassified",
      };
    } else if (basePrediction === y_true && proposedPrediction !== y_true) {
      return {
        basePrediction: isNormalOrAnomalyString(basePrediction),
        proposedPrediction: isNormalOrAnomalyString(proposedPrediction),
        status: "New Model Misclassified",
      };
    } else {
      return {
        basePrediction: isNormalOrAnomalyString(basePrediction),
        proposedPrediction: isNormalOrAnomalyString(proposedPrediction),
        status: "Both Models Misclassified",
      };
    }
  });
}
