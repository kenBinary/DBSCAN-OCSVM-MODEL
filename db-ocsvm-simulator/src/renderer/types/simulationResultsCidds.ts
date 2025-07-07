import { Data } from "./chart";

export interface ParsedSimulationResultCidds {
  y_true: number[];
  baseModel: {
    binaryClassification: Data[];
    multiClassification: Data[];
    y_pred: number[];
  };
  proposedModel: {
    binaryClassification: Data[];
    multiClassification: Data[];
    y_pred: number[];
  };
}
