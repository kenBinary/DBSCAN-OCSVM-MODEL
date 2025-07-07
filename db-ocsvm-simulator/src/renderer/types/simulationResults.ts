export interface SimulationResults {
  networkClassificationBase: { name: string; value: number }[];
  networkAttackTypes: { name: string; value: number }[];
  normalConnections: CIDDS_001[];
  anomalousConnections: CIDDS_001[];
}

export interface TrafficFlowDataPoint {
  name: string; // The time period or label for the data point (e.g., '12 AM', '1 AM')
  Anomalies: number; // The number of anomalies detected during this time period
  Normal: number; // The number of non-anomalies detected during this time period
}

export interface NetworkAttackTypes {
  name: string;
  value: number;
}

export interface ParsedSimulationResults {
  discrepancyTableData: DiscrepancyTableData[];
  baseModelResults: {
    networkClassificationBase: { name: string; value: number }[];
    networkAttackTypesBase: NetworkAttackTypes[];
    normalConnectionsBase: CIDDS_001[];
    anomalousConnectionsBase: CIDDS_001[];
    overallMetricsBase: OverallMetrics;
  };
  proposedModelResults: {
    networkClassificationProposed: { name: string; value: number }[];
    networkAttackTypesProposed: NetworkAttackTypes[];
    normalConnectionsProposed: CIDDS_001[];
    anomalousConnectionsProposed: CIDDS_001[];
    overallMetricsProposed: OverallMetrics;
  };
}

interface OverallMetrics {
  accuracy: string;
  precision: string;
  recall: string;
  f1_score: string;
}

export interface DiscrepancyTableData {
  "Connection #": number;
  Protocol: string;
  Packets: number;
  Bytes: number;
  "Actual Connection Type": string;
  "Base Model Prediction": string;
  "Proposed Model Prediction": string;
  Status: string;
}
