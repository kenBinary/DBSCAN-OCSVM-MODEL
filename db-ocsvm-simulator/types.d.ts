interface Window {
  simulationPage: {
    openFileDialog: () => Promise<string[]>;
    parseCiddsTable: (csvPath: string) => Promise<CIDDS_001[]>;
    predictBaseCidds: (
      datasetPath: string
    ) => Promise<ModelEvaluationResultCidds | null>;
    predictProposedCidds: (
      datasetPath: string
    ) => Promise<ModelEvaluationResultCidds | null>;
    startSimulationCidds: (
      datasetPath: string
    ) => Promise<SimulationCiddsResponse | null>;
    predictBaseNsl: (
      datasetPath: string
    ) => Promise<ModelEvaluationResultNsl | null>;
    predictProposedNsl: (
      datasetPath: string
    ) => Promise<ModelEvaluationResultNsl | null>;
    startSimulationNsl: (
      datasetPath: string
    ) => Promise<SimulationNslResponse | null>;
  };
}

interface EventPayloadMapping {
  openFileDialog: Promise<string[]>;
  parseCiddsTable: Promise<CIDDS_001[]>;
  predictBaseCidds: Promise<ModelEvaluationResultCidds | null>;
  predictProposedCidds: Promise<ModelEvaluationResultCidds | null>;
  startSimulationCidds: Promise<SimulationCiddsResponse | null>;
  predictBaseNsl: Promise<ModelEvaluationResultNsl | null>;
  predictProposedNsl: Promise<ModelEvaluationResultNsl | null>;
  startSimulationNsl: Promise<SimulationNslResponse | null>;
}

interface CIDDS_001 {
  duration: number;
  proto: string;
  packets: number;
  bytes: number;
  attack_type: string;
  attack_class: string;
}

interface NSL_KDD{
  duration: number;
  protocol_type: string;
  bytes: number;
  attack_categorical : string;
  attack_class: string;
}

interface Metrics {
  precision: string;
  recall: string;
  f1_score: string;
  accuracy: string;
}

interface ModelEvaluationResultCidds {
  prediction_result: {
    y_true: number[];
    y_pred: number[];
  };
  test_set: {
    count: number;
    features: number;
    class_distribution: {
      benign: number;
      dos: number;
      portScan: number;
      bruteForce: number;
      pingScan: number;
    };
    latent_shape: {
      count: number;
      features: number;
    };
  };
  detection_rates: {
    benign: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
    bruteForce: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
    dos: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
    pingScan: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
    portScan: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
  };
  confusion_matrix: {
    "True Anomaly": number;
    "False Anomaly": number;
    "True Normal": number;
    "False Normal": number;
  };
  metrics: Metrics;
}

interface SimulationCiddsResponse {
  baseModelResults: ModelEvaluationResultCidds | null;
  proposedModelResults: ModelEvaluationResultCidds | null;
}

interface ModelEvaluationResultNsl {
  test_set: {
    count: number;
    features: number;
    class_distribution: {
      normal: number;
      DoS: number;
      R2L: number;
      Probe: number;
      U2R: number;
    };
  };
  detection_rates: {
    DoS: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
    Probe: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
    R2L: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
    U2R: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
    normal: {
      detection_rate: string;
      count: number;
      correctly_detected: number;
    };
  };
  confusion_matrix: {
    "True Anomaly": number;
    "False Anomaly": number;
    "True Normal": number;
    "False Normal": number;
  };
  metrics: Metrics;
}

interface SimulationNslResponse {
  baseModelResults: ModelEvaluationResultNsl | null;
  proposedModelResults: ModelEvaluationResultNsl | null;
}
