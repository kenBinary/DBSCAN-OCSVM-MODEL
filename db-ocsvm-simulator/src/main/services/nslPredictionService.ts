import { spawn } from "child_process";
import { getExecutableBasePath } from "../pathResolver.js";

const nslBasePaths = {
  "inference-pipeline": getExecutableBasePath(
    "model_executables",
    "nsl",
    "base",
    "inference-pipeline.exe"
  ),
  ocsvm: getExecutableBasePath(
    "model_executables",
    "nsl",
    "base",
    "ocsvm.onnx"
  ),
};

const nslProposedPaths = {
  "inference-pipeline": getExecutableBasePath(
    "model_executables",
    "nsl",
    "proposed",
    "inference-pipeline.exe"
  ),
  autoencoder: getExecutableBasePath(
    "model_executables",
    "nsl",
    "proposed",
    "autoencoder.onnx"
  ),
  dbocsvm: getExecutableBasePath(
    "model_executables",
    "nsl",
    "proposed",
    "dbocsvm.joblib"
  ),
};

export async function startSimulationNsl(
  datasetPath: string
): Promise<SimulationNslResponse | null> {
  try {
    const results: SimulationNslResponse = {
      baseModelResults: null,
      proposedModelResults: null,
    };

    const [baseModelResults, proposedModelResults] = await Promise.all([
      predictBaseNsl(datasetPath),
      predictProposedNsl(datasetPath),
    ]);

    results.baseModelResults = baseModelResults;
    results.proposedModelResults = proposedModelResults;

    return results;
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error:", error.message);
    }
    return null;
  }
}

export async function predictBaseNsl(
  datasetPath: string
): Promise<ModelEvaluationResultNsl | null> {
  try {
    const results = await runBaseInferencePipeline({
      dataset: datasetPath,
      ocsvm: nslBasePaths.ocsvm,
      debug: false,
    });

    return results as ModelEvaluationResultNsl;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

interface BaseModelInferenceOptions {
  debug?: boolean;
  dataset?: string;
  ocsvm?: string;
}
/**
 * Run the inference pipeline executable with the provided options
 * @param {Object} options - Configuration options
 * @param {boolean} [options.debug] - Enable debug output
 * @param {string} [options.dataset] - Path to test set CSV
 * @param {string} [options.autoencoder] - Path to autoencoder ONNX model
 * @param {string} [options.ocsvm] - Path to OCSVM ONNX model
 * @returns {Promise<ModelEvaluationResultNsl>} - The parsed results from the inference pipeline
 */
function runBaseInferencePipeline(
  options: BaseModelInferenceOptions
): Promise<ModelEvaluationResultNsl> {
  return new Promise((resolve, reject) => {
    const args = [];
    if (options.debug) args.push("--debug");
    if (options.dataset) args.push("--dataset", options.dataset);
    if (options.ocsvm) args.push("--ocsvm", options.ocsvm);

    const process = spawn(nslBasePaths["inference-pipeline"], args);

    let stdoutData = "";
    let stderrData = "";

    process.stdout.on("data", (data) => {
      stdoutData += data.toString();
    });

    process.stderr.on("data", (data) => {
      stderrData += data.toString();
    });

    process.on("error", (error) => {
      reject(new Error(`Failed to start inference process: ${error.message}`));
    });

    process.on("close", (code) => {
      if (code !== 0) {
        return reject(
          new Error(`Inference process exited with code ${code}: ${stderrData}`)
        );
      }

      try {
        const result = JSON.parse(stdoutData);
        resolve(result);
      } catch (error) {
        reject(new Error(`Failed to parse inference results: ${error}`));
      }
    });
  });
}

export async function predictProposedNsl(
  datasetPath: string
): Promise<ModelEvaluationResultNsl | null> {
  try {
    const results = await runProposedInferencePipeline({
      dataset: datasetPath,
      autoencoder: nslProposedPaths.autoencoder,
      dbocsvm: nslProposedPaths.dbocsvm,
      debug: false,
    });

    return results as ModelEvaluationResultNsl;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

interface ProposedModelInferenceOptions {
  debug?: boolean;
  dataset?: string;
  autoencoder?: string;
  dbocsvm?: string;
}
/**
 * Run the inference pipeline executable with the provided options
 * @param {Object} options - Configuration options
 * @param {boolean} [options.debug] - Enable debug output
 * @param {string} [options.dataset] - Path to test set CSV
 * @param {string} [options.autoencoder] - Path to autoencoder ONNX model
 * @param {string} [options.ocsvm] - Path to OCSVM ONNX model
 * @returns {Promise<ModelEvaluationResultNsl>} - The parsed results from the inference pipeline
 */
function runProposedInferencePipeline(
  options: ProposedModelInferenceOptions
): Promise<ModelEvaluationResultNsl> {
  return new Promise((resolve, reject) => {
    const args = [];
    if (options.debug) args.push("--debug");
    if (options.dataset) args.push("--dataset", options.dataset);
    if (options.autoencoder) args.push("--autoencoder", options.autoencoder);
    if (options.dbocsvm) args.push("--dbocsvm", options.dbocsvm);

    const process = spawn(nslProposedPaths["inference-pipeline"], args);

    let stdoutData = "";
    let stderrData = "";

    process.stdout.on("data", (data) => {
      stdoutData += data.toString();
    });

    process.stderr.on("data", (data) => {
      stderrData += data.toString();
    });

    process.on("error", (error) => {
      reject(new Error(`Failed to start inference process: ${error.message}`));
    });

    process.on("close", (code) => {
      if (code !== 0) {
        return reject(
          new Error(`Inference process exited with code ${code}: ${stderrData}`)
        );
      }

      try {
        const result = JSON.parse(stdoutData);
        resolve(result);
      } catch (error) {
        reject(new Error(`Failed to parse inference results: ${error}`));
      }
    });
  });
}
