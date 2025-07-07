import { spawn } from "child_process";
import { getExecutableBasePath } from "../pathResolver.js";

const ciddsBasePaths = {
  "inference-pipeline": getExecutableBasePath(
    "model_executables",
    "cidds",
    "base",
    "inference-pipeline.exe"
  ),
  autoencoder: getExecutableBasePath(
    "model_executables",
    "cidds",
    "base",
    "autoencoder.onnx"
  ),
  ocsvm: getExecutableBasePath(
    "model_executables",
    "cidds",
    "base",
    "ocsvm.onnx"
  ),
};

const ciddsProposedPaths = {
  "inference-pipeline": getExecutableBasePath(
    "model_executables",
    "cidds",
    "proposed",
    "inference-pipeline.exe"
  ),
  autoencoder: getExecutableBasePath(
    "model_executables",
    "cidds",
    "proposed",
    "autoencoder.onnx"
  ),
  dbocsvm: getExecutableBasePath(
    "model_executables",
    "cidds",
    "proposed",
    "dbocsvm.joblib"
  ),
};

export async function startSimulationCidds(
  datasetPath: string
): Promise<SimulationCiddsResponse | null> {
  try {
    const results: SimulationCiddsResponse = {
      baseModelResults: null,
      proposedModelResults: null,
    };

    const [baseModelResults, proposedModelResults] = await Promise.all([
      predictBaseCidds(datasetPath),
      predictProposedCidds(datasetPath),
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

export async function predictBaseCidds(
  datasetPath: string
): Promise<ModelEvaluationResultCidds | null> {
  try {
    const results = await runBaseInferencePipeline({
      dataset: datasetPath,
      autoencoder: ciddsBasePaths.autoencoder,
      ocsvm: ciddsBasePaths.ocsvm,
      debug: false,
    });

    return results as ModelEvaluationResultCidds;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

interface BaseModelInferenceOptions {
  debug?: boolean;
  dataset?: string;
  autoencoder?: string;
  ocsvm?: string;
}
/**
 * Run the inference pipeline executable with the provided options
 * @param {Object} options - Configuration options
 * @param {boolean} [options.debug] - Enable debug output
 * @param {string} [options.dataset] - Path to test set CSV
 * @param {string} [options.autoencoder] - Path to autoencoder ONNX model
 * @param {string} [options.ocsvm] - Path to OCSVM ONNX model
 * @returns {Promise<ModelEvaluationResultCidds>} - The parsed results from the inference pipeline
 */
function runBaseInferencePipeline(
  options: BaseModelInferenceOptions
): Promise<ModelEvaluationResultCidds> {
  return new Promise((resolve, reject) => {
    const args = [];
    if (options.debug) args.push("--debug");
    if (options.dataset) args.push("--dataset", options.dataset);
    if (options.autoencoder) args.push("--autoencoder", options.autoencoder);
    if (options.ocsvm) args.push("--ocsvm", options.ocsvm);

    const process = spawn(ciddsBasePaths["inference-pipeline"], args);

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

export async function predictProposedCidds(
  datasetPath: string
): Promise<ModelEvaluationResultCidds | null> {
  try {
    const results = await runProposedInferencePipeline({
      dataset: datasetPath,
      autoencoder: ciddsProposedPaths.autoencoder,
      dbocsvm: ciddsProposedPaths.dbocsvm,
      debug: false,
    });

    return results as ModelEvaluationResultCidds;
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
 * @returns {Promise<ModelEvaluationResultCidds>} - The parsed results from the inference pipeline
 */
function runProposedInferencePipeline(
  options: ProposedModelInferenceOptions
): Promise<ModelEvaluationResultCidds> {
  return new Promise((resolve, reject) => {
    const args = [];
    if (options.debug) args.push("--debug");
    if (options.dataset) args.push("--dataset", options.dataset);
    if (options.autoencoder) args.push("--autoencoder", options.autoencoder);
    if (options.dbocsvm) args.push("--dbocsvm", options.dbocsvm);

    const process = spawn(ciddsProposedPaths["inference-pipeline"], args);

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
