import { useState } from "react";
import { NetworkConnectiontable } from "./components/NetworkConnectiontable";
import { NetworkTrafficChart } from "./components/NetworkTrafficChart";
import { NetworkAttackCategoryChart } from "./components/NetworkAttackCategoryChart";
import { parseSimulationResultsCidds } from "@/utils/parseSimulationResultsCidds";
import { OverallModelPerformance } from "./components/OverallModelPerformance";
import { DiscrepancyTable } from "./components/DiscrepancyTable";
import { ParsedSimulationResultCidds } from "@/types/simulationResultsCidds";
import { NetworkConnectionClassificationTable } from "./components/NetworkConnectionClassificationTable";

export function Simulation() {
  const [datasetFilePath, setDatasetFilePath] = useState<string | null>(null);
  const [networkConnectionsData, setNetworkConnectionsData] = useState<
    CIDDS_001[]
  >([]);

  const [simulationResults, setSimulationResults] =
    useState<SimulationCiddsResponse | null>(null);

  const [isPending, setIsPending] = useState(false);

  async function handleFileUpload() {
    const filePath = await window.simulationPage.openFileDialog();

    if (!filePath || filePath.length === 0) {
      console.error("No file selected");
      return;
    }

    const parsedDataset = await window.simulationPage.parseCiddsTable(
      filePath[0]
    );

    setNetworkConnectionsData(parsedDataset);

    if (filePath.length > 0) {
      setDatasetFilePath(filePath[0]);
    }
  }

  const handleSimulationStart = async () => {
    if (datasetFilePath) {
      setIsPending(true);
      try {
        const result = await window.simulationPage.startSimulationCidds(
          datasetFilePath
        );
        if (result) setSimulationResults(result);
      } finally {
        setIsPending(false);
      }
    }
  };

  let parsedResults: ParsedSimulationResultCidds | null = null;
  if (simulationResults) {
    parsedResults = parseSimulationResultsCidds(simulationResults);
  }

  return (
    <main className="px-40 py-5 flex flex-col gap-6 bg-background-dark">
      <section className="py-2 px-4 text-text-light flex gap-4 justify-between">
        <div className="flex gap-4">
          <button
            onClick={handleFileUpload}
            className="p-1.5 rounded-md bg-accent-500 hover:bg-accent-600 font-bold cursor-pointer"
          >
            Add Network Data
          </button>
        </div>

        <button
          onClick={handleSimulationStart}
          className="p-1.5 rounded-md bg-accent-500 hover:bg-accent-600 font-bold"
        >
          Start Detection
        </button>
      </section>

      <div>
        <p className="text-text-light text-2xl font-bold mb-3">
          Network Connections
        </p>
        <NetworkConnectiontable dataset={networkConnectionsData} />
      </div>

      <div className="col-span-2 grid grid-cols-2 gap-y-4">
        <p className="font-bold text-2xl text-text-light">Base Model</p>
        <p className="font-bold text-2xl text-text-light">New Model</p>
      </div>

      <div className="col-span-2 grid grid-cols-2 gap-y-4">
        <div className="w-full h-96">
          <p className="text-xl text-text-light mt-2 mb2">
            Network Traffic Classification
          </p>

          {parsedResults?.baseModel ? (
            <NetworkTrafficChart
              data={parsedResults.baseModel.binaryClassification}
            />
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>

        <div className="w-full h-96">
          <p className="text-xl text-text-light mt-2 mb-2">
            Network Traffic Classification
          </p>

          {parsedResults?.proposedModel ? (
            <NetworkTrafficChart
              data={parsedResults.proposedModel.binaryClassification}
            />
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>
      </div>

      <div className="col-span-2 grid grid-cols-2 gap-y-4">
        <div className="w-full h-96">
          <p className="text-xl text-text-light mt-2 mb-2">
            Detected Attack Types
          </p>
          {parsedResults?.baseModel ? (
            <NetworkAttackCategoryChart
              data={parsedResults.baseModel.multiClassification}
            />
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>

        <div className="w-full h-96">
          <p className="text-xl text-text-light mt-2 mb-2">
            Detected Attack Types
          </p>
          {parsedResults?.proposedModel ? (
            <NetworkAttackCategoryChart
              data={parsedResults.proposedModel.multiClassification}
            />
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>
      </div>

      <div className="col-span-2 grid grid-cols-2 gap-x-4">
        <div className="min-h-96">
          <p className="text-text-light text-xl mb-2 mt-2">
            Normal Connections
          </p>
          {simulationResults?.baseModelResults ? (
            <NetworkConnectionClassificationTable
              classificationType="Normal"
              dataset={networkConnectionsData}
              yPredicted={
                simulationResults.baseModelResults.prediction_result.y_pred
              }
            ></NetworkConnectionClassificationTable>
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>

        <div>
          <p className="text-text-light text-xl mb-2 mt-2">
            Normal Connections
          </p>
          {simulationResults?.proposedModelResults ? (
            <NetworkConnectionClassificationTable
              classificationType="Normal"
              dataset={networkConnectionsData}
              yPredicted={
                simulationResults.proposedModelResults.prediction_result.y_pred
              }
            ></NetworkConnectionClassificationTable>
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>
      </div>

      <div className="col-span-2 grid grid-cols-2 gap-x-4">
        <div className="min-h-96">
          <p className="text-text-light text-xl mb-2 mt-2">
            Anomalous Connections
          </p>
          {simulationResults?.baseModelResults ? (
            <NetworkConnectionClassificationTable
              classificationType="Anomaly"
              dataset={networkConnectionsData}
              yPredicted={
                simulationResults.baseModelResults.prediction_result.y_pred
              }
            ></NetworkConnectionClassificationTable>
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>

        <div>
          <p className="text-text-light text-xl mb-2 mt-2">
            Anomalous Connections
          </p>
          {simulationResults?.proposedModelResults ? (
            <NetworkConnectionClassificationTable
              classificationType="Anomaly"
              dataset={networkConnectionsData}
              yPredicted={
                simulationResults.proposedModelResults.prediction_result.y_pred
              }
            ></NetworkConnectionClassificationTable>
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>
      </div>

      <div className="col-span-1">
        <div className="text-text-light text-xl">Overall Model Performance</div>

        <div className="min-h-96 flex items-center justify-center">
          {simulationResults?.baseModelResults &&
          simulationResults.proposedModelResults ? (
            <div className="grid grid-cols-2">
              <div>
                <h1 className="text-2xl font-bold text-text-light px-3 mt-2">
                  Base Model
                </h1>
                <OverallModelPerformance
                  baseMetrics={simulationResults.baseModelResults.metrics}
                  referenceMetrics={
                    simulationResults.proposedModelResults.metrics
                  }
                ></OverallModelPerformance>
              </div>

              <div>
                <h1 className="text-2xl font-bold text-text-light px-3 mt-2">
                  New Model
                </h1>
                <OverallModelPerformance
                  baseMetrics={simulationResults.proposedModelResults.metrics}
                  referenceMetrics={simulationResults.baseModelResults.metrics}
                ></OverallModelPerformance>
              </div>
            </div>
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>
      </div>

      <div>
        <p className="text-text-light text-2xl font-bold mb-2 mt-2">
          Discrepancy Table
        </p>

        <div className="min-h-96 flex flex-col items-center justify-center">
          {parsedResults ? (
            <DiscrepancyTable
              parsedResults={parsedResults}
              parsedDataset={networkConnectionsData}
            ></DiscrepancyTable>
          ) : (
            <EmptyResultsMessage isLoading={isPending}></EmptyResultsMessage>
          )}
        </div>
      </div>
    </main>
  );
}

interface EmptyResultsMessageProps {
  isLoading: boolean;
}
function EmptyResultsMessage({ isLoading }: EmptyResultsMessageProps) {
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <span className="loading loading-spinner loading-lg text-accent-600"></span>
      </div>
    );
  }

  return (
    <div className="h-full flex items-center justify-center">
      <p className=" font-bold text-text-light text-center">
        Run simulation to view results
      </p>
    </div>
  );
}
