import { ParsedSimulationResultCidds } from "@/types/simulationResultsCidds";
import { useEffect, useState } from "react";
import { EmptyTable } from "./EmptyTable";
import { generateDiscrepancyTableData } from "@/utils/generateDiscrepancyTableData";
import clsx from "clsx";

interface DiscrepancyTableProps {
  parsedResults: ParsedSimulationResultCidds;
  parsedDataset: CIDDS_001[];
}

export function DiscrepancyTable({
  parsedResults,
  parsedDataset,
}: DiscrepancyTableProps) {
  const features = Object.keys(parsedDataset[0]);

  const discrepancyTableData = generateDiscrepancyTableData(parsedResults);

  const datasetLength = parsedDataset.length;
  const pageSize = 10;
  const [pageNumber, setPageNumber] = useState(0);
  const pageCount = Math.ceil(datasetLength / pageSize);

  useEffect(() => {
    setPageNumber(0);
  }, [parsedDataset]);

  if (datasetLength === 0) {
    return <EmptyTable></EmptyTable>;
  }

  const paginatedDataset = parsedDataset.slice(
    pageNumber * pageSize,
    pageNumber * pageSize + pageSize
  );

  const handlePageChange = (direction: "next" | "prev") => {
    if (direction === "next" && pageNumber < pageCount - 1) {
      setPageNumber((prev) => prev + 1);
    } else if (direction === "prev" && pageNumber > 0) {
      setPageNumber((prev) => prev - 1);
    }
  };

  return (
    <>
      <table className="w-full border-primary-700 border">
        <thead>
          <tr className="text-text-light bg-table-table-header ">
            {features.map((feature) => {
              return (
                <th className="text-text-light  px-2.5 py-3.5" key={feature}>
                  {feature}
                </th>
              );
            })}

            <th className="text-text-light  px-2.5 py-3.5">
              Base Model Prediction
            </th>
            <th className="text-text-light  px-2.5 py-3.5">
              New Model Prediction
            </th>
            <th className="text-text-light  ">Status</th>
          </tr>
        </thead>

        <tbody>
          {paginatedDataset.map((record, index) => {
            const currentIndex = index + pageNumber * pageSize;

            const basePrediction =
              discrepancyTableData[currentIndex].basePrediction;
            const proposedPrediction =
              discrepancyTableData[currentIndex].proposedPrediction;
            const status = discrepancyTableData[currentIndex].status;

            return (
              <tr
                className="border-t border-b text-[#8FB0CC] hover:bg-gray-200 hover:bg-opacity-20"
                key={index}
              >
                <td className="text-center">
                  <span className="bg-yellow-700 rounded-full px-3 py-1 text-text-light font-semibold">
                    {currentIndex + 1}
                  </span>
                </td>

                {Object.entries(record).map(([key, value], idx) => {
                  const isAttackType =
                    key === "attack_type" || key === "attack_class";
                  const isNormal = value === "benign" || value === "normal";

                  if (key === "duration") {
                    return;
                  }

                  if (isAttackType) {
                    return (
                      <td
                        className={`px-3.5 py-3.5 text-center ${
                          isNormal ? "text-green-400" : "text-red-400"
                        }`}
                        key={idx}
                      >
                        <div>{value}</div>
                      </td>
                    );
                  }

                  return (
                    <td className="px-3.5 py-3.5 text-center" key={idx}>
                      <div>{value}</div>
                    </td>
                  );
                })}

                <td className="text-center">
                  <span className="px-3.5 py-3.5 text-center">
                    {basePrediction}
                  </span>
                </td>

                <td className="text-center">
                  <span className="px-3.5 py-3.5 text-center">
                    {proposedPrediction}
                  </span>
                </td>

                <td className="text-center">
                  <span
                    className={clsx("px-3.5 py-3.5 text-center", {
                      "text-green-400": status.includes(
                        "Both Models Correctly Classified"
                      ),
                      "text-red-400": status.includes(
                        "Both Models Misclassified"
                      ),
                      "text-orange-400":
                        status.includes("Base Model Misclassified") ||
                        status.includes("New Model Misclassified"),
                    })}
                  >
                    {status}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="join w-full flex justify-center p-2">
        <button
          className="join-item btn btn-sm"
          onClick={() => {
            handlePageChange("prev");
          }}
        >
          «
        </button>
        <button className="join-item btn btn-sm">
          Page {pageNumber + 1} of {pageCount}
        </button>
        <button
          className="join-item btn btn-sm"
          onClick={() => {
            handlePageChange("next");
          }}
        >
          »
        </button>
      </div>
    </>
  );
}
