import { useEffect, useState } from "react";
import { EmptyTable } from "./EmptyTable";
import clsx from "clsx";

interface NetworkConnectionClassificationTable {
  dataset: CIDDS_001[];
  classificationType: "Normal" | "Anomaly";
  yPredicted: number[]; // Array of predicted labels (-1 or 1) : -1 for anomaly, 1 for normal
}

export function NetworkConnectionClassificationTable({
  dataset,
  classificationType,
  yPredicted,
}: NetworkConnectionClassificationTable) {
  const filteredDataset = dataset
    .map((record, index) => ({ record, originalIndex: index }))
    .filter(({ originalIndex }) => {
      if (classificationType === "Normal") {
        return yPredicted[originalIndex] === 1;
      } else {
        return yPredicted[originalIndex] === -1;
      }
    });

  const datasetLength = filteredDataset.length;
  const pageSize = 6;
  const [pageNumber, setPageNumber] = useState(0);
  const pageCount = Math.ceil(datasetLength / pageSize);

  useEffect(() => {
    setPageNumber(0);
  }, [dataset]);

  if (datasetLength === 0) {
    return <EmptyTable></EmptyTable>;
  }

  const features = Object.keys(filteredDataset[0].record).filter((feature) => {
    return feature !== "attack_class";
  });

  const paginatedDataset = filteredDataset.slice(
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

  if (dataset.length !== yPredicted.length) {
    console.error("Dataset length and yPredicted length do not match");
    return null;
  }

  return (
    <div>
      <table className="w-full border border-primary-700">
        <thead>
          <tr className="text-text-light bg-table-table-header ">
            <th>Connection #</th>
            {features.map((feature, index) => {
              return (
                <th className="px-1.5 py-2.5" key={index}>
                  {feature}
                </th>
              );
            })}
            <th className="px-1.5 py-2.5">Classified As</th>
          </tr>
        </thead>

        <tbody>
          {paginatedDataset.map(({ record, originalIndex }) => {
            return (
              <tr
                className="border-t border-b text-[#8FB0CC] hover:bg-gray-200 hover:bg-opacity-20"
                key={originalIndex}
              >
                <td className="text-center">
                  <span className="bg-yellow-700 rounded-full px-2 py-1 text-text-light font-semibold">
                    {originalIndex + 1}
                  </span>
                </td>

                {Object.entries(record).map(([key, value], idx) => {
                  const isAttackType =
                    key === "attack_type" || key === "attack_class";
                  const isNormal = value === "benign" || value === "normal";

                  if (key === "attack_class") {
                    return;
                  }

                  if (isAttackType) {
                    return (
                      <td
                        className={`px-2.5 py-2.5 text-center ${
                          isNormal ? "text-green-400" : "text-red-400"
                        }`}
                        key={idx}
                      >
                        <div>{value}</div>
                      </td>
                    );
                  }

                  return (
                    <td className="px-2.5 py-2.5 text-center" key={idx}>
                      <div>{value}</div>
                    </td>
                  );
                })}
                <td
                  className={clsx("px-2.5 py-2.5 text-center", {
                    "text-green-400": classificationType === "Normal",
                    "text-red-400": classificationType === "Anomaly",
                  })}
                >
                  <div>{classificationType}</div>
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
    </div>
  );
}
