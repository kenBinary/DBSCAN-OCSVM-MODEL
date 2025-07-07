import { useEffect, useState } from "react";
import { EmptyTable } from "./EmptyTable";

interface NetworkConnectiontableProps {
  dataset: CIDDS_001[];
}

export function NetworkConnectiontable({
  dataset,
}: NetworkConnectiontableProps) {
  const datasetLength = dataset.length;
  const pageSize = 10;
  const [pageNumber, setPageNumber] = useState(0);
  const pageCount = Math.ceil(datasetLength / pageSize);

  useEffect(() => {
    setPageNumber(0);
  }, [dataset]);

  if (datasetLength === 0) {
    return <EmptyTable></EmptyTable>;
  }

  const features = Object.keys(dataset[0]);
  const paginatedDataset = dataset.slice(
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
    <div>
      <table className="w-full border border-primary-700">
        <thead>
          <tr className="text-text-light bg-table-table-header ">
            <th>Connection #</th>
            {features.map((feature) => {
              return (
                <th className="px-2.5 py-3.5 " key={feature}>
                  {feature}
                </th>
              );
            })}
          </tr>
        </thead>

        <tbody>
          {paginatedDataset.map((record, index) => {
            return (
              <tr
                className="border-t border-b text-[#8FB0CC] hover:bg-gray-200 hover:bg-opacity-20"
                key={index}
              >
                <td className="text-center">
                  <span className="bg-yellow-700 rounded-full px-3 py-1 text-text-light font-semibold">
                    {index + 1 + pageNumber * pageSize}
                  </span>
                </td>

                {Object.entries(record).map(([key, value], idx) => {
                  const isAttackType =
                    key === "attack_type" || key === "attack_class";
                  const isNormal = value === "benign" || value === "normal";

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
