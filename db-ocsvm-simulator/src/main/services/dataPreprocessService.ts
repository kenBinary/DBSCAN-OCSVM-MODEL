import { parse } from "csv-parse";
import { createReadStream } from "fs";
import { CIDDS_001 } from "../types/dataset_types.js";

export async function parseDatasetToTableCidds(
  csvPath: string
): Promise<CIDDS_001[]> {
  return new Promise((resolve, reject) => {
    const parsedDataset: CIDDS_001[] = [];

    createReadStream(csvPath)
      .pipe(parse({ columns: true }))
      .on("data", (record) => {
        /*
         * @param record - When columns:true is set, each record is an object
         * where keys are column headers and values are string representations
         * of the data in each cell
         * Example: { [columnName: string]: string, ... }
         */
        let protocol = "";

        for (const [feature, value] of Object.entries(record)) {
          if (feature.startsWith("proto_") && value === "1.0") {
            protocol = feature.split("proto_")[1];
          }
        }

        parsedDataset.push({
          duration: parseFloat(
            (parseFloat(record["duration"]) * 1000).toFixed(3)
          ),
          proto: protocol.trim(),
          packets: parseFloat(
            (parseFloat(record["packets"]) * 1000).toFixed(3)
          ),
          bytes: parseFloat((parseFloat(record["bytes"]) * 1000).toFixed(3)),
          attack_type: record["attack_categorical"],
          attack_class:
            record["attack_categorical"] === "benign" ? "normal" : "anomaly",
        });
      })
      .on("end", () => {
        resolve(parsedDataset);
      })
      .on("error", (err) =>
        reject(new Error(`CSV processing error: ${err.message}`))
      );
  });
}
