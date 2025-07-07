import { dialog } from "electron";

export async function selectDatasetFile() {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [
      {
        name: "Dataset Files",
        extensions: ["csv"],
      },
    ],
  });
  return result.filePaths;
}
