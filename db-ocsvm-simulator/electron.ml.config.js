import { copy, pathExists } from "fs-extra";
import { join } from "path";

const sourcePath = ["./src/main/model_executables"];
const destinationPath = "./dist-electron";

async function copyDirectory(sourcePath, destinationPath) {
  try {
    // Checking for source paths
    const sourceExists = await pathExists(sourcePath);
    if (!sourceExists) {
      console.log(`Source directory ${sourcePath} does not exist`);
      return;
    }

    const folderName = sourcePath.split("/").pop();
    const fullDestPath = join(destinationPath, folderName);

    await copy(sourcePath, fullDestPath, { overwrite: true });
    console.log(`Successfully copied ${sourcePath} to ${fullDestPath}`);
  } catch (error) {
    console.error(`Error copying directory: ${error}`);
  }
}

Promise.all(sourcePath.map((src) => copyDirectory(src, destinationPath)))
  .then(() => console.log("All directories processed"))
  .catch((err) => console.error("Error during copy:", err));
