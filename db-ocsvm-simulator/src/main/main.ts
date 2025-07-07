import { app, BrowserWindow } from "electron";
import { getPreloadPath } from "./pathResolver.js";
import { ipcMainHandle } from "./util.js";
import { selectDatasetFile } from "./services/fileService.js";
import { parseDatasetToTableCidds } from "./services/dataPreprocessService.js";
import {
  predictBaseCidds,
  predictProposedCidds,
  startSimulationCidds,
} from "./services/ciddsPredictionService.js";
import {
  predictBaseNsl,
  predictProposedNsl,
  startSimulationNsl,
} from "./services/nslPredictionService.js";
import path from "path";

const createWindow = () => {
  const isDev = process.env.NODE_ENV === "development";

  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 720,
    webPreferences: {
      preload: getPreloadPath(),
    },
    autoHideMenuBar: false,
  });

  if (isDev) {
    console.log("Running in development mode");
    mainWindow.loadURL("http://localhost:5173/");
    mainWindow.webContents.openDevTools();
  } else {
    const uiPath = path.join(app.getAppPath(), "/dist-react/index.html");
    mainWindow.loadFile(uiPath);
  }
};

app.whenReady().then(() => {
  createWindow();

  ipcMainHandle("openFileDialog", selectDatasetFile);
  ipcMainHandle("parseCiddsTable", parseDatasetToTableCidds);
  ipcMainHandle("predictBaseCidds", predictBaseCidds);
  ipcMainHandle("predictProposedCidds", predictProposedCidds);
  ipcMainHandle("startSimulationCidds", startSimulationCidds);

  ipcMainHandle("predictBaseNsl", predictBaseNsl);
  ipcMainHandle("predictProposedNsl", predictProposedNsl);
  ipcMainHandle("startSimulationNsl", startSimulationNsl);

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
