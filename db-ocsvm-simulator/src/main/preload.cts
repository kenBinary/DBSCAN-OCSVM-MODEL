const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("simulationPage", {
  openFileDialog: () => ipcInvoke("openFileDialog"),
  parseCiddsTable: (csvPath) => ipcInvoke("parseCiddsTable", csvPath),
  predictBaseCidds: (datasetPath) => ipcInvoke("predictBaseCidds", datasetPath),
  predictProposedCidds: (datasetPath) =>
    ipcInvoke("predictProposedCidds", datasetPath),
  startSimulationCidds: (datasetPath) =>
    ipcInvoke("startSimulationCidds", datasetPath),
  predictBaseNsl: (datasetPath) => ipcInvoke("predictBaseNsl", datasetPath),
  predictProposedNsl: (datasetPath) =>
    ipcInvoke("predictProposedNsl", datasetPath),
  startSimulationNsl: (datasetPath) =>
    ipcInvoke("startSimulationNsl", datasetPath),
} satisfies Window["simulationPage"]);

// type helper to unwrap promises
type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;

function ipcInvoke<Channel extends keyof EventPayloadMapping>(
  key: Channel,
  ...args: Parameters<Window["simulationPage"][Channel]>
): Promise<UnwrapPromise<EventPayloadMapping[Channel]>> {
  return ipcRenderer.invoke(key, ...args) as Promise<
    UnwrapPromise<EventPayloadMapping[Channel]>
  >;
}
