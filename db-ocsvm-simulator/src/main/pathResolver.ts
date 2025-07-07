import path from "node:path";
import { app } from "electron";

export function getPreloadPath() {
  return path.join(app.getAppPath(), "dist-electron", "preload.cjs");
}

export function getExecutableBasePath(...segments: string[]) {
  const isProd = app.isPackaged;
  const base = isProd
    ? process.resourcesPath
    : path.join(app.getAppPath(), "dist-electron");
  return path.join(base, ...segments);
}
