import { ipcMain } from "electron";

export function ipcMainHandle<Channel extends keyof EventPayloadMapping>(
  channel: Channel,
  handler: (
    ...args: Parameters<Window["simulationPage"][Channel]>
  ) => ReturnType<Window["simulationPage"][Channel]>
) {
  ipcMain.handle(
    channel,
    (_event, ...args: Parameters<Window["simulationPage"][Channel]>) =>
      handler(...args)
  );
}
