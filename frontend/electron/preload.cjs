const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  selectFile: (options) => ipcRenderer.invoke("dialog:openFile", options),
  startFLNode: (config) => ipcRenderer.invoke("fl:startNode", config),
  stopFLNode: () => ipcRenderer.invoke("fl:stopNode"),
  onFLLog: (callback) => ipcRenderer.on("fl:log", (_event, data) => callback(data)),
});
