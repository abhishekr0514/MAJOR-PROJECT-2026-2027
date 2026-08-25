const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

let mainWindow = null;
let flProcess = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1360,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    title: "MedShield FL — Privacy-Preserving Multimodal Healthcare AI Platform",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const isDev = process.env.NODE_ENV === "development" || !app.isPackaged;
  const startUrl = isDev
    ? "http://localhost:5173"
    : `file://${path.join(__dirname, "../dist/index.html")}`;

  mainWindow.loadURL(startUrl);

  mainWindow.on("closed", () => {
    mainWindow = null;
    if (flProcess) {
      flProcess.kill();
      flProcess = null;
    }
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

// IPC Handler: Native File Open Dialog
ipcMain.handle("dialog:openFile", async (_event, options = {}) => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile"],
    filters: options.filters || [
      { name: "Datasets", extensions: ["csv", "npy", "json"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return result.filePaths[0];
});

// IPC Handler: Start FL Training Process from Electron Desktop UI
ipcMain.handle("fl:startNode", async (_event, config = {}) => {
  if (flProcess) {
    return { success: false, message: "FL process is already running." };
  }

  const projectRoot = path.resolve(__dirname, "../..");
  const server = config.server || "127.0.0.1:8080";
  const hospitalId = config.hospitalId || "hospital_alpha";
  const csvFile = config.csvFile || "client/data/hospital_alpha_data.csv";

  const args = [
    "run",
    "python",
    "client/fl_client.py",
    "--server",
    server,
    "--hospital-id",
    hospitalId,
    "--csv-file",
    csvFile,
  ];

  try {
    flProcess = spawn("uv", args, {
      cwd: projectRoot,
      env: { ...process.env, PYTHONPATH: "." },
    });

    flProcess.stdout.on("data", (data) => {
      if (mainWindow) {
        mainWindow.webContents.send("fl:log", {
          type: "stdout",
          text: data.toString(),
        });
      }
    });

    flProcess.stderr.on("data", (data) => {
      if (mainWindow) {
        mainWindow.webContents.send("fl:log", {
          type: "stderr",
          text: data.toString(),
        });
      }
    });

    flProcess.on("close", (code) => {
      if (mainWindow) {
        mainWindow.webContents.send("fl:log", {
          type: "exit",
          text: `Process exited with code ${code}`,
        });
      }
      flProcess = null;
    });

    return { success: true, message: `Started FL Client for ${hospitalId}` };
  } catch (err) {
    return { success: false, message: err.message };
  }
});

// IPC Handler: Stop FL Training Process
ipcMain.handle("fl:stopNode", async () => {
  if (flProcess) {
    flProcess.kill();
    flProcess = null;
    return { success: true, message: "FL Process terminated." };
  }
  return { success: false, message: "No active FL process." };
});
