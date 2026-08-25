#!/usr/bin/env python3
"""MedShield FL — Dedicated Standalone Hospital Client Desktop App (Tkinter GUI)."""

import os
import sys
import time
import socket
import subprocess
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a remote or local TCP port is open and accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class MedShieldClientApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("🛡️ MedShield FL — Hospital Client Node Desktop Launcher")
        self.geometry("940x700")
        self.minsize(820, 620)
        self.configure(bg="#0f172a")  # Dark Slate bg

        self.process = None
        self.server_process = None
        self.log_queue = queue.Queue()

        self.setup_styles()
        self.create_widgets()
        self.after(100, self.poll_log_queue)

    def setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Custom dark theme styles
        self.style.configure("TFrame", background="#0f172a")
        self.style.configure("Header.TLabel", background="#0f172a", foreground="#10b981", font=("Helvetica", 16, "bold"))
        self.style.configure("SubHeader.TLabel", background="#0f172a", foreground="#94a3b8", font=("Helvetica", 10))
        self.style.configure("Section.TLabelframe", background="#1e293b", foreground="#38bdf8", font=("Helvetica", 11, "bold"))
        self.style.configure("Section.TLabelframe.Label", background="#1e293b", foreground="#38bdf8")
        self.style.configure("TLabel", background="#1e293b", foreground="#e2e8f0", font=("Helvetica", 10))
        self.style.configure("TEntry", font=("Consolas", 10))

    def create_widgets(self):
        # Header Banner Frame
        header_frame = ttk.Frame(self, padding=15)
        header_frame.pack(fill="x")

        title_label = ttk.Label(header_frame, text="🛡️ MedShield FL — Client Node Desktop App", style="Header.TLabel")
        title_label.pack(anchor="w")

        subtitle = ttk.Label(
            header_frame,
            text="Decentralized Privacy-Preserving Multimodal Federated Learning Client Node",
            style="SubHeader.TLabel",
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # Main Content Frame
        content_frame = ttk.Frame(self, padding=15)
        content_frame.pack(fill="both", expand=True)

        # Connection & Parameter Group Box
        param_box = ttk.LabelFrame(content_frame, text=" 🏥 Hospital Node & Network Parameters ", style="Section.TLabelframe", padding=15)
        param_box.pack(fill="x", pady=(0, 15))

        # Row 0: Hospital ID & Server Address
        ttk.Label(param_box, text="Hospital Alias ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.hospital_id_var = tk.StringVar(value="hospital_alpha")
        self.hospital_entry = ttk.Entry(param_box, textvariable=self.hospital_id_var, width=25)
        self.hospital_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(param_box, text="Central FL Server Address:").grid(row=0, column=2, sticky="w", padx=(20, 5), pady=5)
        self.server_addr_var = tk.StringVar(value="127.0.0.1:8080")
        self.server_entry = ttk.Entry(param_box, textvariable=self.server_addr_var, width=25)
        self.server_entry.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        # Row 1: CSV File Selector
        ttk.Label(param_box, text="Clinical Dataset CSV:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        default_csv = PROJECT_ROOT / "client" / "data" / "hospital_alpha_data.csv"
        self.csv_path_var = tk.StringVar(value=str(default_csv))
        self.csv_entry = ttk.Entry(param_box, textvariable=self.csv_path_var, width=50)
        self.csv_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        csv_btn = tk.Button(
            param_box,
            text="📂 Browse CSV",
            command=self.browse_csv,
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            font=("Helvetica", 9, "bold"),
            relief="flat",
            padx=10,
        )
        csv_btn.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        # Row 2: ECG File Selector
        ttk.Label(param_box, text="12-Lead ECG Signals (.npy):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        default_ecg = PROJECT_ROOT / "client" / "data" / "hospital_alpha_ecg.npy"
        self.ecg_path_var = tk.StringVar(value=str(default_ecg))
        self.ecg_entry = ttk.Entry(param_box, textvariable=self.ecg_path_var, width=50)
        self.ecg_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        ecg_btn = tk.Button(
            param_box,
            text="📂 Browse ECG",
            command=self.browse_ecg,
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            font=("Helvetica", 9, "bold"),
            relief="flat",
            padx=10,
        )
        ecg_btn.grid(row=2, column=3, sticky="w", padx=5, pady=5)

        # Row 3: Auto-start local FL Server checkbox option
        self.auto_start_var = tk.BooleanVar(value=True)
        auto_chk = tk.Checkbutton(
            param_box,
            text="Auto-start local FL Server on port 8080 if not running (Simulation Mode)",
            variable=self.auto_start_var,
            bg="#1e293b",
            fg="#38bdf8",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#38bdf8",
            font=("Helvetica", 9),
        )
        auto_chk.grid(row=3, column=0, columnspan=4, sticky="w", padx=5, pady=5)

        # Action Buttons Frame
        btn_frame = ttk.Frame(content_frame)
        btn_frame.pack(fill="x", pady=(0, 15))

        self.start_btn = tk.Button(
            btn_frame,
            text="🚀 Start Local Training & Connect to FL Server",
            command=self.start_training,
            bg="#10b981",
            fg="#ffffff",
            activebackground="#059669",
            font=("Helvetica", 11, "bold"),
            relief="flat",
            pady=8,
            padx=20,
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = tk.Button(
            btn_frame,
            text="🛑 Stop FL Client Node",
            command=self.stop_training,
            bg="#ef4444",
            fg="#ffffff",
            activebackground="#dc2626",
            font=("Helvetica", 11, "bold"),
            relief="flat",
            pady=8,
            padx=20,
            state="disabled",
        )
        self.stop_btn.pack(side="left")

        # Terminal Execution Log Window
        log_box = ttk.LabelFrame(content_frame, text=" 🖥️ Live Training & PII Scrubbing Console ", style="Section.TLabelframe", padding=10)
        log_box.pack(fill="both", expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_box,
            wrap="word",
            bg="#020617",
            fg="#38bdf8",
            insertbackground="#ffffff",
            font=("Consolas", 10),
            height=15,
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.insert("1.0", "[System] MedShield FL Hospital Desktop App Ready.\n[Privacy] Local spaCy NER anonymizer initialized.\n[Ready] Select your dataset CSV and click 'Start Local Training'.\n\n")

    def browse_csv(self):
        path = filedialog.askopenfilename(title="Select Hospital Dataset CSV", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if path:
            self.csv_path_var.set(path)

    def browse_ecg(self):
        path = filedialog.askopenfilename(title="Select 12-Lead ECG Signals (.npy)", filetypes=[("Numpy Files", "*.npy"), ("All Files", "*.*")])
        if path:
            self.ecg_path_var.set(path)

    def log(self, message: str):
        self.log_queue.put(message)

    def poll_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.after(100, self.poll_log_queue)

    def start_training(self):
        csv_file = self.csv_path_var.get().strip()
        if not Path(csv_file).exists():
            messagebox.showerror("File Error", f"CSV file not found:\n{csv_file}")
            return

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        hospital_id = self.hospital_id_var.get().strip()
        server_addr = self.server_addr_var.get().strip()

        self.log("============================================================")
        self.log(f"[Start] Launching FL Client Node for '{hospital_id}'...")
        self.log(f"[Network] Target FL Aggregator Server: {server_addr}")
        self.log(f"[Data] Using Dataset: {csv_file}")
        self.log("============================================================\n")

        # Run process in background thread
        threading.Thread(target=self.run_fl_client, args=(hospital_id, server_addr, csv_file), daemon=True).start()

    def run_fl_client(self, hospital_id: str, server_addr: str, csv_file: str):
        python_bin = PROJECT_ROOT / "client" / ".venv" / "bin" / "python"
        if not python_bin.exists():
            python_bin = sys.executable

        # Parse host and port
        if ":" in server_addr:
            host, port_str = server_addr.split(":", 1)
            port = int(port_str)
        else:
            host, port = server_addr, 8080

        # Check if server port is open
        if not is_port_open(host, port):
            if self.auto_start_var.get() and host in ["127.0.0.1", "localhost"]:
                self.log(f"[System] FL Server at {server_addr} is not running. Auto-starting local Flower Server...")
                server_cmd = [
                    str(python_bin),
                    str(PROJECT_ROOT / "server" / "app" / "features" / "federation" / "fl_server.py"),
                    "--port",
                    str(port),
                    "--rounds",
                    "3",
                ]
                env = os.environ.copy()
                env["PYTHONPATH"] = f".:{str(PROJECT_ROOT / 'server')}"
                try:
                    self.server_process = subprocess.Popen(server_cmd, cwd=str(PROJECT_ROOT), env=env)
                    time.sleep(2.0)
                    self.log("[System] Local Flower FL Server successfully started in background!\n")
                except Exception as err:
                    self.log(f"[Server Error] Failed to auto-start FL server: {err}")
            else:
                self.log(f"\n[Connection Error] Could not connect to FL Server at {server_addr}.")
                self.log("👉 Please start the FL server first or enable 'Auto-start local FL Server' checkbox.")
                self.after(0, self.reset_buttons)
                return

        cmd = [
            str(python_bin),
            str(PROJECT_ROOT / "client" / "fl_client.py"),
            "--server",
            server_addr,
            "--hospital-id",
            hospital_id,
            "--csv-file",
            csv_file,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in self.process.stdout:
                self.log(line.strip())

            self.process.wait()
            self.log(f"\n[Finished] Process exited with code {self.process.returncode}")
        except Exception as e:
            self.log(f"\n[Execution Error] {str(e)}")
        finally:
            self.process = None
            self.after(0, self.reset_buttons)

    def reset_buttons(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def stop_training(self):
        if self.process:
            self.process.terminate()
            self.log("\n[User Action] Termination signal sent to FL Client Node process.")
        if self.server_process:
            self.server_process.terminate()
            self.log("[User Action] Stopped local FL server process.")
        self.stop_btn.config(state="disabled")


if __name__ == "__main__":
    app = MedShieldClientApp()
    app.mainloop()
