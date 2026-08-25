"""Multi-Hospital Federated Learning Simulation Runner Script."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_fl_simulation(rounds: int = 3, port: int = 8080) -> None:
    """Launch FL Central Aggregator Server and 2 Hospital Client Nodes concurrently."""
    print("=" * 70)
    print(f"🚀 MedShield FL Multi-Hospital Simulation Starting ({rounds} rounds, port {port})")
    print("=" * 70)

    # Use virtual environment python executable
    venv_python = Path("client/.venv/bin/python")
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable

    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    env["VIRTUAL_ENV"] = "client/.venv"

    # 1. Start FL Central Server
    server_cmd = [
        python_cmd,
        "server/app/features/federation/fl_server.py",
        "--rounds",
        str(rounds),
        "--port",
        str(port),
        "--min-clients",
        "2",
    ]
    print(f"\n[1/3] Starting Central Aggregator Server on port {port}...")
    server_proc = subprocess.Popen(server_cmd, env=env)

    # Give server time to bind socket
    time.sleep(3)

    # 2. Start Hospital Alpha Client Node
    alpha_cmd = [
        python_cmd,
        "client/fl_client.py",
        "--hospital-id",
        "hospital_alpha",
        "--server",
        f"127.0.0.1:{port}",
    ]
    print("[2/3] Connecting Hospital Alpha Client Node...")
    alpha_proc = subprocess.Popen(alpha_cmd, env=env)

    # 3. Start Hospital Beta Client Node
    beta_cmd = [
        python_cmd,
        "client/fl_client.py",
        "--hospital-id",
        "hospital_beta",
        "--server",
        f"127.0.0.1:{port}",
    ]
    print("[3/3] Connecting Hospital Beta Client Node...")
    beta_proc = subprocess.Popen(beta_cmd, env=env)

    try:
        # Wait for server to finish all FL rounds
        server_proc.wait()
        print("\n✅ FL Central Aggregator Server completed all rounds successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Terminating FL simulation processes...")
    finally:
        # Cleanup client processes
        for proc, name in [(alpha_proc, "Hospital Alpha"), (beta_proc, "Hospital Beta")]:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(f"🛑 Stopped {name} client node.")

        if server_proc.poll() is None:
            server_proc.terminate()

    print("\n" + "=" * 70)
    print("🎉 MedShield Multi-Hospital FL Simulation Complete!")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedShield Multi-Hospital FL Simulation")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds to execute")
    parser.add_argument("--port", type=int, default=8080, help="FL server port")
    args = parser.parse_args()

    run_fl_simulation(rounds=args.rounds, port=args.port)


if __name__ == "__main__":
    main()
