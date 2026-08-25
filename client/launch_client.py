#!/usr/bin/env python3
"""Interactive Client Launcher for MedShield FL."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def print_banner():
    print("=" * 60)
    print("      🛡️   MedShield FL — Client Node Setup Wizard  🛡️")
    print("=" * 60)
    print(" Privacy-Preserving Multimodal Federated Learning Framework\n")


def get_input(prompt: str, default: str) -> str:
    user_val = input(f"👉 {prompt} [{default}]: ").strip()
    return user_val if user_val else default


def interactive_wizard():
    print_banner()

    print("Step 1: Hospital Node Details")
    hospital_id = get_input("Enter Hospital Code/Identifier", "hospital_alpha")
    print()

    print("Step 2: Select Local Datasets")
    default_csv = f"client/data/{hospital_id}_data.csv"
    if not Path(default_csv).exists():
        default_csv = "client/data/hospital_alpha_data.csv"

    csv_path = get_input("Enter path to Hospital Dataset CSV", default_csv)
    while not Path(csv_path).exists():
        print(f"   ⚠️ File not found: '{csv_path}'")
        csv_path = get_input("Please re-enter valid path to Dataset CSV", default_csv)

    default_ecg = f"client/data/{hospital_id}_ecg.npy"
    if not Path(default_ecg).exists():
        default_ecg = "client/data/hospital_alpha_ecg.npy"

    ecg_path = get_input("Enter path to ECG Signals (.npy)", default_ecg)
    print(f"   ✅ Dataset verified: {csv_path}\n")

    print("Step 3: Choose Operation Mode")
    print("  [1] Connect to Central Federated Learning Server (Participatory FL)")
    print("  [2] Run Multi-Hospital FL Training Simulation (Server + Clients)")
    print("  [3] Train Multimodal Model Locally (Offline Standalone)")
    
    choice = input("\n👉 Select Mode (1-3) [1]: ").strip()
    if not choice:
        choice = "1"

    if choice == "1":
        server_addr = get_input("Enter FL Central Server Address", "127.0.0.1:8080")
        print("\n🚀 Starting Federated Client Node...")
        os.system(
            f"PYTHONPATH=. uv run python client/fl_client.py "
            f"--server {server_addr} --hospital-id {hospital_id} --csv-file {csv_path}"
        )

    elif choice == "2":
        rounds = get_input("Enter number of FL training rounds", "3")
        print("\n🚀 Launching Multi-Hospital FL Simulation...")
        os.system(f"PYTHONPATH=. uv run python run_fl_simulation.py --rounds {rounds}")

    elif choice == "3":
        epochs = get_input("Enter local training epochs", "5")
        output_weights = get_input("Enter output weights file path", "client/ml_models/saved_weights/medshield_model.pt")
        print("\n🚀 Starting Local Multimodal Model Training...")
        os.system(
            f"PYTHONPATH=. uv run python client/train_multimodal.py "
            f"--csv {csv_path} --ecg {ecg_path} --epochs {epochs} --output {output_weights}"
        )
    else:
        print("Invalid option selected. Exiting wizard.")


if __name__ == "__main__":
    try:
        interactive_wizard()
    except KeyboardInterrupt:
        print("\n\nWizard cancelled by user. Goodbye!")
