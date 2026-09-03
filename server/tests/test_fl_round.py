"""Automated verification test for multi-hospital Federated Learning round execution."""

import multiprocessing
import time

import flwr as fl
from app.features.federation.fl_server import start_fl_server
from fl_client import MedShieldFLClient, create_dummy_dataloaders


def _server_target():
    start_fl_server(rounds=2, port=8089, strategy_type="FedAvg", min_clients=2)


def _client_target(hospital_id: str):
    train_loader, val_loader = create_dummy_dataloaders(num_samples=100)
    client = MedShieldFLClient(
        hospital_id=hospital_id,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    fl.client.start_client(server_address="127.0.0.1:8089", client=client.to_client())


def test_fl_federated_round():
    # 1. Start FL Aggregator Server Process
    server_proc = multiprocessing.Process(target=_server_target)
    server_proc.start()
    time.sleep(1.5)

    # 2. Launch Hospital Alpha & Hospital Beta Client Processes
    client_alpha = multiprocessing.Process(
        target=_client_target, args=("hospital_alpha",)
    )
    client_beta = multiprocessing.Process(
        target=_client_target, args=("hospital_beta",)
    )

    client_alpha.start()
    client_beta.start()

    client_alpha.join(timeout=20)
    client_beta.join(timeout=20)

    if server_proc.is_alive():
        server_proc.terminate()
        server_proc.join()

    assert client_alpha.exitcode == 0
    assert client_beta.exitcode == 0
