"""Central Flower FL Server Aggregator script."""

import argparse

import flwr as fl

from app.features.federation.strategy import get_fl_strategy


def start_fl_server(
    rounds: int = 5,
    port: int = 8080,
    strategy_type: str = "FedAvg",
    min_clients: int = 2,
) -> None:
    """Launch central Flower FL server aggregator."""
    strategy = get_fl_strategy(strategy_type=strategy_type, min_clients=min_clients)
    print(
        f"[MedShield FL Server] Starting server on 0.0.0.0:{port} "
        f"with strategy '{strategy_type}' for {rounds} rounds..."
    )
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedShield FL Central Server")
    parser.add_argument(
        "--rounds", type=int, default=5, help="Number of FL training rounds"
    )
    parser.add_argument("--port", type=int, default=8080, help="FL server port")
    parser.add_argument(
        "--strategy",
        type=str,
        default="FedAvg",
        choices=["FedAvg", "FedProx"],
        help="Aggregation strategy",
    )
    parser.add_argument(
        "--min-clients", type=int, default=2, help="Minimum connected clients"
    )
    args = parser.parse_args()

    start_fl_server(
        rounds=args.rounds,
        port=args.port,
        strategy_type=args.strategy,
        min_clients=args.min_clients,
    )
