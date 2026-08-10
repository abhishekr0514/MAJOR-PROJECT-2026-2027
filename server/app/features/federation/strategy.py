"""Federated Learning Strategies (FedAvg and FedProx) for MedShield FL."""

from typing import Any

import flwr as fl
from flwr.server.strategy import FedAvg


class FedProx(FedAvg):
    """FedProx strategy implementation for non-IID hospital client data."""

    def __init__(
        self,
        proximal_mu: float = 0.01,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu


def get_fl_strategy(
    strategy_type: str = "FedAvg",
    min_clients: int = 2,
    proximal_mu: float = 0.01,
) -> fl.server.strategy.Strategy:
    """Configure and return the requested FL strategy for hospital weight aggregation."""
    kwargs = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": min_clients,
        "min_evaluate_clients": min_clients,
        "min_available_clients": min_clients,
    }

    if strategy_type.lower() == "fedprox":
        return FedProx(proximal_mu=proximal_mu, **kwargs)

    return FedAvg(**kwargs)
