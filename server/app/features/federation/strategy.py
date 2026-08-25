"""Federated Learning Strategies (FedAvg and FedProx) for MedShield FL with Model Checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

try:
    import torch
    from client.ml_models.full_model import MedShieldDiagnosticNet
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    MedShieldDiagnosticNet = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def save_aggregated_weights(parameters: Parameters, output_path: str = "client/ml_models/saved_weights/medshield_model.pt") -> None:
    """Save aggregated FL global model weights into PyTorch checkpoint file."""
    if not TORCH_AVAILABLE or MedShieldDiagnosticNet is None:
        return

    try:
        ndarray_weights = parameters_to_ndarrays(parameters)
        model = MedShieldDiagnosticNet()
        state_dict = model.state_dict()

        if len(ndarray_weights) == len(state_dict):
            new_state_dict = {}
            for (k, _), weight_arr in zip(state_dict.items(), ndarray_weights):
                new_state_dict[k] = torch.tensor(weight_arr)

            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(new_state_dict, out_file)
            print(f"✅ [FL Aggregator] Updated global model weights saved to '{out_file}'.")
    except Exception as e:
        print(f"⚠️ [FL Aggregator] Could not save aggregated checkpoint: {e}")


class MedShieldFedAvg(FedAvg):
    """Custom FedAvg Strategy with checkpointing of global model parameters."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"🌐 [FL Server Round {server_round}] Aggregated fit weights from {len(results)} hospital client nodes.")
            save_aggregated_weights(aggregated_parameters)

        return aggregated_parameters, metrics


class MedShieldFedProx(MedShieldFedAvg):
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
        return MedShieldFedProx(proximal_mu=proximal_mu, **kwargs)

    return MedShieldFedAvg(**kwargs)
