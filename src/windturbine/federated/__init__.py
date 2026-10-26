"""In-process Flower federated training: F1 FedAvg and the F2-F4 strategies."""

from .fedavg import ClientUpdate, aggregate_updates, load_checkpoint, run_fedavg, run_preflight
from .smoke import build_synthetic_clients
from .strategies import (
    privacy_report,
    run_federated,
    run_strategy_preflight,
    secure_aggregate,
    strategy_options,
    summarize_runs,
)

__all__ = [
    "ClientUpdate",
    "aggregate_updates",
    "build_synthetic_clients",
    "load_checkpoint",
    "privacy_report",
    "run_fedavg",
    "run_federated",
    "run_preflight",
    "run_strategy_preflight",
    "secure_aggregate",
    "strategy_options",
    "summarize_runs",
]
