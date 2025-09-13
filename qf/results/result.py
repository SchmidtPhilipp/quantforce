from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import torch

from qf.utils.experiment_logger import ExperimentLogger


@dataclass
@dataclass
class Result:
    """
    Holds the results of a single step in the environment.
    """

    rewards: Optional[torch.Tensor] = None  # shape [n_agents]
    actions: Optional[torch.Tensor] = None  # shape [n_agents, n_assets + 1]
    asset_holdings: Optional[torch.Tensor] = None  # shape [n_agents, n_assets]
    actor_balance: Optional[torch.Tensor] = None  # shape [n_agents]
    balance: Optional[torch.Tensor] = None  # shape [1]
    date: Optional[torch.Tensor] = None  # shape [1]
    cash: Optional[torch.Tensor] = None  # shape [n_agents]

    def log(self, experiment_logger, run_type: str, tickers: List[str]):
        tickers_and_cash = tickers + ["cash"]

        for field in fields(self):
            key = field.name
            value = getattr(self, key)

            if value is None:
                continue

            if key in {"rewards", "actor_balance", "cash"}:
                for i_agent, agent_value in enumerate(value):
                    experiment_logger.log_scalar(
                        f"{run_type}_{key}/agent_{i_agent}",
                        (
                            agent_value.item()
                            if isinstance(agent_value, torch.Tensor)
                            else agent_value
                        ),
                    )

            elif key == "actions":
                for i_agent, agent_value in enumerate(value):
                    for i_asset, asset_value in enumerate(agent_value):
                        experiment_logger.log_scalar(
                            f"{run_type}_{key}/agent_{i_agent}/{tickers_and_cash[i_asset]}",
                            (
                                asset_value.item()
                                if isinstance(asset_value, torch.Tensor)
                                else asset_value
                            ),
                        )

            elif key == "asset_holdings":
                for i_agent, agent_value in enumerate(value):
                    for i_asset, asset_value in enumerate(agent_value):
                        experiment_logger.log_scalar(
                            f"{run_type}_{key}/agent_{i_agent}/{tickers[i_asset]}",
                            (
                                asset_value.item()
                                if isinstance(asset_value, torch.Tensor)
                                else asset_value
                            ),
                        )

            elif key in {"balance", "date"}:
                experiment_logger.log_scalar(
                    f"{run_type}_{key}",
                    value.item() if isinstance(value, torch.Tensor) else value,
                )

            else:
                # fallback for unexpected keys
                experiment_logger.log_scalar(
                    f"{run_type}_{key}",
                    value.item() if isinstance(value, torch.Tensor) else value,
                )
