from dataclasses import dataclass

import torch


@dataclass
class Portfolio:
    """
    Portfolio dataclass for multi-agent portfolio management.

    Attributes:
        cash (torch.Tensor): Current cash holdings for each agent (shape: [n_agents, 1]).
        weights (torch.Tensor): Current asset portfolio vector for each agent (shape: [n_agents, n_assets]).
        value (torch.Tensor): Current total portfolio value for each agent (shape: [n_agents, 1]).
    """

    cash: torch.Tensor  # shape: (n_agents, 1)
    weights: torch.Tensor  # shape: (n_agents, n_assets)
    value: torch.Tensor  # shape: (n_agents, 1)

    def __init__(
        self,
        n_agents: int,
        initial_balance: float,
        device: str,
        n_assets: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.n_agents = n_agents
        self.initial_balance = initial_balance
        self.device = device
        self.n_assets = n_assets
        self.dtype = dtype  # Store the dtype

        # Scale initial balance for smaller datatypes to avoid overflow
        if dtype == torch.float16:
            scaled_balance = initial_balance / 1000.0  # Scale down for float16
        elif dtype == torch.int8:
            scaled_balance = initial_balance / 10000.0  # Scale down for int8
        else:
            scaled_balance = initial_balance

        self.cash = torch.full(
            (self.n_agents, 1),
            scaled_balance / self.n_agents,
            dtype=self.dtype,  # Use the configurable dtype
            device=self.device,
        )
        self.weights = torch.zeros(
            (self.n_agents, self.n_assets),
            dtype=self.dtype,  # Use the configurable dtype
            device=self.device,
        )
        self.value = torch.full(
            (self.n_agents, 1),
            scaled_balance / self.n_agents,
            dtype=self.dtype,  # Use the configurable dtype
            device=self.device,
        )
