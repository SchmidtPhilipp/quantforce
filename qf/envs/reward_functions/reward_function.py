from abc import ABC, abstractmethod

import torch

from qf.envs.portfolio.portfolio import Portfolio
from qf.envs.reward_functions.config.reward_function_config import BaseRewardConfig


class RewardFunction(ABC):
    """Base class for all reward functions."""

    def __init__(
        self,
        config: BaseRewardConfig,
        n_agents: int,
        device: str,
    ):
        self.config = config
        self.n_agents = n_agents
        self.device = device

    @abstractmethod
    def calculate(
        self,
        current_portfolio: Portfolio,
        previous_portfolio: Portfolio,
        transaction_costs: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate the reward for the current step."""
        pass

    def reset(self):
        """Reset the reward function state. Override in subclasses if needed."""
        pass
