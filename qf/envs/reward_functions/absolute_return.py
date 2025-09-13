import torch

from qf.envs.portfolio.portfolio import Portfolio
from qf.envs.reward_functions.config.reward_function_config import AbsoluteReturnConfig
from qf.envs.reward_functions.reward_function import RewardFunction


class AbsoluteReturn(RewardFunction):
    """Absolute return reward function."""

    def __init__(self, config: AbsoluteReturnConfig, n_agents: int, device: str):
        super().__init__(config, n_agents, device)

    def calculate(
        self,
        current_portfolio: Portfolio,
        previous_portfolio: Portfolio,
        transaction_costs: torch.Tensor = None,
    ) -> torch.Tensor:
        return (
            current_portfolio.value - previous_portfolio.value
        ) * self.config.reward_scaling
