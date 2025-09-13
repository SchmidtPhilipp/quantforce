import torch

from qf.envs.portfolio.portfolio import Portfolio
from qf.envs.reward_functions.config.reward_function_config import LogReturnConfig
from qf.envs.reward_functions.reward_function import RewardFunction


class LogReturn(RewardFunction):
    """Logarithmic return reward function."""

    def __init__(self, config: LogReturnConfig, n_agents: int, device: str):
        super().__init__(config, n_agents, device)

    def calculate(
        self,
        current_portfolio: Portfolio,
        previous_portfolio: Portfolio,
        transaction_costs: torch.Tensor = None,
    ) -> torch.Tensor:
        # Create a mask for negative portfolio values
        negative_mask = (previous_portfolio.value < 0) | (current_portfolio.value < 0)

        # Calculate log returns
        returns = torch.log(current_portfolio.value / (previous_portfolio.value + 1e-8))

        # Apply bad reward to negative portfolio values
        return (
            torch.where(negative_mask, self.config.bad_reward, returns)
            * self.config.reward_scaling
        )
