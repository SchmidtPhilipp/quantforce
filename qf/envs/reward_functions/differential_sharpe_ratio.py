import torch

from qf.envs.portfolio.portfolio import Portfolio
from qf.envs.reward_functions.config.reward_function_config import (
    DifferentialSharpeRatioConfig,
)
from qf.envs.reward_functions.reward_function import RewardFunction


class DifferentialSharpeRatio(RewardFunction):
    """Differential Sharpe ratio reward function according to Moody & Saffell (1998)."""

    def __init__(
        self, config: DifferentialSharpeRatioConfig, n_agents: int, device: str
    ):
        super().__init__(config, n_agents, device)
        self.A = torch.zeros(
            n_agents, dtype=torch.float32, device=device
        )  # First moment
        self.B = torch.zeros(
            n_agents, dtype=torch.float32, device=device
        )  # Second moment

    def calculate(
        self,
        current_portfolio: Portfolio,
        previous_portfolio: Portfolio,
        transaction_costs: torch.Tensor = None,
    ) -> torch.Tensor:

        # Calculate return for this step
        returns = (current_portfolio.value / (previous_portfolio.value + 1e-8)) - 1

        # Update differential Sharpe ratio components
        dA = returns - self.A
        self.A = self.A + self.config.eta * dA

        dB = returns**2 - self.B
        self.B = self.B + self.config.eta * dB

        # Calculate differential Sharpe ratio
        numerator = self.B * dA - 0.5 * self.A * dB
        denominator = (self.B - 0.5 * self.A**2) ** (3 / 2)

        # Avoid division by zero
        return (
            torch.where(
                denominator > 1e-8, numerator / denominator, torch.zeros_like(numerator)
            )
            * self.config.reward_scaling
        )

    def reset(self):
        """Reset the components for a new episode."""
        self.A = torch.zeros(self.n_agents, dtype=torch.float32, device=self.device)
        self.B = torch.zeros(self.n_agents, dtype=torch.float32, device=self.device)
