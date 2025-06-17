from abc import ABC, abstractmethod

import torch


class RewardFunction(ABC):
    """Base class for all reward functions."""

    def __init__(
        self,
        n_agents: int,
        device: str,
        reward_scaling: float = 1.0,
        bad_reward: float = -1.0,
    ):
        self.n_agents = n_agents
        self.device = device
        self.reward_scaling = reward_scaling
        self.bad_reward = bad_reward

    @abstractmethod
    def calculate(
        self,
        current_portfolio_value: torch.Tensor,
        old_portfolio_value: torch.Tensor,
        portfolio_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the reward for the current step."""
        pass

    def apply_scaling_and_punishment(
        self, rewards: torch.Tensor, portfolio_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Apply reward scaling and punishment for zero allocations."""
        # Check for zero portfolio allocations
        zero_allocation_mask = torch.all(portfolio_matrix == 0, dim=1)
        # Apply punishment and scaling
        return torch.where(
            zero_allocation_mask, self.bad_reward, rewards * self.reward_scaling
        )


class LinearRateOfReturn(RewardFunction):
    """Linear rate of return reward function."""

    def calculate(
        self,
        current_portfolio_value: torch.Tensor,
        old_portfolio_value: torch.Tensor,
        portfolio_matrix: torch.Tensor,
    ) -> torch.Tensor:
        return (current_portfolio_value / (old_portfolio_value + 1e-8)) - 1


class LogReturn(RewardFunction):
    """Logarithmic return reward function."""

    def calculate(
        self,
        current_portfolio_value: torch.Tensor,
        old_portfolio_value: torch.Tensor,
        portfolio_matrix: torch.Tensor,
    ) -> torch.Tensor:
        # Create a mask for negative portfolio values
        negative_mask = (old_portfolio_value < 0) | (current_portfolio_value < 0)

        # Calculate log returns
        returns = torch.log(current_portfolio_value / (old_portfolio_value + 1e-8))

        # Apply bad reward to negative portfolio values
        return torch.where(negative_mask, self.bad_reward, returns)


class AbsoluteReturn(RewardFunction):
    """Absolute return reward function."""

    def calculate(
        self,
        current_portfolio_value: torch.Tensor,
        old_portfolio_value: torch.Tensor,
        portfolio_matrix: torch.Tensor,
    ) -> torch.Tensor:
        return current_portfolio_value - old_portfolio_value


class SharpeRatio(RewardFunction):
    """Sharpe ratio reward function with rolling window."""

    def __init__(
        self,
        n_agents: int,
        device: str,
        window_size: int,
        reward_scaling: float = 1.0,
        bad_reward: float = -1.0,
    ):
        super().__init__(n_agents, device, reward_scaling, bad_reward)
        self.window_size = window_size
        self.returns_history = torch.zeros(
            (n_agents, window_size), dtype=torch.float32, device=device
        )
        self.current_idx = 0

    def calculate(
        self,
        current_portfolio_value: torch.Tensor,
        old_portfolio_value: torch.Tensor,
        portfolio_matrix: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate current return
        current_return = (current_portfolio_value / (old_portfolio_value + 1e-8)) - 1

        # Update returns history
        self.returns_history[:, self.current_idx] = current_return
        self.current_idx = (self.current_idx + 1) % self.window_size

        # Calculate mean and std of returns
        mean_return = torch.mean(self.returns_history, dim=1)
        std_return = torch.std(self.returns_history, dim=1)

        # Calculate Sharpe ratio
        sharpe_ratio = mean_return / (std_return + 1e-8)

        return sharpe_ratio


class DifferentialSharpeRatio(RewardFunction):
    """Differential Sharpe ratio reward function according to Moody & Saffell (1998)."""

    def __init__(
        self,
        n_agents: int,
        device: str,
        reward_scaling: float = 1.0,
        bad_reward: float = -1.0,
        eta: float = 0.01,
    ):
        super().__init__(n_agents, device, reward_scaling, bad_reward)
        self.eta = eta
        self.A = torch.zeros(
            n_agents, dtype=torch.float32, device=device
        )  # First moment
        self.B = torch.zeros(
            n_agents, dtype=torch.float32, device=device
        )  # Second moment

    def calculate(
        self,
        current_portfolio_value: torch.Tensor,
        old_portfolio_value: torch.Tensor,
        portfolio_matrix: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate return for this step
        returns = (current_portfolio_value / (old_portfolio_value + 1e-8)) - 1

        # Update differential Sharpe ratio components
        dA = returns - self.A
        self.A = self.A + self.eta * dA

        dB = returns**2 - self.B
        self.B = self.B + self.eta * dB

        # Calculate differential Sharpe ratio
        numerator = self.B * dA - 0.5 * self.A * dB
        denominator = (self.B - 0.5 * self.A**2) ** (3 / 2)

        # Avoid division by zero
        return torch.where(
            denominator > 1e-8, numerator / denominator, torch.zeros_like(numerator)
        )

    def reset(self):
        """Reset the components for a new episode."""
        self.A = torch.zeros(self.n_agents, dtype=torch.float32, device=self.device)
        self.B = torch.zeros(self.n_agents, dtype=torch.float32, device=self.device)
