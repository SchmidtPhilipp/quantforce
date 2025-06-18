from abc import ABC, abstractmethod

import torch

from qf.utils.logging_config import get_logger

logger = get_logger(__name__)

from qf import DEBUG_VERBOSITY, ERROR_VERBOSITY, INFO_VERBOSITY, WARNING_VERBOSITY


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
    """Sharpe ratio reward function with flexible past and future windows."""

    def __init__(
        self,
        n_agents: int,
        device: str,
        past_window: int = 20,
        future_window: int = 0,
        reward_scaling: float = 1.0,
        bad_reward: float = -1.0,
        verbosity: bool = 1,
    ):
        super().__init__(n_agents, device, reward_scaling, bad_reward)
        self.past_window = past_window
        self.future_window = future_window
        self.total_window = past_window + future_window
        self.returns_buffer = None
        self.current_step = 0
        self.verbosity = verbosity

    def reset(self):
        """Reset the reward function state."""
        self.returns_buffer = None
        self.current_step = 0

    def calculate(
        self,
        current_portfolio_value: torch.Tensor,
        old_portfolio_value: torch.Tensor,
        portfolio_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the Sharpe ratio using a flexible window that can look both into the past and future.

        Args:
            current_portfolio_value: Current portfolio value for each agent
            old_portfolio_value: Previous portfolio value for each agent
            portfolio_matrix: Current portfolio allocation matrix

        Returns:
            torch.Tensor: Sharpe ratio for each agent
        """
        # Calculate current return
        current_return = (current_portfolio_value / (old_portfolio_value + 1e-8)) - 1

        # Initialize or update returns buffer
        if self.returns_buffer is None:
            self.returns_buffer = torch.zeros(
                (self.n_agents, self.total_window),
                dtype=torch.float32,
                device=self.device,
            )

        # Shift buffer and add new return
        self.returns_buffer = torch.roll(self.returns_buffer, shifts=-1, dims=1)
        self.returns_buffer[:, -1] = current_return

        # Print debug information about window positions
        if self.verbosity > DEBUG_VERBOSITY and self.current_step >= self.past_window:
            logger.debug(f"\nWindow Debug Info:")
            logger.debug(f"Current step: {self.current_step}")
            logger.debug(
                f"Past window: {self.current_step - self.past_window + 1} to {self.current_step}"
            )
            if self.future_window > 0 and self.current_step >= self.total_window:
                logger.debug(
                    f"Future window: {self.current_step + 1} to {self.current_step + self.future_window}"
                )
            logger.debug(f"Total window size: {self.total_window}")
            logger.debug(f"Buffer shape: {self.returns_buffer.shape}")

        # Only calculate Sharpe ratio if we have enough data
        if self.current_step >= self.past_window:
            # Calculate mean and std over the past window
            past_returns = self.returns_buffer[:, : self.past_window]
            mean_return = torch.mean(past_returns, dim=1)
            std_return = torch.std(past_returns, dim=1)

            # Calculate Sharpe ratio
            sharpe_ratio = mean_return / (std_return + 1e-8)

            # If we have future data, incorporate it into the calculation
            if self.future_window > 0 and self.current_step >= self.total_window:
                future_returns = self.returns_buffer[:, self.past_window :]
                future_mean = torch.mean(future_returns, dim=1)
                future_std = torch.std(future_returns, dim=1)

                # Combine past and future metrics
                combined_mean = (mean_return + future_mean) / 2
                combined_std = torch.sqrt((std_return**2 + future_std**2) / 2)

                # Calculate combined Sharpe ratio
                sharpe_ratio = combined_mean / (combined_std + 1e-8)

                # Print additional debug info for combined calculation
                if self.verbosity > DEBUG_VERBOSITY:
                    logger.debug(
                        f"Past metrics - Mean: {mean_return.mean():.4f}, Std: {std_return.mean():.4f}"
                    )
                    logger.debug(
                        f"Future metrics - Mean: {future_mean.mean():.4f}, Std: {future_std.mean():.4f}"
                    )
                    logger.debug(
                        f"Combined metrics - Mean: {combined_mean.mean():.4f}, Std: {combined_std.mean():.4f}"
                    )
        else:
            # Return zeros if we don't have enough data
            sharpe_ratio = torch.zeros(
                self.n_agents, dtype=torch.float32, device=self.device
            )
            if self.verbosity > DEBUG_VERBOSITY:
                logger.debug(f"\nWindow Debug Info:")
                logger.debug(f"Current step: {self.current_step}")
                logger.debug(f"Waiting for enough data (need {self.past_window} steps)")

        self.current_step += 1
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
