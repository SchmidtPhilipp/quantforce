import torch

from qf.envs.portfolio.portfolio import Portfolio
from qf.envs.reward_functions.config.reward_function_config import (
    CostAdjustedSharpeRatioConfig,
)
from qf.envs.reward_functions.reward_function import RewardFunction


class CostAdjustedSharpeRatio(RewardFunction):
    """
    Sharpe ratio reward function that accounts for transaction costs using a window approach.
    Similar to WindowSharpeRatio but calculates net returns by subtracting actual transaction costs.
    Supports both regular returns and log returns.
    """

    def __init__(
        self,
        config: CostAdjustedSharpeRatioConfig,
        n_agents: int,
        device: str,
        dataset,
    ):
        super().__init__(config, n_agents, device)
        self.dataset = dataset
        self.current_step = 0

    def reset(self):
        """Reset the reward function state."""
        self.current_step = 0

    def calculate(
        self,
        current_portfolio: Portfolio,
        previous_portfolio: Portfolio,
        transaction_costs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate cost-aware Sharpe ratio using windowed price data and accounting for transaction costs.

        Args:
            current_portfolio: Current portfolio for each agent
            previous_portfolio: Previous portfolio for each agent
            transaction_costs: Transaction costs for this step (if available)

        Returns:
            torch.Tensor: Cost-aware Sharpe ratio for each agent
        """
        adjusted_step = self.current_step - 1
        prices = torch.tensor(
            self.dataset.xs("Close", axis=1, level=1).values,
            dtype=torch.float32,
            device=self.device,
        )  # [T, n_assets]

        rewards = torch.zeros(self.n_agents, device=self.device)

        # Only calculate if we have enough data for both past and future windows
        if (adjusted_step >= self.config.past_window) and (
            adjusted_step + self.config.future_window < prices.shape[0]
        ):
            # Get windowed prices
            start_idx = adjusted_step - self.config.past_window
            end_idx = adjusted_step + self.config.future_window
            windowed_prices = prices[start_idx : end_idx + 1]

            # make a unit vector with a one at the current step
            unit_vector = torch.zeros(windowed_prices.shape[0] - 1, device=self.device)
            unit_vector[self.config.past_window] = 1

            # Calculate portfolio values over the window
            portfolio_values = (
                current_portfolio.weights @ windowed_prices.T
            )  # [n_agents, window_length]

            # Calculate returns (regular or log returns)
            if self.config.use_log_returns:
                raise NotImplementedError(
                    "Do not use log returns for this reward function"
                )
                # Log returns: ln(P_t / P_{t-1})
                returns = torch.log(
                    (
                        portfolio_values[:, 1:]
                        - torch.outer(transaction_costs, unit_vector)
                    )
                    / (portfolio_values[:, :-1] + 1e-8)
                )  # [n_agents, window_length-1]
            else:
                # Regular returns: (P_t / P_{t-1}) - 1
                returns = (
                    portfolio_values[:, 1:]
                    - torch.outer(transaction_costs, unit_vector)
                    - portfolio_values[:, :-1]
                ) / (
                    portfolio_values[:, :-1] + 1e-8
                )  # [n_agents, window_length-1]

            # Calculate Sharpe ratio
            mean_return = torch.mean(returns, dim=1)
            std_return = torch.std(returns, dim=1)
            sharpe_ratio = mean_return / (std_return + 1e-8)

            rewards = sharpe_ratio

        # Handle NaN or Inf values
        if torch.any(torch.isnan(rewards)) or torch.any(torch.isinf(rewards)):
            if self.config.verbosity > 0:
                print("⚠️  Cost-adjusted Sharpe ratio contains NaN or Inf")
            rewards = torch.nan_to_num(
                rewards,
                nan=self.config.bad_reward,
                posinf=self.config.bad_reward,
                neginf=self.config.bad_reward,
            )

        self.current_step += 1
        return rewards * self.config.reward_scaling
