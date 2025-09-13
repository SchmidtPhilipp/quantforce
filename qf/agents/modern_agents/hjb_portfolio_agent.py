from typing import Optional

import pandas as pd
import torch

import qf
from qf.agents.agent import Agent
from qf.agents.config.modern_agents.hjb_portfolio_agent_config import (
    HJBPortfolioAgentConfig,
)


class HJBPortfolioAgent(Agent):
    """
    Hamilton-Jacobi-Bellman Portfolio Agent implementing optimal portfolio allocation
    based on continuous-time stochastic control theory.

    This agent solves the HJB equation for optimal portfolio weights under
    geometric Brownian motion asset dynamics with constant relative risk aversion (CRRA).
    """

    def __init__(self, env, config: Optional[HJBPortfolioAgentConfig] = None):
        self.config = config or HJBPortfolioAgentConfig.get_default_config()
        super().__init__(env, self.config)

        # Core HJB parameters
        self.risk_aversion = self.config.risk_aversion
        # Use environment risk-free rate if available, otherwise use config
        self.risk_free_rate = self.env.env_config.risk_free_rate
        self.time_horizon = self.config.time_horizon

        # Portfolio constraints
        self.allow_shorting = self.config.allow_shorting
        self.leverage_constraint = self.config.leverage_constraint
        self.min_weight = self.config.min_weight
        self.max_weight = self.config.max_weight

        # Cache for efficiency
        self._cached_mu = None
        self._cached_sigma = None
        self._cached_prices = None
        self._last_update_date = None

    def estimate_gbm_parameters(
        self, price_df: pd.DataFrame, dt: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized GBM parameter estimation using vectorized operations.
        """
        # Convert to torch tensors once
        prices_tensor = torch.tensor(price_df.values, dtype=torch.float32)

        # Vectorized log returns computation
        log_prices = torch.log(prices_tensor)
        log_returns = (log_prices[1:] - log_prices[:-1]) / dt  # shape (T-1, d)

        # Efficient parameter estimation
        r_mean = torch.mean(log_returns, dim=0)  # (d,)
        r_cov = torch.cov(log_returns.T) / dt  # (d, d)

        # Vectorized conversion to price parameters
        sigma_diag = torch.sqrt(torch.diag(r_cov))  # (d,)
        mu = r_mean + 0.5 * sigma_diag**2  # (d,)

        return mu, r_cov

    def _update_cache_if_needed(self, state):
        """Update cached parameters only if data has changed."""
        current_date = state.date

        if (
            self._last_update_date != current_date
            or self._cached_mu is None
            or self._cached_sigma is None
            or self._cached_prices is None
        ):

            # Update data config efficiently
            data_config = self.env.env_config.data_config
            data_config.end = current_date
            data_config.indicators = ["Close"]
            data_config.backfill_method = "shrinkage"

            # Get historical data
            self.historical_data = qf.get_data(data_config=data_config)

            # Compute and cache parameters
            self._cached_mu, self._cached_sigma = self.estimate_gbm_parameters(
                self.historical_data
            )
            self._cached_prices = torch.tensor(
                self.historical_data.iloc[-1].values, dtype=torch.float32
            )
            self._last_update_date = current_date

    def estimate_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached parameters or update if needed."""
        # This method is kept for compatibility but will be called from act()
        return self._cached_mu, self._cached_sigma

    def compute_optimal_weights(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute optimal portfolio weights using the analytical solution to the HJB equation.

        For CRRA utility under GBM dynamics, the optimal weights are:
        w* = (1/γ) * Σ^(-1) * (μ - r*1)

        Args:
            mu: Expected returns tensor of shape (n_assets,)
            sigma: Covariance matrix tensor of shape (n_assets, n_assets)

        Returns:
            Optimal portfolio weights of shape (n_assets,)
        """
        # Vectorized computation
        excess_returns = mu - self.risk_free_rate

        # Compute optimal risky weights (Merton's solution)
        try:
            sigma_inv = torch.linalg.inv(sigma)
        except torch.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            sigma_inv = torch.linalg.pinv(sigma)

        weights_risky = (1.0 / self.risk_aversion) * (sigma_inv @ excess_returns)

        return weights_risky

    def apply_constraints(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply portfolio constraints to the optimal weights."""
        # Vectorized constraint application
        if not self.allow_shorting:
            weights = torch.clamp(weights, min=0.0)

        # Apply individual weight constraints (max_weight always applies)
        weights = torch.clamp(weights, max=self.max_weight)

        # Apply leverage constraint
        weights_sum = weights.sum()
        if weights_sum > self.leverage_constraint:
            weights = weights * (self.leverage_constraint / weights_sum)

        return weights

    def train(self, total_timesteps: int = 1, use_tqdm: bool = True):
        pass

    def act(self, state):
        """
        Generate optimal portfolio action based on current state.

        Args:
            state: Environment state object

        Returns:
            Portfolio weights tensor of shape (1, n_assets + 1) including cash
        """
        # Efficient cache update
        self._update_cache_if_needed(state)

        # Get cached parameters efficiently
        mu, sigma = self._cached_mu, self._cached_sigma

        # Compute optimal weights
        weights_risky = self.compute_optimal_weights(mu, sigma)

        # Apply constraints
        weights_risky = self.apply_constraints(weights_risky)

        # Efficient weight normalization
        weights_sum = weights_risky.sum()
        if weights_sum > 1.0:
            weights_risky = weights_risky / weights_sum
            cash_weight = torch.tensor(0.0, dtype=torch.float32)
        else:
            cash_weight = 1.0 - weights_sum

        # Combine risky weights with cash weight
        weights = torch.cat([weights_risky, cash_weight.unsqueeze(0)], dim=0)

        # Return in correct format for environment
        return weights.unsqueeze(0)  # Shape: (1, n_assets + 1)

    def _save_impl(self, path):
        pass

    def _load_impl(self, path):
        pass
