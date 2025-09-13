from typing import Optional

import pandas as pd
import torch

import qf
from qf.agents.agent import Agent
from qf.agents.config.modern_agents.hjb_portfolio_agent_with_costs_config import (
    HJBPortfolioAgentWithCostsConfig,
)


class HJBPortfolioAgentWithCosts(Agent):
    def __init__(self, env, config: Optional[HJBPortfolioAgentWithCostsConfig] = None):
        self.config = config or HJBPortfolioAgentWithCostsConfig.get_default_config()
        super().__init__(env, config=self.config)

        self.risk_aversion = self.config.risk_aversion
        self.risk_free_rate = self.env.env_config.risk_free_rate
        self.proportional_cost = self.env.env_config.trade_cost_percent
        self.fixed_cost = self.env.env_config.trade_cost_fixed

        # Cache for efficiency
        self._cached_mu = None
        self._cached_sigma = None
        self._cached_prices = None
        self._last_update_date = None

        self.leverage_constraint = self.config.leverage_constraint
        self.min_weight = self.config.min_weight
        self.max_weight = self.config.max_weight

    def compute_merton_weights(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Compute optimal Merton weights without transaction costs."""
        excess_returns = mu - self.risk_free_rate
        sigma_inv = torch.linalg.pinv(sigma)
        weights_risky = (1.0 / self.risk_aversion) * sigma_inv @ excess_returns
        weights_risky = torch.clamp(weights_risky, min=0.0)
        return weights_risky

    def compute_expected_value(
        self, weights: torch.Tensor, mu: torch.Tensor, T: float, cash_position: float
    ) -> torch.Tensor:
        """Compute expected portfolio value under GBM dynamics."""
        # Vectorized computation
        risky_weights = weights[..., :-1]  # last is cash
        excess_returns = mu - self.risk_free_rate
        expected_growth = self.risk_free_rate + torch.dot(risky_weights, excess_returns)
        return cash_position * torch.exp(expected_growth * T)

    def compute_transaction_costs(
        self,
        w_old: torch.Tensor,
        w_new: torch.Tensor,
        prices: torch.Tensor,
        capital: float,
        kappa_prop: float,
        kappa_fix: float,
    ) -> torch.Tensor:
        """Compute transaction costs for portfolio rebalancing."""
        # Vectorized computation
        delta = torch.abs(w_new - w_old)[..., :-1]  # exclude cash dimension
        trade_volume = delta * prices
        cost_prop = kappa_prop * torch.sum(trade_volume)
        cost_fix = kappa_fix * torch.sum((delta > 1e-6).float())
        return cost_prop + cost_fix

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

    def train(self, total_timesteps: int = 1, use_tqdm: bool = True):
        pass

    def act(self, state):
        # Efficient cache update
        self._update_cache_if_needed(state)

        # Use cached parameters
        mu, sigma = self._cached_mu, self._cached_sigma
        prices = self._cached_prices
        w_old = state.actions.squeeze(0)  # shape (d+1,)
        capital = state.cash + torch.sum(state.portfolio * prices)

        T = self.config.time_horizon
        kappa_prop = self.proportional_cost
        kappa_fix = self.fixed_cost

        # Step 1: Compute candidate weights (Merton) - vectorized
        weights_risky = self.compute_merton_weights(mu, sigma)
        sum_risky = weights_risky.sum()

        # Efficient weight normalization
        if sum_risky > 1.0:
            weights_risky = weights_risky / sum_risky
            cash_weight = torch.tensor(0.0, dtype=torch.float32)
        else:
            cash_weight = 1.0 - sum_risky

        w_new = torch.cat([weights_risky, cash_weight.unsqueeze(0)])

        # Step 2: Compute expected values - vectorized
        EV_old = self.compute_expected_value(w_old, mu, T, capital)
        cost = self.compute_transaction_costs(
            w_old, w_new, prices, capital, kappa_prop, kappa_fix
        )
        EV_new = self.compute_expected_value(w_new, mu, T, capital - cost)

        if EV_new > EV_old:
            self.env.last_rebalancing_step = -1
            return w_new.unsqueeze(0)
        else:
            return w_old.unsqueeze(0)

    def _save_impl(self, path):
        pass

    def _load_impl(self, path):
        pass
