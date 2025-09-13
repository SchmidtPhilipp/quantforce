import pandas as pd
import torch
import numpy as np


class MultiAssetBrownianMotionLogReturn:
    """
    Multi-asset Brownian motion model for log returns.

    This class provides methods for estimating drift and covariance parameters
    from historical price data under the assumption of geometric Brownian motion (GBM).
    """

    @staticmethod
    def estimate_drift_and_covariance(
        prices: pd.DataFrame, delta_t: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates the drift vector and covariance matrix using the Brownian motion model.

        Parameters:
            prices (pd.DataFrame): DataFrame of asset prices with shape (n_timesteps, n_assets).
            delta_t (float): Time interval between observations.

        Returns:
            tuple: (drift vector, covariance matrix) as torch.Tensor
        """
        # Convert to torch tensors once
        prices_tensor = torch.tensor(prices.values, dtype=torch.float32)

        # Vectorized log returns computation
        log_prices = torch.log(prices_tensor)
        log_returns = (
            log_prices[1:] - log_prices[:-1]
        ) / delta_t  # Shape: (n_timesteps - 1, n_assets)

        # Estimate drift vector
        drift = torch.mean(log_returns, dim=0)  # Shape: (n_assets,)

        # Estimate covariance matrix using the correct formula
        centered_log_returns = log_returns - drift * delta_t
        covariance = (centered_log_returns.T @ centered_log_returns) / (
            len(log_returns) * delta_t
        )  # Shape: (n_assets, n_assets)

        return drift, covariance

    @staticmethod
    def estimate_linear_return_expectation_and_covariance(
        drift: torch.Tensor, covariance: torch.Tensor, T: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates the expectation and covariance of linear returns.

        Parameters:
            drift (torch.Tensor): Drift vector with shape (n_assets,).
            covariance (torch.Tensor): Covariance matrix with shape (n_assets, n_assets).
            T (float): Total time horizon.

        Returns:
            tuple: (expected linear return vector, covariance matrix of linear returns) as torch.Tensor
        """
        # Compute expected linear return vector
        m = drift * T + 0.5 * T * torch.diag(covariance)
        expected_linear_return = torch.exp(m) - 1  # Shape: (n_assets,)

        # Compute covariance matrix of linear returns
        a = torch.exp(m)  # Shape: (n_assets,)
        exp_term = torch.exp(
            torch.outer(m, torch.ones_like(m))
            + torch.outer(torch.ones_like(m), m)
            + T * covariance
        )
        covariance_linear_return = exp_term - torch.outer(
            a, a
        )  # Shape: (n_assets, n_assets)

        return expected_linear_return, covariance_linear_return
