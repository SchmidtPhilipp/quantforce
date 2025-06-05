import numpy as np
import pandas as pd

class MultiAssetBrownianMotionLogReturn:
    @staticmethod
    def estimate_drift_and_covariance(prices: pd.DataFrame, delta_t: float):
        """
        Estimates the drift vector and covariance matrix using the Brownian motion model.

        Parameters:
            prices (pd.DataFrame): DataFrame of asset prices with shape (n_timesteps, n_assets).
            delta_t (float): Time interval between observations.

        Returns:
            tuple: (drift vector, covariance matrix)
        """
        # Compute log returns
        log_prices = np.log(prices)
        log_returns = log_prices.diff().dropna() / delta_t  # Shape: (n_timesteps - 1, n_assets)

        # Estimate drift vector
        drift = log_returns.mean(axis=0).values  # Shape: (n_assets,)

        # Estimate covariance matrix
        centered_log_returns = log_returns - drift * delta_t
        covariance = (centered_log_returns.T @ centered_log_returns) / (len(log_returns) * delta_t)  # Shape: (n_assets, n_assets)

        return drift, covariance

    @staticmethod
    def estimate_linear_return_expectation_and_covariance(drift: np.ndarray, covariance: np.ndarray, T: float):
        """
        Estimates the expectation and covariance of linear returns.

        Parameters:
            drift (np.ndarray): Drift vector with shape (n_assets,).
            covariance (np.ndarray): Covariance matrix with shape (n_assets, n_assets).
            T (float): Total time horizon.

        Returns:
            tuple: (expected linear return vector, covariance matrix of linear returns)
        """
        # Compute expected linear return vector
        m = drift * T + 0.5 * T * np.diag(covariance)
        expected_linear_return = np.exp(m) - 1  # Shape: (n_assets,)

        # Compute covariance matrix of linear returns
        a = np.exp(m)  # Shape: (n_assets,)
        exp_term = np.exp(np.outer(m, np.ones_like(m)) + np.outer(np.ones_like(m), m) + T * covariance)
        covariance_linear_return = exp_term - np.outer(a, a)  # Shape: (n_assets, n_assets)

        return expected_linear_return, covariance_linear_return