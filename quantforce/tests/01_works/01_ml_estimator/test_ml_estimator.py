import sys
import os

# Include ../../ to access the get_data and tickers modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


import numpy as np
import pandas as pd
from datetime import datetime
from data import get_data
from utils.plot import plot_lines_grayscale, plot_dual_axis

from data import DOWJONES
import numpy as np
import pandas as pd

def simulate_bm_prices(p0, drift, sigma, n_steps, dt, random_seed=None):
    """
    Simuliere Preisverläufe nach dem Brownian-Motion-Modell.
    p0: Startpreise (shape: [d])
    drift: Driftvektor (shape: [d])
    sigma: Kovarianzmatrix (shape: [d, d])
    n_steps: Anzahl Zeitschritte
    dt: Zeitschrittgröße
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    d = len(p0)
    prices = np.zeros((n_steps + 1, d))
    prices[0] = p0
    chol_sigma = np.linalg.cholesky(sigma)
    for t in range(1, n_steps + 1):
        # Ziehe Standardnormalvektor
        z = np.random.randn(d)
        # Brownsche Inkremente
        dW = np.sqrt(dt) * z
        prices[t] = prices[t-1] + drift * dt + chol_sigma @ dW
    return pd.DataFrame(prices, columns=[f"Asset{i+1}" for i in range(d)])

def estimate_bm_drift_and_cov(prices: pd.DataFrame, dt: float = 1.0):
    """
    Estimate drift vector, covariance matrix, mean vector and covariance matrix of returns
    under the Brownian motion model for multi-asset price data.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices, shape (n_steps, n_assets).
    dt : float
        Time step between observations (default: 1.0).

    Returns
    -------
    drift_hat : np.ndarray
        Estimated drift vector, shape (n_assets,).
    sigma_hat : np.ndarray
        Estimated covariance matrix of Brownian increments, shape (n_assets, n_assets).
    mean_vector : np.ndarray
        Estimated mean vector of returns over [0, T], shape (n_assets,).
    cov_returns : np.ndarray
        Estimated covariance matrix of returns over [0, T], shape (n_assets, n_assets).
    """
    prices = prices.values if isinstance(prices, pd.DataFrame) else np.asarray(prices)
    n, d = prices.shape
    T = (n - 1) * dt

    # Compute increments
    delta_prices = prices[1:] - prices[:-1]  # shape (n-1, d)
    n_increments = delta_prices.shape[0]

    # Drift estimator
    drift_hat = (prices[-1] - prices[0]) / T  # shape (d,)

    # Covariance estimator
    drift_term = drift_hat * dt  # shape (d,)
    centered_increments = delta_prices - drift_term  # shape (n-1, d)
    sigma_hat = (centered_increments.T @ centered_increments) / (n_increments * dt)  # shape (d, d)

    # Mean vector of returns
    mean_vector = (prices[-1] - prices[0]) / prices[0]  # shape (d,)

    # Covariance matrix of returns
    D_inv = np.diag(1.0 / prices[0])  # shape (d, d)
    cov_returns = T * D_inv @ sigma_hat @ D_inv  # shape (d, d)

    return drift_hat, sigma_hat, mean_vector, cov_returns

import pandas as pd
import numpy as np

def compute_covariance_from_prices(price_df: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    """
    Computes the covariance matrix of asset returns from a DataFrame of prices.

    Parameters:
    - price_df (pd.DataFrame): DataFrame of asset prices (T x d).
    - log_returns (bool): If True, use log returns; otherwise, use simple returns.

    Returns:
    - cov_df (pd.DataFrame): Covariance matrix as a DataFrame.
    """
    if not isinstance(price_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if price_df.shape[0] < 2:
        raise ValueError("At least two time steps are required to compute returns.")

    # Compute returns
    if log_returns:
        returns = np.log(price_df / price_df.shift(1)).dropna()
    else:
        returns = price_df.pct_change().dropna()

    # Compute covariance matrix
    cov_df = returns.cov()

    return np.array(cov_df)



if __name__ == "__main__":
    # prices: DataFrame mit shape (n_steps, n_assets)
    
    tickers = DOWJONES
    start = "2024-01-01"
    end = "2025-01-01"

    #prices = get_data(tickers, start, end, indicators=("Close",), verbosity=0)
    #drift, sigma, mean_ret, cov_ret = estimate_bm_drift_and_cov(prices)

    # Beispiel-Parameter
    n_steps = 5000
    dt = 1.0
    p0 = np.array([100, 120, 80])
    drift = np.array([0.2, -0.1, 0.05])
    sigma = np.array([[1.0, 0.3, 0.2],
                    [0.3, 2.0, 0.1],
                    [0.2, 0.1, 1.5]])

    # Simuliere Preise
    prices = simulate_bm_prices(p0, drift, sigma, n_steps, dt, random_seed=42)

    # Schätze die Parameter
    drift_hat, sigma_hat, mean_vector, cov_returns = estimate_bm_drift_and_cov(prices, dt=dt)

    cov_returns2 = compute_covariance_from_prices(prices, log_returns=False)

    print("Wahre Drift:", drift)
    print("Geschätzte Drift:", drift_hat)
    print("\nWahre Sigma:\n", sigma)
    print("Geschätzte Sigma:\n", sigma_hat)

    # Schätzungsfehler
    drift_error = np.linalg.norm(drift - drift_hat)
    sigma_error = np.linalg.norm(sigma - sigma_hat)

    print("\nDrift-Schätzungsfehler:", drift_error)
    print("Sigma-Schätzungsfehler:", sigma_error)


    print("#" * 50)
    print("Geschätzter Mittelwert der Returns:", mean_vector)
    print("Geschätzte Kovarianz der Returns:\n", cov_returns)
    
    #print("Geschätzte Kovarianz der Returns:\n", cov_returns2)






