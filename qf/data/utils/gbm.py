import numpy as np
import pandas as pd
from typing import Optional


def generate_geometric_brownian_motion(
    S0: float = 1000,
    mu: float = 0.40,
    sigma: float = 0.02,
    T: float = 1,
    N: int = 252,
    seed: Optional[int] = 42,
    ticker: str = "GBM",
) -> pd.DataFrame:
    """
    Generate a univariate Geometric Brownian Motion price series.

    Parameters
    ----------
    S0 : float
        Initial price.
    mu : float
        Drift coefficient.
    sigma : float
        Volatility coefficient.
    T : float
        Total time (in years).
    N : int
        Number of time steps.
    seed : int, optional
        Random seed for reproducibility.
    ticker : str
        Name for the ticker in the MultiIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns (ticker, 'Close') and DatetimeIndex.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    df = pd.DataFrame(S, columns=["Close"])
    df.index = pd.date_range(start="2024-01-01", periods=N, freq="D")
    df.columns = pd.MultiIndex.from_product([[ticker], ["Close"]])
    return df


def generate_multivariate_geometric_brownian_motion(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    T: float = 1,
    N: int = 252,
    seed: Optional[int] = 42,
    tickers: Optional[list] = None,
) -> pd.DataFrame:
    """
    Generate a multivariate Geometric Brownian Motion price series.

    Parameters
    ----------
    S0 : np.ndarray
        Initial prices, shape (d,).
    mu : np.ndarray
        Drift vector, shape (d,).
    sigma : np.ndarray
        Covariance matrix, shape (d, d).
    T : float
        Total time (in years).
    N : int
        Number of time steps.
    seed : int, optional
        Random seed for reproducibility.
    tickers : list, optional
        List of ticker names for the MultiIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns (ticker, 'Close') and DatetimeIndex.
    """
    d = len(S0)
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N)
    # Generate correlated Brownian increments
    Z = np.random.standard_normal(size=(N, d))
    L = np.linalg.cholesky(sigma)
    W = np.cumsum(Z @ L, axis=0) * np.sqrt(dt)
    X = (mu - 0.5 * np.diag(sigma)) * t[:, None] + W
    S = S0 * np.exp(X)
    if tickers is None:
        tickers = [f"GBM_{i+1}" for i in range(d)]
    arrays = [tickers, ["Close"] * d]
    columns = pd.MultiIndex.from_arrays(arrays)
    df = pd.DataFrame(S, columns=columns)
    df.index = pd.date_range(start="2025-01-01", periods=N, freq="D")
    return df
