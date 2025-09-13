from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from pypfopt import expected_returns, risk_models

import qf

# Global cache for estimators
_estimator_cache = {}

_estimators: Dict[str, Callable] = {}


def register_estimator(name: str):
    def decorator(func):
        _estimators[name] = func
        return func

    return decorator


def estimate_expected_rate_of_returns_and_covariance(
    prices: pd.DataFrame, method: str = "sample_cov", **kwargs
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Optimized estimation with caching for repeated calls.
    """
    # Create cache key
    cache_key = f"{method}_{hash(prices.to_string())}_{hash(str(kwargs))}"

    if cache_key in _estimator_cache:
        return _estimator_cache[cache_key]

    if method not in _estimators:
        available = list(_estimators.keys())
        raise ValueError(f"Unknown method '{method}'. Available: {available}")

    # Compute and cache result
    result = _estimators[method](prices, **kwargs)
    _estimator_cache[cache_key] = result

    # Limit cache size to prevent memory issues
    if len(_estimator_cache) > 100:
        # Remove oldest entries
        oldest_key = next(iter(_estimator_cache))
        del _estimator_cache[oldest_key]

    return result


def list_available_methods() -> list:
    return list(_estimators.keys())


def clear_estimator_cache():
    """Clear the estimator cache to free memory."""
    global _estimator_cache
    _estimator_cache.clear()


# ============================================================================
# Built-in Estimation Methods (clean linear return interpretation)
# ============================================================================


def _expected_rate(prices: pd.DataFrame, frequency: int):
    return expected_returns.mean_historical_return(
        prices, frequency=frequency, log_returns=False, compounding=False
    )


@register_estimator("sample_cov")
def _sample_cov(prices: pd.DataFrame, frequency: int = 365, **kwargs):
    expected_rate_of_returns = _expected_rate(prices, frequency)
    return_covariance = risk_models.risk_matrix(
        prices, method="sample_cov", log_returns=False, frequency=frequency
    )
    return expected_rate_of_returns, return_covariance


@register_estimator("exp_cov")
def _exp_cov(prices: pd.DataFrame, frequency: int = 365, **kwargs):
    expected_rate_of_returns = _expected_rate(prices, frequency)
    return_covariance = risk_models.risk_matrix(
        prices, method="exp_cov", log_returns=False, frequency=frequency
    )
    return expected_rate_of_returns, return_covariance


@register_estimator("ledoit_wolf")
def _ledoit_wolf(prices: pd.DataFrame, frequency: int = 365, **kwargs):
    expected_rate_of_returns = _expected_rate(prices, frequency)
    return_covariance = risk_models.risk_matrix(
        prices, method="ledoit_wolf", log_returns=False, frequency=frequency
    )
    return expected_rate_of_returns, return_covariance


@register_estimator("ledoit_wolf_constant_variance")
def _ledoit_wolf_cv(prices: pd.DataFrame, frequency: int = 365, **kwargs):
    expected_rate_of_returns = _expected_rate(prices, frequency)
    return_covariance = risk_models.risk_matrix(
        prices,
        method="ledoit_wolf_constant_variance",
        log_returns=False,
        frequency=frequency,
    )
    return expected_rate_of_returns, return_covariance


@register_estimator("ledoit_wolf_single_factor")
def _ledoit_wolf_sf(prices: pd.DataFrame, frequency: int = 365, **kwargs):
    expected_rate_of_returns = _expected_rate(prices, frequency)
    return_covariance = risk_models.risk_matrix(
        prices,
        method="ledoit_wolf_single_factor",
        log_returns=False,
        frequency=frequency,
    )
    return expected_rate_of_returns, return_covariance


@register_estimator("ledoit_wolf_constant_correlation")
def _ledoit_wolf_cc(prices: pd.DataFrame, frequency: int = 365, **kwargs):
    expected_rate_of_returns = _expected_rate(prices, frequency)
    return_covariance = risk_models.risk_matrix(
        prices,
        method="ledoit_wolf_constant_correlation",
        log_returns=False,
        frequency=frequency,
    )
    return expected_rate_of_returns, return_covariance


@register_estimator("oracle_approximating")
def _oracle_approximating(prices: pd.DataFrame, frequency: int = 365, **kwargs):
    expected_rate_of_returns = _expected_rate(prices, frequency)
    return_covariance = risk_models.risk_matrix(
        prices, method="oracle_approximating", log_returns=False, frequency=frequency
    )
    return expected_rate_of_returns, return_covariance


@register_estimator("ML_brownian_motion_logreturn")
def _ml_brownian_motion(
    prices: pd.DataFrame,
    delta_t: str = None,
    T: float = 1.0,
    log_returns: bool = True,
    **kwargs,
):
    """ML estimation using Brownian motion model (log-return base, linearized)."""
    from qf.agents.classic_agents.estimators.ml_brownian_motion_logreturn import (
        MultiAssetBrownianMotionLogReturn,
    )

    if delta_t is None:
        raise ValueError("delta_t is required for ML_brownian_motion_logreturn")

    if delta_t.endswith("d"):
        delta_t_float = float(delta_t[:-1])
    elif delta_t.endswith("h"):
        delta_t_float = float(delta_t[:-1]) / 24.0
    elif delta_t.endswith("m"):
        delta_t_float = float(delta_t[:-1]) / (24.0 * 60.0)
    else:
        raise ValueError(
            f"Unsupported delta_t format: {delta_t}. Expected '1d', '1h', '1m'."
        )

    drift, covariance = MultiAssetBrownianMotionLogReturn.estimate_drift_and_covariance(
        prices, delta_t_float
    )

    # Convert torch tensors to pandas for consistency
    expected_rate_of_returns = pd.Series(drift.numpy(), index=prices.columns)
    return_covariance = pd.DataFrame(
        covariance.numpy(), index=prices.columns, columns=prices.columns
    )

    return expected_rate_of_returns, return_covariance


@register_estimator("stepwise_statistics")
def _stepwise_statistics(prices: pd.DataFrame, frequency: int = 365, **kwargs):
    expected_rate_of_returns = _expected_rate(prices, frequency)
    return_covariance = risk_models.risk_matrix(
        prices, method="sample_cov", log_returns=False, frequency=frequency
    )
    return expected_rate_of_returns, return_covariance


@register_estimator("gbm_direct")
def _gbm_direct(prices: pd.DataFrame, dt: float = 1.0 / 252, **kwargs):
    """Direct GBM parameter estimation from log returns using PyTorch."""

    # Convert to torch tensors
    prices_tensor = torch.tensor(prices.values, dtype=torch.float32)

    # Vectorized log returns computation
    log_prices = torch.log(prices_tensor)
    log_returns = (log_prices[1:] - log_prices[:-1]) / dt  # Shape (T-1, d)

    # Efficient parameter estimation
    r_mean = torch.mean(log_returns, dim=0)  # (d,)
    r_cov = torch.cov(log_returns.T) / dt  # (d, d)

    # Vectorized conversion to price parameters
    sigma_diag = torch.sqrt(torch.diag(r_cov))  # (d,)
    expected_rate_of_returns = r_mean + 0.5 * sigma_diag**2  # (d,)

    # Convert back to pandas for consistency with other estimators
    expected_rate_of_returns = pd.Series(
        expected_rate_of_returns.numpy(), index=prices.columns
    )
    return_covariance = pd.DataFrame(
        r_cov.numpy(), index=prices.columns, columns=prices.columns
    )

    return expected_rate_of_returns, return_covariance
