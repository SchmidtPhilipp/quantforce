"""
Least Squares Imputation Methods

This module provides Least Squares Imputation methods for financial time series data.
The methods include:

1. Linear LSE imputation
2. Log-linear LSE imputation
"""

import numpy as np
import pandas as pd


def lse_linear_impute_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply linear LSE imputation to each column of a DataFrame.
    """

    def impute_col(series: pd.Series) -> pd.Series:
        t = np.arange(len(series))
        mask = ~series.isna()
        t_known = t[mask]
        y_known = series[mask].values

        if len(y_known) < 2:
            return series  # not enough points to fit

        X = np.vstack([t_known, np.ones_like(t_known)]).T
        beta, _, _, _ = np.linalg.lstsq(X, y_known, rcond=None)

        y_hat = beta[0] * t + beta[1]
        series = series.copy()
        series[~mask] = y_hat[~mask]
        return series

    return df.apply(impute_col, axis=0)


def lse_loglinear_impute_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log-linear LSE imputation to each column of a DataFrame.
    """

    def impute_col(series: pd.Series) -> pd.Series:
        t = np.arange(len(series))
        mask = ~series.isna()
        y_known = series[mask].values
        t_known = t[mask]

        if len(y_known) < 2:
            return series  # not enough points to fit

        if np.any(y_known <= 0):
            raise ValueError(
                f"Log-linear requires positive values, found non-positive in column {series.name}"
            )

        log_y = np.log(y_known)
        X = np.vstack([t_known, np.ones_like(t_known)]).T
        beta, _, _, _ = np.linalg.lstsq(X, log_y, rcond=None)

        log_y_hat = beta[0] * t + beta[1]
        y_hat = np.exp(log_y_hat)

        series = series.copy()
        series[~mask] = y_hat[~mask]
        return series

    return df.apply(impute_col, axis=0)
