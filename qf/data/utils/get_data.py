"""Get historical financial data and add technical indicators.

This module provides a function to download historical financial data and add technical indicators.
It supports different data imputation methods and allows for reindexing the data to a full date range.
"""

from typing import List, Union

import pandas as pd

from qf import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_IMPUTATION_METHOD,
    DEFAULT_DOWNLOADER,
    DEFAULT_INDICATORS,
    DEFAULT_INTERVAL,
    DEFAULT_N_TRADING_DAYS,
    VERBOSITY,
)

from .clean_data import drop_columns
from .data_manager import DataManager
from .preprocessor import add_technical_indicators


def get_data(
    tickers: Union[str, List[str]],
    start: str,
    end: str,
    indicators: List[str] | str = DEFAULT_INDICATORS,
    verbosity: int = VERBOSITY,
    cache_dir: str = DEFAULT_CACHE_DIR,
    downloader: str = DEFAULT_DOWNLOADER,
    n_trading_days: int = DEFAULT_N_TRADING_DAYS,
    imputation_method: str = DEFAULT_DATA_IMPUTATION_METHOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Downloads historical financial data and adds technical indicators.

    Args:
        tickers: Single ticker or list of tickers
        start: Start date
        end: End date
        indicators: List of technical indicators to add
        verbosity: Verbosity level
        cache_dir: Directory to store cached data
        downloader: Data downloader to use
        n_trading_days: Number of trading days to use (252 or 365)
        imputation_method: Method to handle missing values ('bfill', 'shrinkage', 'removal', 'keep_nan')
        interval: Data interval (e.g., '1d', '1wk')

    Returns:
        pd.DataFrame: Multi-index DataFrame with tickers and OHLCV data
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # Set default indicators if none are provided
    if indicators is None:
        indicators = DEFAULT_INDICATORS

    # Initialize data manager and load data
    data_manager = DataManager(
        cache_dir=cache_dir,
        interval=interval,
        downloader=downloader,
        verbosity=verbosity,
    )
    data = data_manager.get_data(tickers, start, end)

    # If data is empty, return it without further processing
    if data.empty:
        if verbosity > 0:
            print("No data available, returning empty DataFrame")
        return data

    # Convert dates to datetime if they are strings
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # If we are using 365 trading days, reindex to full calendar days
    if n_trading_days == 365:
        # Get full date range
        full_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
        # Reindex to full calendar days, forward filling missing values
        data = data.reindex(full_range).ffill()
    elif n_trading_days != 252:
        raise ValueError("n_trading_days must be either 252 or 365.")

    # Drop the 'Adj Close' column if it exists
    if "Adj Close" in data.columns.get_level_values(1):
        adj_close_tickers = set(t for t, field in data.columns if field == "Adj Close")
        data = data.drop(
            columns=[(ticker, "Adj Close") for ticker in adj_close_tickers]
        )

    # Handle missing values based on imputation method
    if imputation_method == "bfill":
        data = data.bfill()
    elif imputation_method == "shrinkage":
        # In shrinkage we cut the rows where any data is missing
        data = data.dropna()
    elif imputation_method == "removal":
        # If we find a ticker that has missing values, we remove it from the dataframe
        data = data.dropna(axis=1, how="any")
    elif imputation_method == "keep_nan":
        # Keep NaN values as is
        pass
    else:
        raise ValueError(
            f"Unknown imputation method: {imputation_method}. Use 'bfill', 'shrinkage', 'removal', or 'keep_nan'."
        )

    # Add the technical indicators to the data
    data = add_technical_indicators(data, indicators=indicators, verbosity=verbosity)

    # Remove the columns that are not 'Close' or the specified indicators
    data = drop_columns(data, indicators)

    # Ensure that close is in the dataframe
    if "Close" not in data.columns.get_level_values(1):
        raise ValueError("The 'Close' column is missing from the data.")

    return data
