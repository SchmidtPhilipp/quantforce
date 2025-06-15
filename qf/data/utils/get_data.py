"""Get historical financial data and add technical indicators.

This module provides a function to download historical financial data and add technical indicators.
It supports different data imputation methods and allows for reindexing the data to a full date range.
"""

from typing import List

from qf import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_IMPUTATION_METHOD,
    DEFAULT_DOWNLOADER,
    DEFAULT_INDICATORS,
    DEFAULT_N_TRADING_DAYS,
    VERBOSITY,
)

from .clean_data import drop_columns, reindex_data
from .load_data import load_data
from .preprocessor import add_technical_indicators


def get_data(
    tickers,
    start,
    end,
    indicators: List[str] | str = DEFAULT_INDICATORS,
    verbosity: int = VERBOSITY,
    cache_dir: str = DEFAULT_CACHE_DIR,
    downloader: str = DEFAULT_DOWNLOADER,
    n_trading_days: int = DEFAULT_N_TRADING_DAYS,
    imputation_method: str = DEFAULT_DATA_IMPUTATION_METHOD,
):
    """
    Downloads historical financial data and adds technical indicators.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # Set default indicators if none are provided
    if indicators is None:
        indicators = DEFAULT_INDICATORS

    # load the data either from cache or download it
    data = load_data(
        tickers,
        start,
        end,
        verbosity=verbosity,
        cache_dir=cache_dir,
        downloader=downloader,
    )

    # If we are using 365 trading days for the purpose of injecting other information at non trading days we reindex
    # the data to a dataframe over the full date range.
    if n_trading_days != 252 and n_trading_days != 365:
        raise ValueError("n_trading_days must be either 252 or 365.")

    if n_trading_days == 365:
        data = reindex_data(data, start, end)

    # If the frame still contains missing values throw an error
    # if data.isnull().values.any():
    # raise ValueError("The data contains missing values.")

    # We anyways use adjusted close prices, so we drop the 'Adj Close' column if it exists.
    if "Adj Close" in data.columns.get_level_values(1):
        adj_close_tickers = set(t for t, field in data.columns if field == "Adj Close")
        data = data.drop(
            columns=[(ticker, "Adj Close") for ticker in adj_close_tickers]
        )

    if imputation_method == "bfill":
        data = data.bfill()
    elif imputation_method == "shrinkage":
        # In shrinkage we cut the rows where any data is missing.
        data = data.dropna()
    elif imputation_method == "removal":
        # if we find a ticker that has missing values, we remove it from the dataframe
        data = data.dropna(axis=1, how="any")
    else:
        raise ValueError(
            f"Unknown imputation method: {imputation_method}. Use 'bfill' or 'shrinkage'."
        )

    # Add the technical indicators to the data
    data = add_technical_indicators(data, indicators=indicators, verbosity=verbosity)

    # remove the columns that are not 'Close' or the specified indicators
    data = drop_columns(data, indicators)

    # Ensure that close is in the dataframe
    if "Close" not in data.columns.get_level_values(1):
        raise ValueError("The 'Close' column is missing from the data.")

    # data = data.dropna()

    return data
